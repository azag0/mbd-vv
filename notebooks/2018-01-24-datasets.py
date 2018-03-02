
# coding: utf-8

# ## Intro

# In[ ]:


from pymbd import MBDCalc, from_volumes, ang, vdw_params, get_kgrid

from mbdvv.app import app, kcal, ev
from mbdvv.utils import last, listify, chunks
from mbdvv.physics import reduced_grad, terf, vv_pol, calc_vvpol

from scipy.special import erf
import numpy as np
import pandas as pd
from math import ceil
import os
from collections import OrderedDict
from itertools import product, islice
from functools import partial
from pkg_resources import resource_stream
from tqdm import tqdm
import re
pd.options.display.max_rows = 999

from matplotlib import pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# ### Common functions

# In[ ]:


def savefig(fig, name, ext='pdf', **kwargs):
    fig.savefig(f'../media/{name}.{ext}', transparent=True, bbox_inches='tight', **kwargs)
    
def ene_int(x, ds):
    key = x.iloc[0].name[:2]
    enes = x.reset_index('scale label'.split(), drop=True).ene.unstack('fragment')
    cluster = ds.clusters[key]
    try:
        enes_int = cluster.get_int_ene(enes)
    except KeyError:
        return np.nan*enes.iloc(1)[0]
    return enes_int

def ref_delta(x, ds):
    ref = ds.df.loc(0)[x.name[:2]].energy
    if ds.name == 'S66' and ref > 0:
        ref = np.nan
    ene = x.ene.reset_index('scale label'.split(), drop=True)
    delta = ene-ref
    reldelta = delta/abs(ref)
    return pd.DataFrame(OrderedDict({
        'ene': ene,
        'delta': ene-ref,
        'reldelta': (ene-ref)/abs(ref),
    }))

def ene_dft_vdw(x):
    ipbe = x.index == 'PBE'
    x = x.where(ipbe, lambda y: y + x['PBE'])
    x.index = x.index.where(ipbe, 'PBE+' + x.index)
    return x
    
def ds_stat(x):
    return pd.Series(OrderedDict({
        'N': len(x.dropna()),
        'MRE': x['reldelta'].mean(),
        'MARE': abs(x['reldelta']).mean(),
        'MdRE': x['reldelta'].median(),
        'MdARE': abs(x['reldelta']).median(),
        'SDRE': abs(x['reldelta']).std(),
        'ME': x['delta'].mean(),
        'MAE': abs(x['delta']).mean(),
    }))

def splice_key(df, indexes):
    return df.reset_index().assign(
        label=lambda x: x.key.map(lambda y: y[0]),
        scale=lambda x: x.key.map(lambda y: y[1]),
    ).drop('key', 1).set_index(['label', 'scale', *indexes])

def get_nk(lattice, density):
    rec_lattice = 2*np.pi*np.linalg.inv(lattice.T)
    rec_lens = np.sqrt((rec_lattice**2).sum(1))
    nkpts = np.ceil(rec_lens/(density/ang**2))
    return int(nkpts[0]), int(nkpts[1]), int(nkpts[2])


# ### MBD functions

# In[ ]:


class MBDException(Exception):
    pass


class NegativeEigs(MBDException):
    pass


class NegativeAlpha(MBDException):
    pass


def scaled_eigs(x):
    return np.where(x >= 0, x, -erf(np.sqrt(np.pi)/2*x**4)**(1/4))


def mbd_rsscs(mbd_calc, coords, alpha_0, C6, R_vdw, beta, lattice=None,
              k_grid=None, rpa=False, scs=False, scale_eigs=True, fortran=False):
    def _array(obj, *args, **kwargs):
        if obj is not None:
            return np.array(obj, *args, **kwargs)

    if fortran:
        return mbd_calc.mbd_energy(coords, alpha_0, C6, R_vdw, beta, lattice, k_grid)
    coords = _array(coords, dtype=float, order='F')
    alpha_0 = _array(alpha_0, dtype=float)
    C6 = _array(C6, dtype=float)
    R_vdw = _array(R_vdw, dtype=float)
    freq, freq_w = mbd_calc.omega_grid
    omega = 4./3*C6/alpha_0**2
    if not scs:
        alpha_0_rsscs = alpha_0
        C6_rsscs = C6
        R_vdw_rsscs = R_vdw
        omega_rsscs = omega
    else:
        alpha_dyn = alpha_0/(1+(freq[:, None]/omega)**2)
        alpha_dyn_rsscs = np.empty_like(alpha_dyn)
        for a, a_scr in zip(alpha_dyn, alpha_dyn_rsscs):
            sigma = (np.sqrt(2./np.pi)*a/3)**(1./3)
            a_nlc = np.linalg.inv(
                np.diag(np.repeat(1./a, 3)) + mbd_calc.dipole_matrix(
                    coords, 'fermi,dip,gg', sigma=sigma, R_vdw=R_vdw,
                    beta=beta, lattice=lattice,
                )
            )
            a_scr[:] = np.array([a_nlc[i::3, i::3].sum(1) for i in range(3)]).sum(0)/3
        alpha_0_rsscs = alpha_dyn_rsscs[0, :]
        if np.any(alpha_0_rsscs <= 0):
            raise NegativeAlpha(alpha_0_rsscs)
        C6_rsscs = 3./np.pi*np.sum(freq_w[:, None]*alpha_dyn_rsscs**2, 0)
        R_vdw_rsscs = R_vdw*(alpha_0_rsscs/alpha_0)**(1./3)
        omega_rsscs = 4./3*C6_rsscs/alpha_0_rsscs**2
    pre = np.repeat(omega_rsscs*np.sqrt(alpha_0_rsscs), 3)
    if lattice is None:
        k_grid = [None]
    else:
        assert k_grid is not None
        k_grid = get_kgrid(lattice, k_grid)
    ene = 0
    for k_point in k_grid:
        T = mbd_calc.dipole_matrix(
            coords, 'fermi,dip', R_vdw=R_vdw_rsscs, beta=beta,
            lattice=lattice, k_point=k_point
        )
        if rpa:
            for u, uw in zip(freq[1:], freq_w[1:]):
                A = np.diag(np.repeat(alpha_0_rsscs/(1+(u/omega_rsscs)**2), 3))
                eigs = np.linalg.eigvals(A@T)
                eigs = np.real(eigs)
                if scale_eigs:
                    eigs = scaled_eigs(eigs)
                if np.any(eigs <= -1):
                    raise NegativeEigs(k_point, u, eigs)
                if not scale_eigs:
                    log_eigs = np.log(1+eigs)
                else:
                    log_eigs = np.log(1+eigs)-eigs
                ene += 1/(2*np.pi)*np.sum(log_eigs)*uw
        else:
            eigs = np.linalg.eigvalsh(
                np.diag(np.repeat(omega_rsscs**2, 3))+np.outer(pre, pre)*T
            )
            if np.any(eigs < 0):
                raise NegativeEigs(k_point, eigs)
            ene += np.sum(np.sqrt(eigs))/2-3*np.sum(omega_rsscs)/2
    ene /= len(k_grid)
    return ene


# ### Polarizabilities

# In[ ]:


with app.context():
    _fname = app.get('s66')[0].loc(0)['Benzene ... AcOH', 1.0, 'fragment-1'].gridfile
    
pd.read_hdf(_fname).loc[lambda x: x.rho > 0].pipe(reduced_grad).loc[lambda x: x < 1].hist(bins=100); 


# In[ ]:


rgrad_cutoff = partial(terf, k=60, x0=0.07)

_fig, _ax = plt.subplots()
_x = np.arange(0, 1, 0.01)
_ax.plot(_x, rgrad_cutoff(_x));
_ax.grid(which='both')
_ax.set_xticks(np.arange(0, 1, 0.05), minor=True);


# In[ ]:


def bin_alpha_vv(df, bins):
    prefix = df.index.names
    if prefix == [None]:
        prefix = []
    subsums = (
        df
        .assign(
            binidx=lambda x: np.digitize(reduced_grad(x).clip(bins[0]+1e-10, bins[-1]-1e-10), bins),
        )
        .set_index('binidx', append=True)
        .pipe(calc_vvpol, 0, rgrad_cutoff).stack().reset_index(-1, drop=True)
        .groupby(prefix + ['binidx'])
        .apply(lambda x: pd.Series({'vv_pol': x.vvpol.sum(), 'vv_pol_nm': x.vvpol_nm.sum()}))
    )
    return subsums

def plot_binned(ax, bins, y):
    return ax.bar((bins[1:]+bins[:-1])/2, y, bins[1]-bins[0])


# In[ ]:


_bins = np.linspace(0, 1, 100)
_subsums = bin_alpha_vv(pd.read_hdf(_fname), _bins)
_fig, _ax = plt.subplots()
plot_binned(_ax, _bins, _subsums.vv_pol);


# In[ ]:


with app.context():
    _fname = app.get('x23')[0] .loc(0)['Benzene', 1.0, 'crystal'].gridfile
    
pd.read_hdf(_fname).loc[lambda x: x.rho > 0].pipe(reduced_grad).loc[lambda x: x < 1].hist(bins=100); 


# In[ ]:


_bins = np.linspace(0, 1, 100)
_subsums = bin_alpha_vv(pd.read_hdf(_fname), _bins)
_fig, _ax = plt.subplots()
plot_binned(_ax, _bins, _subsums.vv_pol);


# In[ ]:


with app.context():
    _df, _ds = app.get('solids')
_df = _df['solids']


# In[ ]:


solids_pts = pd.concat(
    dict(_df.gridfile.loc[:, 1., 'crystal'].apply(lambda x: pd.read_hdf(x))),
    names=('label', 'i_point')
)['i_atom part_weight rho rho_grad_norm'.split()]


# In[ ]:


def add_group(x, ds):
    return x.assign(group=ds.df.loc(0)[x.name, 1.].group)

(
    solids_pts
    .set_index('i_atom', append=True)
    .pipe(calc_vvpol, 0, rgrad_cutoff).stack().reset_index(-1, drop=True)
    .groupby('label i_atom'.split()).sum()
    .groupby('label').apply(add_group, _ds).set_index('group', append=True)
    .pipe(lambda x: x.vvpol_nm/x.vvpol)
    .groupby('group').describe()
)


# In[ ]:


def plot_hist_solids(axes, bins, subsums):
    binmids = (bins[1:]+bins[:-1])/2 
    width = bins[1]-bins[0]
    for ax, (label, df) in zip((ax for ax_row in axes for ax in ax_row), subsums.groupby('label')):
        rgrad = binmids[df.index.get_level_values('binidx')-1]
        ax.bar(rgrad, df.vv_pol, width, label='all')
        ax.bar(rgrad, df.vv_pol_nm, width, label='nonmetallic')
        ax.set_title(label)
        if label == 'Ag':
            ax.legend()
    
    
_bins = np.linspace(0, 1, 50)
_subsums = bin_alpha_vv(solids_pts.reset_index('i_point'), _bins)
_fig, _axes = plt.subplots(ceil(len(_subsums.index.levels[0])/4), 4, figsize=(8, 22))
plot_hist_solids(_axes, _bins, _subsums)
_fig.tight_layout()
savefig(_fig, 'alpha-rgrad-hists')


# In[ ]:


def get_atomic_quants():
    with app.context():
        dfs_dft, ds = app.get('solids')
    with MBDCalc(4) as mbd_calc:
        freq_w = mbd_calc.omega_grid[0].copy()
        alpha_vv = (
            solids_pts
            .set_index('i_atom', append=True)
            .pipe(calc_vvpol, freq_w, rgrad_cutoff)
            .groupby('label i_atom'.split()).sum()
        )
    df = []
    for (*key, fragment), data, _ in dfs_dft['solids'].loc(0)[:, 1.].itertuples():
        if fragment != 'crystal':
            continue
        species = listify(data['elems']['atom'])
        free_atoms = last(data['free_atoms'])
        species_idx = free_atoms['species']-1
        volume_ratios = last(data['volumes'])/free_atoms['volumes'][species_idx]
        alpha_0_free, C6_free, R_vdw_free = from_volumes(species, 1, kind='TS')
        alpha_vv_free = free_atoms['vv_pols'][:, species_idx]
        alpha_0_vv = last(data['vv_pols']).copy()[0]
        C6_vv_free = 3/np.pi*np.sum(last(data['omega_grid_w'])[:, None]*alpha_vv_free**2, 0)
        df.append(pd.DataFrame(OrderedDict({
            'label': key[0],
            'i_atom': list(range(1, len(species_idx)+1)),
            'species': species,
            'hirsh_ratios': volume_ratios,
            'alpha_0_free': alpha_0_free,
            'alpha_0_vvfree': alpha_vv_free[0],
            'alpha_0_hirsh': alpha_0_free*volume_ratios,
            'alpha_0_vv': alpha_0_vv,
            'alpha_0_vv2': alpha_vv.loc(0)[key[0]].vvpol.values.T[0],
            'alpha_0_vv_nm': alpha_vv.loc(0)[key[0]].vvpol_nm.values.T[0],
            'alpha_0_vvcorr': alpha_0_vv*alpha_0_free/alpha_vv_free[0],
            'alpha_0_vv_nmcorr': alpha_vv.loc(0)[key[0]].vvpol_nm.values.T[0]*alpha_0_free/alpha_vv_free[0],
            'C6_free': C6_free,
            'C6_hirsh': C6_free*volume_ratios**2,
            'C6_vvfree': 3/np.pi*np.sum(last(data['omega_grid_w'])[:, None]*alpha_vv_free**2, 0),
            'C6_vv': 3/np.pi*np.sum(freq_w[:, None]*alpha_vv.loc(0)[key[0]].vvpol.values.T**2, 0),
            'C6_vv_nm': 3/np.pi*np.sum(freq_w[:, None]*alpha_vv.loc(0)[key[0]].vvpol_nm.values.T**2, 0),
            'C6_vvcorr': 3/np.pi*np.sum(freq_w[:, None]*alpha_vv.loc(0)[key[0]].vvpol.values.T**2, 0)*C6_free/C6_vv_free,
            'C6_vv_nmcorr': 3/np.pi*np.sum(freq_w[:, None]*alpha_vv.loc(0)[key[0]].vvpol_nm.values.T**2, 0)*C6_free/C6_vv_free,
            'Rvdw_free': R_vdw_free,
            'Rvdw_17_free': 2.5*alpha_0_free**(1/7),
            'Rvdw_hirsh': R_vdw_free*volume_ratios**(1/3),
            'Rvdw_hirsh_17base': (2.5*alpha_0_free**(1/7))*volume_ratios**(1/3),
            'Rvdw_vv17': 2.5*alpha_0_vv**(1/7),
            'Rvdw_vvcorr17': 2.5*(alpha_0_vv*alpha_0_free/alpha_vv_free[0])**(1/7),
        })))
    return (
        pd
        .concat(df).set_index('label i_atom'.split())
        .assign(group=lambda x: ds.df.loc(0)[[(lbl, 1.) for lbl in x.index.get_level_values('label')]].group.values)
        .groupby('group').apply(lambda x: x.sort_index())
    )
        
_df = get_atomic_quants().round(2)
_df.to_csv('../results/solids-vdw-params.csv')


# In[ ]:


def get_atomic_quants():
    with app.context():
        df_dft, ds, alpha_vv = app.get('s66')
        alpha_vv = pd.read_hdf(alpha_vv['alpha.h5'].path)
    df = []
    for (*key, fragment), data, _ in df_dft.loc(0)[:, 1.].itertuples():
        if fragment != 'complex':
            continue
        df_alpha_vv = alpha_vv.loc(0)[(*key, fragment)]
        species = listify(data['elems']['atom'])
        free_atoms = last(data['free_atoms'])
        species_idx = free_atoms['species']-1
        volume_ratios = last(data['volumes'])/free_atoms['volumes'][species_idx]
        alpha_0_free, _, R_vdw_free = from_volumes(species, 1)
        alpha_vv_free = free_atoms['vv_pols'][0, species_idx]
        alpha_0_vv = last(data['vv_pols']).copy()[0]
        df.append(pd.DataFrame(OrderedDict({
            'label': key[0],
            'i_atom': list(range(1, len(species_idx)+1)),
            'species': species,
            'hirsh_ratios': volume_ratios,
            'vv_ratios': alpha_0_vv/alpha_vv_free,
            'alpha_0_free': alpha_0_free,
            'alpha_0_vv_free': alpha_vv_free,
            'alpha_0_hirsh': alpha_0_free*volume_ratios,
            'alpha_0_vv': alpha_0_vv,
            'alpha_0_vv[nm]': df_alpha_vv.vvpol_nm.values.T[0],
            'alpha_0_vvcorr': alpha_0_vv*alpha_0_free/alpha_vv_free,
            'Rvdw_free': R_vdw_free,
            'Rvdw_17_free': 2.5*alpha_0_free**(1/7),
            'Rvdw_hirsh': R_vdw_free*volume_ratios**(1/3),
            'Rvdw_vv17': 2.5*alpha_0_vv**(1/7),
            'Rvdw_vvcorr17': 2.5*(alpha_0_vv*alpha_0_free/alpha_vv_free)**(1/7),
        })))
    return (
        pd
        .concat(df).set_index('label i_atom'.split())
        .assign(group=lambda x: ds.df.loc(0)[[(lbl, 1.) for lbl in x.index.get_level_values('label')]].group.values)
        .groupby('group').apply(lambda x: x.sort_index())
    )
        
_df = get_atomic_quants().round(2)
_df.to_csv('../results/s66-vdw-params.csv')


# ## MBD energies

# In[ ]:


def mbd_from_data(calc, data, beta, vv_scale=None, vvpol=None,
                  corr=False, vdw17=False, vdw17base=False, vv_py=False, vdwvvscale=False,
                  vdw='TS', **kwargs):
    if vvpol is not None and vvpol != '' and 'alpha_vv' not in data:
        return np.nan
    coords = data['coords']['value'].T
    species = listify(data['elems']['atom'])
    lattice = data['lattice_vector']['value'].T if 'lattice_vector' in data else None
    volumes = last(data['volumes'])
    if vv_py:
        alpha_vv = data['alpha_vv'].vv_pol.values.T
        freq_w = calc.omega_grid[1]
    else:
        alpha_vv = last(data['vv_pols']).copy()
        freq_w = last(data['omega_grid_w'])
    free_atoms = last(data['free_atoms'])
    species_idx = free_atoms['species']-1
    volumes_free = free_atoms['volumes'][species_idx]
    alpha_vv_free = free_atoms['vv_pols'][:, species_idx]
    C6_vv_free = 3/np.pi*np.sum(last(data['omega_grid_w'])[:, None]*alpha_vv_free**2, 0)
    alpha_0_free = np.array([vdw_params.get(sp)[f'alpha_0({vdw})'] for sp in species])
    C6_free = np.array([vdw_params.get(sp)[f'C6({vdw})'] for sp in species])
    C6_vv = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv**2, 0)
    if vvpol is not None and vvpol != '':
        alpha_vv_nm = data['alpha_vv'].vvpol_nm.values.T
        C6_vv_nm = 3/np.pi*np.sum(calc.omega_grid[1][:, None]*alpha_vv_nm**2, 0)
        alpha_vv_mtl = data['alpha_vv'].vvpol.values.T-data['alpha_vv'].vvpol_nm.values.T
        C6_vv_mtl = 3/np.pi*np.sum(calc.omega_grid[1][:, None]*alpha_vv_mtl**2, 0)
    if not vv_scale:
        volume_scale = volumes/volumes_free
    else:
        volume_scale = (alpha_vv[0]/alpha_vv_free[0])**(1/vv_scale)
    alpha_0, C6, R_vdw = from_volumes(species, volume_scale, kind=vdw)
    if vdw17base:
        R_vdw = (2.5*alpha_0_free**(1/7))*volume_scale**(1/3)
    if vdwvvscale:
        R_vdw_free = np.array([vdw_params.get(sp)['R_vdw'] for sp in species])
        R_vdw = R_vdw_free*(alpha_vv[0]/alpha_vv_free[0])**(1/3)
    def correct_alpha(alpha_0, C6):
        if corr:
            alpha_0 = alpha_0*alpha_0_free/alpha_vv_free[0]
            C6 = C6*C6_free/C6_vv_free
        return alpha_0, C6
        
    if vvpol in ('', 'diff'):
        alpha_0, C6 = correct_alpha(alpha_vv[0], C6_vv)
    elif vvpol == 'nm':
        alpha_0, C6 = correct_alpha(alpha_vv_nm[0], C6_vv_nm)
    if vdw17:
        R_vdw = 2.5*alpha_0**(1/7)
    ene = mbd_rsscs(
        calc,
        coords,
        alpha_0, C6, R_vdw,
        beta,
        lattice=lattice,
        **kwargs
    )
    if vvpol != 'diff':
        return ene
    alpha_0, C6 = correct_alpha(alpha_vv_mtl[0], C6_vv_mtl)
    ene_mtl = mbd_rsscs(
        calc,
        coords,
        alpha_0, C6, R_vdw,
        beta,
        lattice=lattice,
        **kwargs
    )
    return ene-ene_mtl

def all_mbd_variants(calc, data, variants, kdensity=None):
    if 'lattice_vector' in data:
        if kdensity is None:
            k_grid = np.repeat(4, 3)
        else:
            k_grid = get_nk(data['lattice_vector']['value'], kdensity)
    else:
        k_grid = None
    enes = []
    for kwargs in variants:
        kwargs = kwargs.copy()
        beta = kwargs.pop('beta', 0.83)
        throw = kwargs.pop('throw', False)
        try:
            ene = mbd_from_data(calc, data, beta, k_grid=k_grid, **kwargs)
        except MBDException as e:
            if throw:
                raise e
            ene = np.nan
        enes.append(ene)
    return enes

def get_variant_label(flags):
    inner = []
    for k, v in flags.items():
        if k in ('rpa', 'scs', 'vdw17', 'corr', 'fortran', 'vdwvvscale') and v:
            inner.append(k)
        elif k in ('vv_scale', 'beta', 'vdw'):
            inner.append(f'{k}[{v}]')
        elif k == 'vvpol':
            inner.append('vvpol' + (f'[{v}]' if v else ''))
        elif k == 'vdw17base':
            inner.append('17base')
        elif k == 'scale_eigs' and not v:
            inner.append('noeigscale')
        elif k == 'throw':
            pass
        else:
            raise ValueError(k, v)
    return f'MBD({",".join(inner)})'


# ### Solids

# In[ ]:


def calculate_solids(variants):
    with app.context():
        dfs_dft, ds = app.get('solids')
    atom_enes = (
        dfs_dft['atoms'].set_index('conf', append='True').data
        .apply(lambda y: y['energy'][0]['value'][0] if y else None, 1)
        .unstack().min(1)
    )
    df = []
    with MBDCalc(4) as mbd_calc:
        alpha_vv = (
            solids_pts
            .set_index('i_atom', append=True)
            .pipe(calc_vvpol, mbd_calc.omega_grid[0], rgrad_cutoff)
            .groupby('label i_atom'.split()).sum()
        )
        for (*key, fragment), data, _ in tqdm(list(dfs_dft['solids'].loc(0)[:, 1.].itertuples())):
            if fragment == 'crystal':
                pbe_ene = data['energy'][0]['value'][0]
            else:
                pbe_ene = atom_enes[fragment]
            df.append((*key, fragment, 'PBE', pbe_ene))
            if fragment == 'crystal':
                try:
                    data = {**data, 'alpha_vv': alpha_vv.loc(0)[key[0]]}
                    enes = all_mbd_variants(mbd_calc, data, variants)
                except MBDException as e:
                    # this happens only if `variants` contains 'throw': True
                    print(*key, repr(e))
                    continue
            else:
                enes = [0. for _ in variants]
            enes = {get_variant_label(v): ene for v, ene in zip(variants, enes)}
            for mbd_label, ene in enes.items():
                df.append((*key, fragment, mbd_label, ene))
    df = pd.DataFrame(df, columns='label scale fragment method ene'.split())         .set_index('label scale fragment method'.split())
    return df, ds


# In[ ]:


_variants = [
    {'scs': True},
    {'rpa': True, 'scs': True},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': '', 'corr': True, 'beta': 0.8, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'beta': 0.8, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17base': True},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.82, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.78, 'vdw17base': True, 'vdw': 'BG'},
    {'rpa': True, 'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17base': True, 'vdw': 'BG'},
    {'rpa': True, 'vvpol': '', 'corr': True, 'beta': 0.8, 'vdw17base': True, 'vdw': 'BG'},
    {'rpa': True, 'vvpol': 'nm', 'beta': 0.8, 'vdw17base': True, 'vdw': 'BG'},
    {'rpa': True, 'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw': 'BG'},
    {'rpa': True, 'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17base': True},
    {'rpa': True, 'vvpol': 'nm', 'corr': True, 'beta': 0.82, 'vdw17base': True, 'vdw': 'BG'},
    {'rpa': True, 'vvpol': 'nm', 'corr': True, 'beta': 0.78, 'vdw17base': True, 'vdw': 'BG'},
]
_df, _ds = calculate_solids(_variants)


# In[ ]:


def add_group(x, ds):
    return x.assign(group=ds.df.loc(0)[x.name[:2]].group)

_res = (
    pd.concat([_df]).loc[lambda x: ~x.index.duplicated('last')]
    .groupby('label scale'.split()).apply(ene_int, _ds)
    .apply(ene_dft_vdw, 1).stack(dropna=False)
    .pipe(lambda x: x*ev).to_frame('ene')
    .groupby('label scale'.split()).apply(ref_delta, _ds)
    .groupby('label scale'.split()).apply(add_group, _ds)
    .groupby('group method scale'.split()).apply(ds_stat)
    .round(4)
)

_res.to_csv('../results/solids-energies.csv')
_res


# ### S66

# In[ ]:


def calculate_ds(dsname, variants, alpha_vv=None):
    with app.context():
        if alpha_vv is None:
            df_dft, ds, alpha_vv = app.get(dsname)
            alpha_vv = pd.read_hdf(alpha_vv['alpha.h5'].path)
        else:
            df_dft, ds = app.get(dsname)
    df = []
    with MBDCalc(4) as mbd_calc:
        for (*key, fragment), data, _ in tqdm(list(df_dft.itertuples())):
            try:
                data = {**data, 'alpha_vv': alpha_vv.loc(0)[(*key, fragment)]}
            except KeyError:
                pass
            if data is None:
                continue
            pbe_ene = data['energy'][0]['value'][0]
            df.append((*key, fragment, 'PBE', pbe_ene))
            enes = all_mbd_variants(mbd_calc, data, variants, kdensity=0.8)
            enes = {get_variant_label(v): ene for v, ene in zip(variants, enes)}
            for mbd_label, ene in enes.items():
                df.append((*key, fragment, mbd_label, ene))
    df = pd.DataFrame(df, columns='label scale fragment method ene'.split())         .set_index('label scale fragment method'.split())
    return df, ds

def analyse_s66(df, ds):
    df = (
        df
        .groupby('label scale'.split()).apply(ene_int, ds)
        .apply(ene_dft_vdw, 1).stack(dropna=False)
        .pipe(lambda x: x*kcal).to_frame('ene')
        .groupby('label scale'.split()).apply(ref_delta, ds)
    )
    return pd.concat((
        df.loc(0)[:, 1.].groupby('method scale'.split()).apply(ds_stat),
        df.loc(0)[:, 2.].groupby('method scale'.split()).apply(ds_stat),
        df.groupby('method').apply(ds_stat).assign(scale=np.inf).set_index('scale', append=True),
    )).sort_index()


# In[ ]:


_variants = [
    {'scs': True},
    {'scs': True, 'vdw': 'BG'},
    {'rpa': True, 'scs': True},
    {'scs': True, 'vdw17': True},
    {'scs': True, 'vv_scale': 1},
    {'vvpol': 'nm', 'beta': 0.8},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8},
    {'vvpol': '', 'corr': True, 'beta': 0.8},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17base': True},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.81, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.79, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17': True},
]
_df , _ds = calculate_ds('s66', _variants)


# In[ ]:


_res = analyse_s66(_df, _ds).round(4)
_res.to_csv('../results/s66-energies.csv')
_res


# ### X23

# In[ ]:


with app.context():
    _df, _ = app.get('x23')
with MBDCalc(4) as _mbd_calc:
    _freq = _mbd_calc.omega_grid[0]
def _f(x):
    return (
        pd
        .concat(
            dict(x.apply(lambda x: pd.read_hdf(x) if x else None)),
            names='label scale fragment i_point'.split()
        )
        .set_index('i_atom', append=True)
        .pipe(calc_vvpol, _freq, rgrad_cutoff)
        .groupby('scale fragment i_atom'.split()).sum()
    )
x23_alpha_vv = _df.gridfile.groupby('label').apply(_f)


# In[ ]:


_variants = [
    {'scs': True},
    {'vvpol': 'nm', 'beta': 0.8},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8},
    {'vvpol': '', 'corr': True, 'beta': 0.8},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17base': True},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.81, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.79, 'vdw17base': True, 'vdw': 'BG'},
    {'vvpol': 'nm', 'corr': True, 'beta': 0.8, 'vdw17': True},
]
_df , _ds = calculate_ds('x23', _variants, x23_alpha_vv)


# In[ ]:


(
    _df
    .groupby('label scale'.split()).apply(ene_int, _ds)
    .apply(ene_dft_vdw, 1).stack(dropna=False)
    .pipe(lambda x: x*kcal).to_frame('ene')
    .groupby('label scale'.split()).apply(ref_delta, _ds)
    .groupby('method scale'.split()).apply(ds_stat)
    # .loc(0)[:, :, ['PBE+MBD(scs)', 'PBE+MBD(vvpol[nm],corr,beta[0.8],17base)']]
)

