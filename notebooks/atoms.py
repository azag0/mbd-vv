
# coding: utf-8

# In[1]:


from mbdvv import app, get_solids, get_s22_set, get_s66_set, kcal, ev
from pymbd import MBDCalc, from_volumes, ang, vdw_params, get_kgrid

from scipy.special import erf
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from itertools import product, islice
from functools import partial
from pkg_resources import resource_stream
from tqdm import tqdm
import re

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[2]:


def last(obj):
    if not isinstance(obj, list):
        return obj
    assert len(obj) == 2
    return obj[-1]

def listify(obj):
    if isinstance(obj, list):
        return obj
    return [obj]

def chunks(iterable, n):
    iterable = iter(iterable)
    while True:
        chunk = list(islice(iterable, n))
        if not chunk:
            break
        yield chunk


# In[3]:


def ene_int(x, ds):
    key = x.iloc[0].name[:2]
    enes = x.reset_index('scale label'.split(), drop=True).ene.unstack('fragment')
    cluster = ds.clusters[key]
    enes_int = cluster.get_int_ene(enes)
    return enes_int

def ref_delta(x, ds):
    ref = ds.df.loc(0)[x.name[:2]].energy
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
        'SDRE': x['reldelta'].std(),
        'ME': x['delta'].mean(),
        'MAE': abs(x['delta']).mean(),
    }))

def splice_key(df, indexes):
    return df.reset_index().assign(
        label=lambda x: x.key.map(lambda y: y[0]),
        scale=lambda x: x.key.map(lambda y: y[1]),
    ).drop('key', 1).set_index(['label', 'scale', *indexes])


# In[4]:


class MBDException(Exception):
    pass


class NegativeEigs(MBDException):
    pass


class NegativeAlpha(MBDException):
    pass


def scaled_eigs(x):
    return np.where(x >= 0, x, -erf(np.sqrt(np.pi)/2*x**4)**(1/4))


def mbd_rsscs(mbd_calc, coords, alpha_0, C6, R_vdw, beta, lattice=None,
              k_grid=None, rpa=False, noscs=False, scale_eigs=True):
    def _array(obj, *args, **kwargs):
        if obj is not None:
            return np.array(obj, *args, **kwargs)

    coords = _array(coords, dtype=float, order='F')
    alpha_0 = _array(alpha_0, dtype=float)
    C6 = _array(C6, dtype=float)
    R_vdw = _array(R_vdw, dtype=float)
    freq, freq_w = mbd_calc.omega_grid
    omega = 4./3*C6/alpha_0**2
    if noscs:
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


# In[5]:


def mbd_from_data(calc, data, beta, vv_scale=None, vv_pol=False,
                  vv_corr=True, vdw17=False, **kwargs):
    coords = data['coords']['value'].T
    species = listify(data['elems']['atom'])
    lattice = data['lattice_vector']['value'] if 'lattice_vector' in data else None
    volumes = last(data['volumes'])
    alpha_vv = last(data['vv_pols']).copy()
    free_atoms = last(data['free_atoms'])
    species_idx = free_atoms['species']-1
    volumes_free = free_atoms['volumes'][species_idx]
    alpha_vv_free = free_atoms['vv_pols'][:, species_idx]
    freq_w = last(data['omega_grid_w'])
    C6_vv = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv**2, 0)
    C6_vv_free = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv_free**2, 0)
    alpha_0_free = np.array([vdw_params.get(sp)['alpha_0'] for sp in species])
    C6_free = np.array([vdw_params.get(sp)['C6'] for sp in species])

    if not vv_scale:
        volume_scale = volumes/volumes_free
    else:
        volume_scale = (alpha_vv[0]/alpha_vv_free[0])**(1/vv_scale)
    alpha_0, C6, R_vdw = from_volumes(species, volume_scale)
    if vv_pol:
        alpha_0 = alpha_vv[0]
        C6 = C6_vv
        if vv_corr:
            alpha_0 *= alpha_0_free/alpha_vv_free[0]
            C6 *= C6_free/C6_vv_free
    if vdw17:
        R_vdw = 2.5*alpha_0**(1/7)
    return mbd_rsscs(
        calc,
        coords,
        alpha_0, C6, R_vdw,
        beta,
        lattice=lattice,
        **kwargs
    )


# In[6]:


def all_mbd_variants(calc, data, variants):
    k_grid = np.repeat(4, 3) if 'lattice_vector' in data else None
    enes = {}
    for label, kwargs in variants.items():
        kwargs = kwargs.copy()
        beta = kwargs.pop('beta', 0.83)
        throw = kwargs.pop('throw', False)
        try:
            ene = mbd_from_data(calc, data, beta, k_grid=k_grid, **kwargs)
        except MBDException as e:
            if throw:
                raise e
            ene = np.nan
        enes[label] = ene
    return enes


# In[104]:


def calculate_solids(variants):
    dfs_dft, ds = get_solids(app.ctx)
    atom_enes = dfs_dft['atoms'].set_index('conf', append=True).ene.unstack().min(1)
    df = []
    with MBDCalc(4) as mbd_calc:
        for (*key, fragment), data in tqdm(list(dfs_dft['solids'].loc(0)[:, 1.].itertuples())):
            if fragment == 'crystal':
                pbe_ene = data['energy'][0]['value'][0]
            else:
                pbe_ene = atom_enes[fragment]
            df.append((*key, fragment, 'PBE', pbe_ene))
            if fragment == 'crystal':
                try:
                    enes = all_mbd_variants(mbd_calc, data, variants)
                except MBDException as e:
                    # this happens only if `variants` contains 'throw': True
                    print(label, scale, repr(e))
                    continue
            else:
                enes = {v: 0. for v in variants}
            for mbd_label, ene in enes.items():
                df.append((*key, fragment, mbd_label, ene))
    df = pd.DataFrame(df, columns='label scale fragment method ene'.split())         .set_index('label scale fragment method'.split())
    return df, ds

def analyse_solids(df, ds):
    return (
        df
        .groupby('label scale'.split()).apply(ene_int, ds)
        .apply(ene_dft_vdw, 1).stack(dropna=False)
        .pipe(lambda x: x*ev).to_frame('ene')
        .groupby('label scale'.split()).apply(ref_delta, dataset)
        .groupby('method scale'.split()).apply(ds_stat)
    )


# In[105]:


variants = {
    'MBD': {},
    'MBD(RPA)': {'rpa': True},
    'MBD(vdw17)': {'vdw17': True},
    'MBD(VV-scale[1])': {'vv_scale': 1},
    'MBD(vvpol)': {'vv_pol': True},
    'MBD(RPA,vvpol)': {'rpa': True, 'vv_pol': True},
    'MBD(vvpol,noscs)': {'vv_pol': True, 'noscs': True},
    'MBD(RPA,vvpol,noscs)': {'rpa': True, 'vv_pol': True, 'noscs': True},
    'MBD(vvpol,nocorr)': {'vv_pol': True, 'vv_corr': False},
    'MBD(RPA,vvpol,nocorr)': {'rpa': True, 'vv_pol': True, 'vv_corr': False},
    'MBD(vvpol,nocorr,noscs)': {'vv_pol': True, 'vv_corr': False, 'noscs': True},
    'MBD(RPA,vvpol,nocorr,noscs)': {'rpa': True, 'vv_pol': True, 'vv_corr': False, 'noscs': True},
    'MBD(vvpol,vdw17)': {'vv_pol': True, 'vdw17': True},
    'MBD(vvpol,nocorr,vdw17)': {'vv_pol': True, 'vv_corr': False, 'vdw17': True},
    'MBD(RPA,vvpol,nocorr,vdw17)': {'rpa': True, 'vv_pol': True, 'vv_corr': False, 'vdw17': True},
    'MBD(RPA,vvpol,nocorr,vdw17,noscs)': {'rpa': True, 'vv_pol': True, 'vv_corr': False, 'vdw17': True, 'noscs': True},
}
dataframe, dataset = calculate_solids(variants)
analyse_solids(dataframe, dataset)


# In[9]:


def calculate_s66(variants):
    df_dft, ds = get_s66_set(app.ctx)
    df = []
    with MBDCalc(4) as mbd_calc:
        for (*key, fragment), data in tqdm(list(df_dft.itertuples())):
            pbe_ene = data['energy'][0]['value'][0]
            df.append((*key, fragment, 'PBE', pbe_ene))
            enes = all_mbd_variants(mbd_calc, data, variants)
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
        .groupby('label scale'.split()).apply(ref_delta, dataset)
    )
    return pd.concat((
        df.loc(0)[:, 1.].groupby('method scale'.split()).apply(ds_stat),
        df.loc(0)[:, 2.].groupby('method scale'.split()).apply(ds_stat),
        df.groupby('method').apply(ds_stat).assign(scale=np.inf).set_index('scale', append=True),
    )).sort_index()


# In[10]:


variants = {
    'MBD': {},
    'MBD(RPA)': {'rpa': True},
    'MBD(VV-scale[1])': {'vv_scale': 1},
    'MBD(VV-scale[1.33])': {'vv_scale': 1.33},
    'MBD(vdw17)': {'vdw17': True},
    'MBD(vdw17,beta=0.8)': {'vdw17': True, 'beta': 0.8},
    'MBD(vdw17,beta=0.86)': {'vdw17': True, 'beta': 0.86},
    'MBD(vvpol)': {'vv_pol': True},
    'MBD(vvpol,noscs)': {'vv_pol': True, 'noscs': True},
    'MBD(vvpol,nocorr)': {'vv_pol': True, 'vv_corr': False},
    'MBD(vvpol,nocorr,noscs)': {'vv_pol': True, 'vv_corr': False, 'noscs': True},
    'MBD(vvpol,noscs,vdw17)': {'vv_pol': True, 'vdw17': True, 'noscs': True},
    'MBD(vvpol,noscs,nocorr,vdw17)': {'vv_pol': True, 'vv_corr': False, 'vdw17': True, 'noscs': True},
}
dataframe, dataset = calculate_s66(variants)
analyse_s66(dataframe, dataset)

