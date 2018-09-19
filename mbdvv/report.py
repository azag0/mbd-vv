from itertools import product, combinations_with_replacement
from collections import OrderedDict
from pkg_resources import resource_filename

import pymbd
from pymbd import from_volumes, ang, vdw_params, get_kgrid
import pyatsol

import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm

from mbdvv.app import app
from mbdvv.functasks import calc_vvpol
from mbdvv.utils import last, listify

import matplotlib as mpl
import matplotlib.cm  # noqa

mbd = pymbd.interactive(4)
_I = pd.IndexSlice

vdw_params = pd.DataFrame(vdw_params).T.replace('', np.nan)
vdw_params['C6(surf)'] = vdw_params['C6(TS)']
vdw_params['alpha_0(surf)'] = vdw_params['alpha_0(TS)']
vdw_params['R_vdw(surf)'] = vdw_params['R_vdw']
_vdw_params_surf = pd.read_csv(resource_filename('mbdvv', 'data/vdw-surf.csv')) \
    .set_index('label')
vdw_params.loc[_vdw_params_surf.index, 'C6(surf) alpha_0(surf) R_vdw(surf)'.split()] = \
    _vdw_params_surf.values

pyatsol.data['Cr']['shell_occ'] = [2, 2, 6, 2, 6, 4, 2]
pyatsol.data['Cu']['shell_occ'] = [2, 2, 6, 2, 6, 9, 2]
pyatsol.data['Nb']['shell_occ'] = [2, 2, 6, 2, 6, 10, 2, 6, 3, 2]
pyatsol.data['Mo']['shell_occ'] = [2, 2, 6, 2, 6, 10, 2, 6, 4, 2]
pyatsol.data['Ru']['shell_occ'] = [2, 2, 6, 2, 6, 10, 2, 6, 6, 2]
pyatsol.data['Rh']['shell_occ'] = [2, 2, 6, 2, 6, 10, 2, 6, 7, 2]
pyatsol.data['Pd']['shell_n'] = [1, 2, 2, 3, 3, 3, 4, 4, 4, 5]
pyatsol.data['Pd']['shell_l'] = [0, 0, 1, 0, 1, 2, 0, 1, 2, 0]
pyatsol.data['Pd']['shell_occ'] = [2, 2, 6, 2, 6, 10, 2, 6, 8, 2]
pyatsol.data['Ag']['shell_occ'] = [2, 2, 6, 2, 6, 10, 2, 6, 9, 2]


def _array(obj, *args, **kwargs):
    if obj is not None:
        return np.array(obj, *args, **kwargs)


class MBDException(Exception):
    pass


class NegativeEigs(MBDException):
    pass


class NegativeAlpha(MBDException):
    pass


def my_cm():
    cm = mpl.cm.get_cmap('magma')
    cm.set_bad((0, 0, 0))
    return cm


class MyLogNorm(mpl.colors.LogNorm):
    def autoscale_None(self, A):
        if self.vmin is not None and self.vmax is not None:
            return
        A = np.ma.masked_less_equal(A, 0, copy=False)
        if self.vmax is None and A.size:
            self.vmax = A.max()
        self.vmin = self.vmax/self.vmin


def scaled_eigs(x):
    return np.where(x >= 0, x, -erf(np.sqrt(np.pi)/2*x**4)**(1/4))


def screening(coords, alpha_0, R_vdw=None, beta=0.,
              lattice=None, damping='fermi,dip,gg', param_a=6.):
    coords = _array(coords, dtype=float, order='F')
    alpha_0 = _array(alpha_0, dtype=float)
    R_vdw = _array(R_vdw, dtype=float)
    sigma = (np.sqrt(2/np.pi)*alpha_0/3)**(1/3)
    alpha_nlc = np.linalg.inv(
        np.diag(np.repeat(1/alpha_0, 3)) + mbd.dipole_matrix(
            coords, damping, sigma=sigma, R_vdw=R_vdw,
            beta=beta, lattice=lattice,
        )
    )
    return 1/3*np.array(
        [alpha_nlc[i::3, i::3].sum(1) for i in range(3)]
    ).sum(0)


def mbd_energy(coords, alpha_0, C6, R_vdw, beta,
               lattice=None, k_grid=None, rpa=False, scs=False, no_eigscale=False,
               ts=False, ord2=False, damping='fermi,dip', param_a=6., scr_damping=None,
               get_vdw_params=False, no_vdwscs=False, vdwscs=False):
    coords = _array(coords, dtype=float, order='F')
    alpha_0 = _array(alpha_0, dtype=float)
    C6 = _array(C6, dtype=float)
    R_vdw = _array(R_vdw, dtype=float)
    freq, freq_w = mbd.omega_grid
    omega = 4/3*C6/alpha_0**2
    if scs or vdwscs:
        alpha_dyn = alpha_0/(1+(freq[:, None]/omega)**2)
        alpha_dyn_rsscs = np.empty_like(alpha_dyn)
        for a, a_scr in zip(alpha_dyn, alpha_dyn_rsscs):
            a_scr[:] = screening(
                coords, a, R_vdw, beta, lattice=lattice,
                damping=scr_damping or damping + ',gg', param_a=param_a
            )
        alpha_0_rsscs = alpha_dyn_rsscs[0, :]
        if np.any(alpha_0_rsscs <= 0):
            raise NegativeAlpha(alpha_0_rsscs)
        C6_rsscs = 3./np.pi*np.sum(freq_w[:, None]*alpha_dyn_rsscs**2, 0)
        R_vdw_rsscs = R_vdw*(alpha_0_rsscs/alpha_0)**(1/3)
        omega_rsscs = 4/3*C6_rsscs/alpha_0_rsscs**2
    if not scs:
        alpha_0_rsscs = alpha_0
        C6_rsscs = C6
        omega_rsscs = omega
    if scs and no_vdwscs or not scs and not vdwscs:
        R_vdw_rsscs = R_vdw
    if get_vdw_params:
        return pd.DataFrame({
            'alpha_0': alpha_0_rsscs,
            'C6': C6_rsscs,
            'R_vdw': R_vdw_rsscs,
        }, index=range(1, len(coords)+1)).rename_axis('i_atom')
    if ts:
        return mbd.ts_energy(
            coords, alpha_0_rsscs, C6_rsscs, R_vdw_rsscs, beta, lattice,
            d=param_a, damping=damping
        )
    pre = np.repeat(omega_rsscs*np.sqrt(alpha_0_rsscs), 3)
    if lattice is None:
        k_grid = [None]
    else:
        assert k_grid is not None
        k_grid = get_kgrid(lattice, k_grid)
    ene = 0
    for k_point in k_grid:
        T = mbd.dipole_matrix(
            coords, damping, R_vdw=R_vdw_rsscs, beta=beta, a=param_a,
            lattice=lattice, k_point=k_point
        )
        if not rpa:
            eigs = np.linalg.eigvalsh(
                np.diag(np.repeat(omega_rsscs**2, 3))+np.outer(pre, pre)*T
            )
            if np.any(eigs < 0):
                raise NegativeEigs(k_point, eigs)
            ene += np.sum(np.sqrt(eigs))/2-3*np.sum(omega_rsscs)/2
        else:
            for u, uw in zip(freq[1:], freq_w[1:]):
                A = np.diag(np.repeat(alpha_0_rsscs/(1+(u/omega_rsscs)**2), 3))
                eigs = np.real(np.linalg.eigvals(A@T))
                if ord2:
                    log_eigs = -eigs**2/2
                if not no_eigscale:
                    eigs = scaled_eigs(eigs)
                if np.any(eigs <= -1):
                    raise NegativeEigs(k_point, u, eigs)
                if ord2:
                    log_eigs = -eigs**2/2
                elif not no_eigscale:
                    log_eigs = np.log(1+eigs)-eigs
                else:
                    log_eigs = np.log(1+eigs)
                ene += 1/(2*np.pi)*np.sum(log_eigs)*uw
    ene /= len(k_grid)
    return ene


def get_nk(lattice, density):
    rec_lattice = 2*np.pi*np.linalg.inv(lattice.T)
    rec_lens = np.sqrt((rec_lattice**2).sum(1))
    nkpts = np.ceil(rec_lens/(density/ang**2))
    return int(nkpts[0]), int(nkpts[1]), int(nkpts[2])


def mbd_from_data(aims_data,
                  vv=None, vv_norm=False, df_alpha_vv=None,
                  free_atoms_vv=None, Rvdw17_base=False,
                  Rvdw_vol_scale=1/3, Rvdw17=False, Rvdw_scale_vv=False,
                  vdw_ref='TS', nm_diff=False, kdensity=None, k_grid=None,
                  **kwargs):
    coords = aims_data['coords']['value'].T
    species = listify(aims_data['elems']['atom'])
    lattice = aims_data['lattice_vector']['value'].T \
        if 'lattice_vector' in aims_data else None
    if lattice is not None:
        if k_grid is None:
            if kdensity is None:
                k_grid = np.repeat(4, 3)
            else:
                k_grid = get_nk(aims_data['lattice_vector']['value'], kdensity)
    volumes = last(aims_data['volumes'])
    free_atoms = last(aims_data['free_atoms'])
    species_idx = free_atoms['species']-1
    volumes_free = free_atoms['volumes'][species_idx]
    if df_alpha_vv is not None:
        alpha_vv = df_alpha_vv['alpha_vv']
        if isinstance(vv, str):
            alpha_vv_nm = df_alpha_vv['alpha_vv_' + vv]
        if free_atoms_vv is not None:
            alpha_0_vv_free = free_atoms_vv['alpha_0'][species].values
            C6_vv_free = free_atoms_vv['C6'][species].values
        elif vv_norm == 'aims':
            alpha_vv_free = free_atoms['vv_pols'][:, species_idx]
            alpha_0_vv_free = alpha_vv_free[0]
            freq_w = last(aims_data['omega_grid_w'])
            C6_vv_free = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv_free**2, 0)
        freq_w = mbd.omega_grid[1]
    else:
        alpha_vv = last(aims_data['vv_pols']).copy()
        try:
            alpha_vv_nm = last(aims_data['vv_pols_' + vv]).copy()
        except (TypeError, KeyError):
            alpha_vv_nm = None
        alpha_vv_free = free_atoms['vv_pols'][:, species_idx]
        alpha_0_vv_free = alpha_vv_free[0]
        freq_w = last(aims_data['omega_grid_w'])
        C6_vv_free = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv_free**2, 0)
    vdw_params_i = vdw_params.loc(0)[species]
    alpha_0_free = vdw_params_i[f'alpha_0({vdw_ref})'].values
    C6_free = vdw_params_i[f'C6({vdw_ref})'].values
    R_vdw_free = vdw_params_i['R_vdw' if vdw_ref != 'surf' else 'R_vdw(surf)'].values

    def vv_normalize(alpha_0, C6):
        alpha_0 = alpha_0*alpha_0_free/alpha_0_vv_free
        C6 = C6*C6_free/C6_vv_free
        return alpha_0, C6

    if vv is True or nm_diff:
        C6_vv = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv**2, 0)
        alpha_0, C6 = alpha_vv[0], C6_vv
    elif isinstance(vv, str):
        C6_vv_nm = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv_nm**2, 0)
        alpha_0, C6 = alpha_vv_nm[0], C6_vv_nm
    else:
        alpha_0 = alpha_0_free*(volumes/volumes_free)
        C6 = C6_free*(volumes/volumes_free)**2
    if vv is not None and vv_norm:
        alpha_0, C6 = vv_normalize(alpha_0, C6)
    if Rvdw17:
        R_vdw = 2.5*alpha_0**(1/7)
    else:
        if Rvdw17_base:
            R_vdw_free = 2.5*alpha_0_free**(1/7)
        assert Rvdw_scale_vv in {True, False, 'cutoff'}
        if Rvdw_scale_vv is True:
            vol_scale = alpha_vv[0]/alpha_0_vv_free
        elif Rvdw_scale_vv == 'cutoff':
            vol_scale = alpha_vv_nm[0]/alpha_0_vv_free
        else:
            vol_scale = volumes/volumes_free
        R_vdw = R_vdw_free*vol_scale**Rvdw_vol_scale
    ene = mbd_energy(
        coords, alpha_0, C6, R_vdw, lattice=lattice, k_grid=k_grid, **kwargs
    )
    if kwargs.get('get_vdw_params'):
        ene = ene.assign(species=species)
    if not nm_diff:
        return ene
    alpha_vv_mtl = alpha_vv-alpha_vv_nm
    C6_vv_mtl = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv_mtl**2, 0)
    alpha_0, C6 = alpha_vv_mtl[0], C6_vv_mtl
    if vv_norm:
        alpha_0, C6 = vv_normalize(alpha_0, C6)
    ene_mtl = mbd_energy(
        coords, alpha_0, C6, R_vdw, lattice=lattice, k_grid=k_grid, **kwargs
    )
    return ene-ene_mtl


known_spec_flags = {
    'rpa', 'scs', 'Rvdw17', 'vv_norm', 'ts', 'ord2',
    'beta', 'damping', 'param_a', 'Rvdw_vol_scale', 'C_vv',
    'vdw_ref', 'Rvdw17_base', 'vv', 'no_eigscale', 'scr_damping',
    'Rvdw_scale_vv', 'vdwscs', 'no_vdwscs', 'nm_diff', 'kdensity',
    'k_grid',
}


def get_spec_label(flags):
    inner = []
    for k, v in flags.items():
        if k not in known_spec_flags:
            raise ValueError('Uknown flag:', k)
        if k == 'Rvdw17_base':
            k = '17base'
        elif k == 'aims_vv_free':
            k = 'aimsvvfree'
        if isinstance(v, bool):
            assert v
            inner.append(k)
        else:
            inner.append(f'{k}[{v}]')
    return f'MBD({",".join(inner)})'


def evaluate_mbd_specs(specs, aims_data, alpha_vv=None, free_atoms_vv=None,
                       get_vdw_params=False):
    alpha_vv_cache = {}
    df = []
    payload = product(aims_data.itertuples(), specs)
    total = len(aims_data)*len(specs)
    for ((*key, fragment), aims_data_i, _), spec in tqdm(payload, total=total):
        if aims_data_i is None:
            continue
        if not spec:
            ene_pbe = aims_data_i['energy'][0]['value'][0]
            df.append((*key, fragment, 'PBE', ene_pbe))
            continue
        spec = spec.copy()
        spec_label = spec.pop('_label', None)
        if not spec_label:
            spec_label = get_spec_label(spec)
        if len(last(aims_data_i['volumes'])) == 1 and \
                'lattice_vector' not in aims_data_i:
            df.append((*key, fragment, spec_label, 0.))
            continue
        C_vv = spec.pop('C_vv', None)
        if C_vv:
            C_vv_key = C_vv, *key, fragment
            alpha_vv_i = alpha_vv_cache.get(C_vv_key)
            if alpha_vv_i is None:
                alpha_vv_i = alpha_vv.loc(0)[C_vv_key]
                alpha_vv_i = {
                    'alpha_vv': alpha_vv_i['vvpol'].values.T,
                    'alpha_vv_nm': alpha_vv_i['vvpol_nm'].values.T,
                    'alpha_vv_lg': alpha_vv_i['vvpol_lg'].values.T,
                    'alpha_vv_lg2': alpha_vv_i['vvpol_lg2'].values.T,
                }
                alpha_vv_cache[C_vv_key] = alpha_vv_i
            vv_norm = spec.get('vv_norm')
            if vv_norm in {'sph', 'nonsph'}:
                free_atoms_vv_i = free_atoms_vv.loc(0)[C_vv, vv_norm == 'sph']
            elif vv_norm in {'aims', None}:
                free_atoms_vv_i = None
            else:
                raise ValueError(f'Unkonwn vv_norm: {vv_norm}')
        else:
            alpha_vv_i, free_atoms_vv_i = None, None
        try:
            result = mbd_from_data(
                aims_data_i,
                df_alpha_vv=alpha_vv_i, free_atoms_vv=free_atoms_vv_i,
                get_vdw_params=get_vdw_params, **spec
            )
        except MBDException as e:
            result = np.nan
        df.append((*key, fragment, spec_label, result))
    df = pd.DataFrame(df, columns='label scale fragment method ene'.split())
    methods = df['method'].unique().tolist()
    df['method'] = df['method'].astype(pd.api.types.CategoricalDtype(categories=methods))
    df.set_index('label scale fragment method'.split(), inplace=True)
    return df


def get_key_from_df(df):
    idx = df.iloc[:1].index
    return idx.get_level_values('label').values[0], \
        idx.get_level_values('scale').values[0]


def ene_dft_vdw(df):
    df = df.ene.unstack('method')
    is_pbe = df.columns == 'PBE'
    df.loc(1)[~is_pbe] = df.loc(1)[~is_pbe].apply(lambda x: x + df['PBE'], axis=0)
    renamed_methods = ['PBE+' + x if x != 'PBE' else x for x in df.columns]
    df.columns = df.columns.set_categories(renamed_methods, rename=True)
    df = df.stack().to_frame('ene')
    return df


def process_cluster(df, ds):
    key = get_key_from_df(df)
    cluster = ds.clusters[key]
    cluster_df = ds.df.loc(0)[key]
    ref = cluster_df.energy
    if ds.name == 'S66' and ref > 0:
        ref = np.nan
    try:
        ene = cluster.get_int_ene(df.ene.unstack('fragment'))
    except KeyError:
        ene = df.ene.unstack('fragment')['complex']*np.nan
    delta = ene-ref
    df = pd.DataFrame(OrderedDict({
        'ene': ene,
        'delta': delta,
        'reldelta': delta/abs(ref),
        'group': cluster_df.get('group')
    }))
    return df


def dataset_stats(df):
    return pd.Series(OrderedDict({
        'N': len(df.dropna()),
        'MRE': df['reldelta'].mean(),
        'MARE': abs(df['reldelta']).mean(),
        'MdRE': df['reldelta'].median(),
        'MdARE': abs(df['reldelta']).median(),
        'SDRE': abs(df['reldelta']).std(),
        'ME': df['delta'].mean(),
        'MAE': abs(df['delta']).mean(),
    }))


def dataset_scale_stats(df):
    df = pd.concat([
        dataset_stats(df.xs(1, level='scale')),
        dataset_stats(df.xs(2, level='scale')),
        dataset_stats(df),
    ], names=['subset'], keys=['equilibrium', 'x2 equilibrium', 'all'])
    return df


def specs_to_binding_enes(specs, ds, aims_data, alpha_vvs=None,
                          free_atoms_vv=None, *, unit):
    energies = evaluate_mbd_specs(
        specs, aims_data, alpha_vvs, free_atoms_vv
    )
    return (
        energies
        .pipe(ene_dft_vdw).pipe(lambda df: df*unit)
        .groupby('label scale'.split(), group_keys=False).apply(process_cluster, ds)
    )


def calc_vvpol_custom(x, freq, pol_func):
    idx = x.index
    n = x.rho.values
    grad = x.rho_grad_norm.values
    kin = x.kin_dens.values
    w = x.part_weight.values
    alpha = pol_func(n, grad, kin, u=freq[:, None])
    alpha = pd.DataFrame(alpha*w).T
    alpha.index = idx
    return alpha


def C6_dataset_from_alpha(alpha, freq_w):
    mol_C6 = []
    alpha_0 = alpha.iloc(1)[0]
    C6 = pd.Series(3/np.pi*np.sum(freq_w[None, :]*alpha.values**2, 1), index=alpha_0.index)
    vdw_params = pd.DataFrame({'alpha_0': alpha_0, 'C6': C6})
    molecules = dict(iter(vdw_params.groupby('label')))
    for (lbl1, mol1), (lbl2, mol2) in \
            combinations_with_replacement(molecules.items(), 2):
        C6AA = mol1.C6.values[:, None]
        C6BB = mol2.C6.values[None, :]
        alpha_0A = mol1.alpha_0.values[:, None]
        alpha_0B = mol2.alpha_0.values[None, :]
        C6_inter = 2*C6AA*C6BB/(alpha_0B/alpha_0A*C6AA+alpha_0A/alpha_0B*C6BB)
        mol_C6.append((lbl1, lbl2, C6_inter.sum()))
    return pd.DataFrame(mol_C6, columns='system1 system2 C6'.split()) \
        .set_index('system1 system2'.split())


def calc_tspol(species, hirsh, freq):
    free_params = vdw_params.loc[species]
    alpha_0 = free_params['alpha_0(TS)'].values
    C6 = free_params['C6(TS)'].values
    omega = 4/3*C6/alpha_0**2
    alpha_dyn = (alpha_0*hirsh.values)/(1+freq[:, None]**2/omega**2)
    alpha_dyn = pd.DataFrame(alpha_dyn).T
    alpha_dyn.index = species.index
    return alpha_dyn


def rescale_alpha(alpha, ratios_alpha, ratios_C6):
    alpha = alpha.copy()
    alpha.iloc[:, 0] *= 1/ratios_alpha
    for i in range(1, alpha.shape[1]):
        alpha.iloc[:, i] *= 1/np.sqrt(ratios_C6)
    return alpha


def evaluate_func_on_C6(func, pts, ref, pts_free=None, species=None,
                        rescale=False):
    freq, freq_w = mbd.omega_grid
    alpha = calc_vvpol_custom(pts, freq, func) \
        .groupby('label i_atom'.split()).sum()
    if rescale:
        free_params = free_atoms_vv(pts_free, func)
        ratios = vv_ref_ratios(free_params).loc[species]
        alpha = rescale_alpha(
            alpha, ratios['alpha_0'].values, ratios['C6'].values
        )
    return evaluate_alpha_on_C6(alpha, freq_w, ref)


def evaluate_alpha_on_C6(alpha, freq_w, ref):
    C6 = C6_dataset_from_alpha(alpha, freq_w)
    C6 = C6.query(
        'system1 not in ["Li", "H", "N"] and system2 not in ["Li", "H", "N"]'
    )
    delta = C6-ref
    C6 = C6.assign(
        ref=ref,
        delta=delta,
        reldelta=delta/abs(ref)
    )
    return C6


def evaluate_ts_scs_on_C6(coords, hirsh, ref):
    freq, freq_w = mbd.omega_grid
    alphas = {}
    alphas['TS'] = calc_tspol(coords.species, hirsh, freq)
    alphas['SCS'] = pd.concat([
        alpha.apply(
            lambda alpha: screening(
                coords.loc[label, list('xyz')].values*ang,
                alpha,
                damping='dip,gg',
            ),
        ) for label, alpha in alphas['TS'].groupby('label')
    ])
    alphas = pd.concat({
        method: evaluate_alpha_on_C6(alpha, freq_w, ref)
        for method, alpha in alphas.items()
    }, names=['method'], sort=False)
    return alphas


def plot_matrix_df(ax, df, **kwargs):
    plot = ax.contour(df.columns.values, df.index.values, df.values, **kwargs)
    ax.clabel(plot, inline=1)
    return plot


def plot_rgrad_alpha(ax, df):
    return ax.hist2d(
        df.rgrad, df.alpha,
        range=[(0, .4), (0, 12)],
        bins=100,
        weights=df.vvpol*df.part_weight,
        norm=MyLogNorm(1e3),
        cmap=my_cm()
    )


def setup_C6_set():
    coords = (
        pd.read_hdf(
            resource_filename('mbdvv', 'data/C6-data.h5'),
            'coords'
        )
        .reset_index().rename({'system': 'label', 'level_1': 'i_atom'}, axis=1)
        .assign(i_atom=lambda x: x.i_atom+1).set_index('label i_atom'.split())
    )
    with app.context():
        aims_data = app.get('C6s')
    hirsh = {
        label: data['volumes'] /
        data['free_atoms']['volumes'][data['free_atoms']['species']-1]
        for (label, _, _), data, _ in aims_data.itertuples()
    }
    hirsh = pd.DataFrame(
        [
            (label, i, val)
            for label, arr in hirsh.items()
            for i, val in enumerate(arr)
        ],
        columns='label i_atom hirsh'.split()
    ).set_index('label i_atom'.split()).hirsh
    ref = (
        pd.read_hdf(resource_filename('mbdvv', 'data/C6-data.h5'), 'C6_ref')
        .set_index('system1 system2'.split())
    )
    pts = pd.concat(
        dict(
            aims_data['gridfile']
            .loc[:, 1., 'main']
            .apply(lambda x: pd.read_hdf(x))
        ),
        names=('label', 'i_point')
    )['i_atom part_weight rho rho_grad_norm kin_dens'.split()] \
        .set_index('i_atom', append=True)
    return coords, hirsh, ref, pts


def setup_s66():
    with app.context():
        aims_data, ds, alpha_vvs = app.get('s66')
    alpha_vvs = pd.concat({
        C: pd.read_hdf(alpha_vv['alpha.h5'].path)
        for C, alpha_vv in alpha_vvs.items()
    }, names=['C_vv'])
    alpha_vvs.index = alpha_vvs.index.set_names('label', 1)
    return aims_data, ds, alpha_vvs


def setup_s12l():
    with app.context():
        aims_data, ds, alpha_vvs = app.get('s12l')
    alpha_vvs = pd.concat({
        C: pd.read_hdf(alpha_vv['alpha.h5'].path)
        for C, alpha_vv in alpha_vvs.items()
    }, names=['C_vv'])
    alpha_vvs.index = alpha_vvs.index.set_names('label', 1)
    return aims_data, ds, alpha_vvs


def setup_x23():
    with app.context():
        aims_data, ds, alpha_vvs = app.get('x23')
    alpha_vvs = pd.concat({
        C: pd.read_hdf(alpha_vv['alpha.h5'].path)
        for C, alpha_vv in alpha_vvs.items()
    }, names=['C_vv'])
    alpha_vvs.index = alpha_vvs.index.set_names('label', 1)
    return aims_data, ds, alpha_vvs


def setup_solids():
    with app.context():
        aims_data, ds = app.get('solids')
    aims_data_atoms = aims_data['atoms'].assign(
        ene=lambda x: x.data.apply(
            lambda y: y['energy'][0]['value'][0] if y else None, 1
        )
    ).groupby('symbol').apply(lambda x: x.iloc[x.reset_index()['ene'].idxmin()]['data'])
    aims_data['solids'].sort_index(inplace=True)
    for species, data in aims_data_atoms.items():
        n = len(aims_data['solids'].loc(0)[:, :, species])
        aims_data['solids'].loc[_I[:, :, species], 'data'] = n*[data]
    pts = pd.concat(
        dict(
            aims_data['solids'].gridfile.loc[:, [1.], ['crystal']]
            .apply(lambda x: pd.read_hdf(x))
        ),
        names='label scale fragment i_point'.split()
    )['i_atom part_weight rho rho_grad_norm kin_dens'.split()] \
        .set_index('i_atom', append=True) \
        .assign(kin_dens=lambda x: x.kin_dens/2)
    alpha_vvs = pd.concat({
        C_vv: pts
        .pipe(calc_vvpol, mbd.omega_grid[0], C=C_vv)
        .groupby('label scale fragment i_atom'.split()).sum()
        for C_vv in [0.0093, 0.0101]
    }, names=['C_vv'])
    return aims_data['solids'], pts, ds, alpha_vvs


def setup_surface():
    with app.context():
        return app.get('surface')


def free_atoms_pts():
    nrad = 700
    conf_pot = np.zeros((nrad,))
    densities = {}
    r, rw = pyatsol.lib.init_grid(nrad, 1e-3, 1.017)
    payload = [
        ((species, data), spherical)
        for (species, data), spherical in
        product(pyatsol.data.items(), [True, False])
        if data['in'] and data['N'] == data['Z']
    ]
    for (species, data), spherical in tqdm(payload):
        dens = pyatsol.lib.solve_atom(
            data['Z'],
            data['shell_n'], data['shell_l'], data['shell_occ'],
            conf_pot, spherical=spherical, xc_func='PBE'
        )
        densities[spherical, species] = pd.DataFrame(dict(zip(
            'r rw n grad lap kin'.split(),
            [r, rw, *(x.sum(1) if x.shape == (nrad, 2) else x for x in dens[:4])]
        )))
    densities = pd.concat(densities, names='spherical species'.split())
    return densities


def free_atoms_vv(pts, func):
    freq, freq_w = mbd.omega_grid
    df = {}
    for specie, data in pts.groupby('species'):
        alpha_sp = func(
            data['n'].values, data['grad'].values, data['kin'].values,
            u=freq[:, None]
        )
        alpha_sp[:, 0] = 0
        alpha_dyn = np.sum(
            4*np.pi*data['r'].values**2*data['rw'].values*alpha_sp, 1
        )
        df[specie] = {
            'alpha_0': alpha_dyn[0],
            'C6': 3/np.pi*np.sum(alpha_dyn**2*freq_w)
        }
    df = pd.DataFrame(df).T
    df.index.rename('species', inplace=True)
    return df


def vv_ref_ratios(vv_params):
    vv_params = vv_params.copy()
    free_params = vdw_params.loc[vv_params.index.get_level_values('species')]
    vv_params['alpha_0'] /= free_params['alpha_0(TS)'].values
    vv_params['C6'] /= free_params['C6(TS)'].values
    return vv_params


def findmin(obj, **kwargs):
    kwargs.setdefault('kind', 'cubic')
    if isinstance(obj, pd.Series):
        f = interp1d(obj.index, obj.values, **kwargs)
        x0 = (obj.index[0]+obj.index[-1])/2
        bounds = ((obj.index[0]+1e-3, obj.index[-1]-1e-3),)
    elif isinstance(obj, pd.DataFrame):
        f0 = interp2d(obj.index, obj.columns, obj.values, **kwargs)
        f = lambda x: f0(x[0], x[1])  # noqa
        x0 = (
            (obj.index[0]+obj.index[-1])/2,
            (obj.columns[0]+obj.columns[-1])/2,
        )
        bounds = (
            (obj.index[0]+1e-3, obj.index[-1]-1e-3),
            (obj.columns[0]+1e-3, obj.columns[-1]-1e-3),
        )
    res = minimize(f, x0, bounds=bounds)
    if not res.success:
        raise RuntimeError(res.status, res.message)
    return res.x, res.fun
