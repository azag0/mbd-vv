from collections import defaultdict

import pandas as pd

from .app import ev
from .report import mbd_energy, vdw_params


def get_layered_dataset(results):
    idxs = []
    data_vars = defaultdict(list)
    freq_df = None
    grids_df = []
    for label, shift, xc, kz, data, gridfile in results:
        if gridfile:
            grids_df.append((label, shift, gridfile))
            continue
        if not data:
            continue
        idxs.append((label, shift, xc, kz))
        data_vars['coords'].append(data['coords']['value'].T)
        data_vars['lattice_vector'].append(data['lattice_vector']['value'].T)
        data_vars['elems'].append(data['elems']['atom'])
        data_vars['energy'].append(data['energy'][0]['value'][0])
        data_vars['energy_vdw'].append(data['energy'][1]['value'][0] or None)
        data_vars['volumes'].append(data['volumes'] if 'volumes' in data else None)
        data_vars['vv_pols'].append(data['vv_pols'].T if 'vv_pols' in data else None)
        data_vars['vv_pols_nm'].append(data['vv_pols_nm'].T if 'vv_pols_nm' in data else None)
        data_vars['C6_vv_nm'].append(data['C6_vv_nm'] if 'C6_vv_nm' in data else None)
        data_vars['alpha_0_vv_nm'].append(data['alpha_0_vv_nm'] if 'alpha_0_vv_nm' in data else None)
        data_vars['volumes_free'].append(data['free_atoms']['volumes'] if 'free_atoms' in data else None)
        data_vars['C6_vv_free'].append(data['free_atoms']['C6_vv'] if 'free_atoms' in data else None)
        data_vars['vv_pols_free'].append(data['free_atoms']['vv_pols'].T if 'free_atoms' in data else None)
        data_vars['alpha_0_vv_free'].append(data['free_atoms']['alpha_0_vv'] if 'free_atoms' in data else None)
        data_vars['species'].append(data['free_atoms']['species'] if 'free_atoms' in data else None)
        if freq_df is None and 'omega_grid' in data:
            freq_df = pd.DataFrame({
                'val': data['omega_grid'],
                'w': data['omega_grid_w'],
            })
    idx = pd.MultiIndex.from_tuples(idxs, names='label shift xc kz'.split())
    results_df = pd.DataFrame(data_vars, index=idx)
    grids_df = (
        pd.DataFrame(grids_df, columns='label shift gridfile'.split())
        .set_index('label shift'.split())
    )
    return results_df, grids_df, freq_df


def mbd_from_row(row, **kwargs):
    alpha_0_ratio = row['alpha_0_vv_nm']/row['alpha_0_vv_free'][row['species']-1]
    C6_ratio = row['C6_vv_nm']/row['C6_vv_free'][row['species']-1]
    return mbd_energy(
        row['coords'],
        vdw_params['alpha_0(TS)'][row['elems']].values*alpha_0_ratio,
        vdw_params['C6(TS)'][row['elems']].values*C6_ratio,
        2.5*vdw_params['alpha_0(TS)'][row['elems']].values**(1/7)*alpha_0_ratio**(1/3),
        lattice=row['lattice_vector'],
        **kwargs
    )


def binding_per_area(df):
    return -df['energy_uc']*ev*1e3/(df['area']*df['n_layer'])
