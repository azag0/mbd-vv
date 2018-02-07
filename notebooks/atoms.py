
# coding: utf-8

# In[1]:


from mbdvv import app, get_solids, get_s22_set, get_s66_set, kcal, ev
from mbdvv.mbd import MBDException, mbd_rsscs
from pymbd import MBDCalc, from_volumes, ang, vdw_params

from scipy.special import erf
import numpy as np
import pandas as pd
import os
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


def ene_int(x, ds, get_key):
    key = get_key(x)
    enes = dict(x[['fragment', 'ene']].itertuples(index=False))
    cluster = ds.clusters[key]
    return pd.Series({'ene': cluster.get_int_ene(enes)})

def ref_delta(x, ds, get_key):
    ene = x.iloc[0]
    ref = ds.clusters[get_key(x)].energies['ref']
    delta = ene-ref
    reldelta = delta/abs(ref)
    return pd.Series({
        'ene': ene,
        'delta': ene-ref,
        'reldelta': (ene-ref)/abs(ref),
    })

def ene_dft_vdw(x):
    return pd.Series({
        'PBE': x['PBE'],
        'PBE+MBD': x['PBE']+x['MBD'],
        'PBE+MBD(RPA)': x['PBE']+x['MBD(RPA)'],
    })
    
def ds_stat(x):
    return pd.Series({
        'N': len(x.dropna()),
        'STD': x['reldelta'].std(),
        'mean': x['reldelta'].mean(),
        'MARE': abs(x['reldelta']).mean(),
        'median': x['reldelta'].median(),
        'MAE': abs(x['delta']).mean(),
    })

def splice_key(df, indexes):
    return df.reset_index().assign(
        label=lambda x: x.key.map(lambda y: y[0]),
        scale=lambda x: x.key.map(lambda y: y[1]),
    ).drop('key', 1).set_index(['label', 'scale', *indexes])


# In[4]:


def mbd_from_data(calc, data, beta, **kwargs):
    coords = data['coords']['value'].T
    species = listify(data['elems']['atom'])
    lattice = data['lattice_vector']['value'] if 'lattice_vector' in data else None
    volumes = last(data['volumes'])
    alpha_vv = last(data['vv_pols'])
    free_atoms = last(data['free_atoms'])
    species_idx = free_atoms['species']-1
    volumes_free = free_atoms['volumes'][species_idx]
    alpha_vv_free = free_atoms['vv_pols'][:, species_idx]
    freq_w = last(data['omega_grid_w'])

    alpha_0, C6, R_vdw = from_volumes(species, volumes/volumes_free)
    alpha_0_free = np.array([vdw_params.get(sp)['alpha_0'] for sp in species])
    C6_vv = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv**2, 0)
    C6_vv_free = 3/np.pi*np.sum(freq_w[:, None]*alpha_vv_free**2, 0)
    R_vdw_1 = 2.5*alpha_0**(1/7)
    R_vdw_vv = 2.5*alpha_vv[0]**(1/7)
    return mbd_rsscs(
        calc,
        coords,
        alpha_0, C6, R_vdw,
        beta,
        lattice=lattice,
        **kwargs
    )


# In[5]:


def all_mbd_variants(calc, data):
    variants = {
        'MBD': {},
        'MBD(RPA)': {'rpa': True},
    }
    k_grid = np.repeat(4, 3) if 'lattice_vector' in data else None
    enes = {}
    for label, kwargs in variants.items():
        try:
            ene = mbd_from_data(calc, data, 0.83, k_grid=k_grid, **kwargs)
        except MBDException as e:
            ene = np.nan
        enes[label] = ene
    return enes


# In[6]:


def calculate_solids():
    df_dft, ds = get_solids(app.ctx)
    atom_enes = df_dft['atoms'].unstack().min(1).to_frame('ene').reset_index()[['atom', 'ene']].set_index('atom').ene
    df = []
    with MBDCalc() as mbd_calc:
        for (_, label, scale), data in tqdm(list(df_dft['solids'].loc(0)['solids', :, 1.].itertuples())):
            key = label, scale
            atoms = [
                ''.join(c) for c in
                chunks(re.split(r'([A-Z])', label)[1:], 2)
            ]
            pbe_ene = data['energy'][0]['value'][0]
            df.append((key, 'solid', 'PBE', pbe_ene))
            for atom in atoms:
                df.append((key, atom, 'PBE', atom_enes[atom]))
            enes = all_mbd_variants(mbd_calc, data)
            for mbd_label, ene in enes.items():
                df.append((key, 'solid', mbd_label, ene))
    df = pd.DataFrame(df, columns='key fragment method ene'.split())         .set_index('key fragment method'.split())
    return df, ds

dataframe, dataset = calculate_solids()


# In[7]:


(
    dataframe.loc(0)[:, :, 'PBE'].reset_index()
    .groupby('key method'.split())
    .apply(ene_int, dataset, lambda x: x['key'].iloc[0])
    .pipe(lambda df: pd.concat((
        df,
        df.xs('PBE', level='method').join(
            dataframe.query('method != "PBE"')
            .xs('solid', level='fragment')
            .reset_index('method'),
            lsuffix='_pbe', rsuffix='_vdw'
        ).apply(lambda x: pd.Series({
            'ene': x.ene_pbe+x.ene_vdw,
            'method': 'PBE+' + str(x.method)
        }), 1).set_index('method', append=True),
    )).sort_index())
    .assign(ene=lambda x: x.ene*ev)
    .apply(ref_delta, axis=1, args=(dataset, lambda x: x.name[0]))
    .pipe(splice_key, ['method'])
    .groupby('method').apply(ds_stat)
)


# In[8]:


def calculate_s66():
    df_dft, ds = get_s66_set(app.ctx)
    df = []
    with MBDCalc() as mbd_calc:
        for (key, fragment), data in tqdm(list(df_dft.loc(0)[ds.name].itertuples())):
            pbe_ene = data['energy'][0]['value'][0]
            df.append((key, fragment, 'PBE', pbe_ene))
            enes = all_mbd_variants(mbd_calc, data)
            for mbd_label, ene in enes.items():
                df.append((key, fragment, mbd_label, ene))
    df = pd.DataFrame(df, columns='key fragment method ene'.split())         .set_index('key fragment method'.split())
    return df, ds

dataframe, dataset = calculate_s66()


# In[9]:


(
    dataframe.reset_index()
    .groupby('key method'.split())
    .apply(ene_int, dataset, lambda x: x['key'].iloc[0])
    ['ene'].unstack().apply(ene_dft_vdw, 1).rename_axis('method', 1).stack().to_frame('ene')
    .assign(ene=lambda x: x.ene*kcal)
    .apply(ref_delta, 1, args=(dataset, lambda x: x.name[0]))
    .pipe(splice_key, ['method'])
    .groupby('method').apply(ds_stat)
)

