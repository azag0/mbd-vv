#!/usr/bin/env python
from pathlib import Path
from itertools import product
from functools import partial

import numpy as np
import pandas as pd

from mbdvv.physics import reduced_grad, vv_pol, alpha_kin
import mbdvv.report as rp

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style='ticks', context='paper')
mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('mathtext', fontset='stix')


def savefig(fig, name, ext='pdf', **kwargs):
    fig.savefig(
        f'media/{name}.{ext}',
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
        **kwargs)


def my_pol(n, grad, kin, C1=8*np.sqrt(0.0093), C2=0., u=0.):
    tw = grad**2/(8*n)
    return n/(4*np.pi/3*n+((C1*tw+C2*kin)/n)**2+u**2)


FREE_ATOMS_H5 = Path('data/free-atoms.h5')
if FREE_ATOMS_H5.exists():
    free_atoms_pts = pd.read_hdf(FREE_ATOMS_H5, 'table')
else:
    free_atoms_pts = rp.free_atoms_pts()
    free_atoms_pts.to_hdf(FREE_ATOMS_H5, 'table')


free_atoms_vv = (
    pd.concat({
        (C_vv, sph): rp.free_atoms_vv(
            free_atoms_pts.loc[sph], partial(my_pol, C1=8*np.sqrt(C_vv))
        )
        for C_vv, sph in product([0.0093, 0.0101], [True, False])
    }, names='C_vv spherical'.split())
    .merge(
        rp.vdw_params.rename_axis('species').query('N == Z'),
        left_on='species',
        right_index=True,
        how='outer'
    )
    .sort_index()
)


def plot_species(ax, df, num, den, label=None):
    df = df.sort_values('N').query('N <= 54 & N != 46')
    what = df[num]/df[den]
    ax.plot(df['N'], what.values, label=label)


fig, ax = plt.subplots(figsize=(3.5, 1.1))
ax.axhline(1, color='black')
plot_species(ax, free_atoms_vv.loc(0)[0.0093, False], 'C6', 'C6(TS)', 'VV')
ax.set_xlim(1, 54)
ax.set_xticks([2, 10, 18, 30, 36, 48, 54])
ax.set_xticklabels('He Ne Ar Zn Kr Cd Xe'.split())
ax.set_ylim(0, None)
ax.set_yticks([0, 1])
ax.set_ylabel(r'$C_6^\mathrm{VV}/C_6^\mathrm{ref}$')

savefig(fig, 'vv-periodic-table')

solid_groups = {
    'SC': 'semiconductors',
    'MM': 'main-group metals',
    'TM': 'transition metals',
    'TMCN': 'TM carbides & nitrides',
    'IC': 'ionic crystals',
}

SOLIDS_VV_H5 = Path('data/solids-vv.h5')
if SOLIDS_VV_H5.exists():
    solids_vv = pd.read_hdf(SOLIDS_VV_H5, 'table')
else:
    _, solids_pts, solids_ds, _ = rp.setup_solids()
    solids_vv = (
        solids_pts
        .loc[lambda x: x.rho > 0]
        .assign(
            alpha=lambda x: alpha_kin(x.rho, x.rho_grad_norm, x.kin_dens),
            rgrad=lambda x: reduced_grad(x.rho, x.rho_grad_norm),
            vvpol=lambda x: vv_pol(x.rho, x.rho_grad_norm),
        )
        .groupby('label')
        .apply(lambda x: x.assign(
            group=solid_groups[solids_ds.df.loc(0)[x.name, 1.].group]
        ))
    )
    solids_vv.to_hdf(SOLIDS_VV_H5, 'table')

fig, axes = plt.subplots(
    2, 3, figsize=(4, 3), gridspec_kw=dict(hspace=0.3, wspace=0.1)
)
grouped = solids_vv.groupby('group')
for ax, group in zip(axes.flat, solid_groups.values()):
    df = grouped.get_group(group)
    im = rp.plot_rgrad_alpha(ax, df, norm=62)[-1]
    ax.set_title(group)
    ax.set_xticks([0, 0.3])
    ax.set_yticks([0, 1, 10])
for i, j in product(range(axes.shape[0]), range(axes.shape[1])):
    ax = axes[i, j]
    if (i, j) in {(1, 0), (1, 1), (0, 2)}:
        ax.set_xticklabels([0, 0.3])
        ax.set_xlabel(r'$s[n]$')
    else:
        ax.set_xticklabels([])
    if j == 0:
        ax.set_yticklabels([0, 1, 10])
        ax.set_ylabel(r'$\chi[n]$')
    else:
        ax.set_yticklabels([])
fig.colorbar(im, ax=axes[-1, -1], fraction=1)
axes[0, 2].set_xlabel(r'$\hspace{2}s[n]$')
axes[-1, -1].set_visible(False)

savefig(fig, 'solids-hists')

aims_data_s66, _, alpha_vvs_s66 = rp.setup_s66()
aims_data_x23, _, _ = rp.setup_x23()

fig, axes = plt.subplots(
    1, 3, figsize=(4, 1.3), gridspec_kw=dict(wspace=0.1), sharey=True
)
payload = zip(
    axes,
    ['monomer', 'dimer', 'crystal'],
    [1, 2, 4],
    [
        aims_data_s66.loc(0)['Benzene ... AcOH', 1.0, 'fragment-1'],
        aims_data_s66.loc(0)['Benzene ... Benzene (pi-pi)', 1.0, 'complex'],
        aims_data_x23.loc(0)['Benzene', 1.0, 'crystal'],
    ]
)
for ax, label, nmol, row in payload:
    df = pd.read_hdf(row.gridfile)
    df = (
        df
        .loc[lambda x: x.rho > 0]
        .assign(kin_dens=lambda x: x.kin_dens/2)
        .assign(
            alpha=lambda x: alpha_kin(x.rho, x.rho_grad_norm, x.kin_dens),
            rgrad=lambda x: reduced_grad(x.rho, x.rho_grad_norm),
            vvpol=lambda x: vv_pol(x.rho, x.rho_grad_norm)/nmol,
        )
    )
    rp.plot_rgrad_alpha(ax, df)
    ax.set_title(label)
    ax.set_xticks([0, 0.3])
    ax.set_xticklabels([0, 0.3])
    ax.set_xlabel(r'$s[n]$')
    ax.set_yticks([0, 1, 10])
    ax.set_yticklabels([0, 1, 10])
axes[0].set_ylabel(r'$\chi[n]$')

savefig(fig, 'benzene')

VDW_PARAMS_S66_H5 = Path('data/vdw-params-s66.h5')
if VDW_PARAMS_S66_H5.exists():
    vdw_params_s66 = pd.read_hdf(VDW_PARAMS_S66_H5, 'table')
else:
    vdw_params_s66 = pd.concat(
        dict(rp.evaluate_mbd_specs(
            [
                {'beta': np.nan, '_label': 'TS'},
                {'scs': True, 'beta': 0.83, '_label': 'rsSCS'},
                {'vv': True, 'C_vv': 0.0101, 'beta': np.nan, '_label': 'VV'},
                {'vv': 'nm', 'C_vv': 0.0101, 'beta': np.nan, '_label': 'nmVV'},
                {'vv': 'lg', 'C_vv': 0.0101, 'beta': np.nan, '_label': 'lgVV'},
                {'vv': 'lg2', 'C_vv': 0.0101, 'beta': np.nan, '_label': 'lg2VV'},
            ],
            aims_data_s66,
            alpha_vvs_s66,
            get_vdw_params=True,
        ).ene),
        names='label scale fragment method'.split()
    )
    vdw_params_s66.to_hdf(VDW_PARAMS_S66_H5, 'table')
vdw_params_s66 = (
    vdw_params_s66
    .rename_axis('quantity', axis=1)['alpha_0 C6'.split()].stack()
    .unstack('i_atom').sum(axis=1).unstack('fragment')
    .pipe(lambda x: (x['complex']-x['fragment-1']-x['fragment-2']) /
          (x['fragment-1']+x['fragment-2']))
    .unstack()
)

fig, axes = plt.subplots(
    1, 4, figsize=(4.3, 1), gridspec_kw=dict(wspace=0.1), sharey=True
)
payload = zip(axes, ['TS', 'rsSCS', 'VV', 'lg2VV'])
for ax, method in payload:
    df = vdw_params_s66.loc(0)[:, :, method]
    ax.hist(df['alpha_0'], histtype='step', bins=np.linspace(-.02, .12, 20))
    ax.hist(df['C6'], histtype='step', bins=np.linspace(-.02, .12, 20))
    ax.set_xticks([0, 0.1])
    ax.set_xticklabels(['0%', '10%'])
    ax.set_xlabel(r'$\Delta_\mathrm{rel}$')
    ax.set_yticks([])
    ax.set_title({'lg2VV': r"$\mathrm{VV}\prime$"}.get(method, method))
axes[0].legend(
    labels=[r'$\alpha_0$', '$C_6$'],
    loc='upper center',
    ncol=2,
    bbox_to_anchor=(0.5, -0.35),
    bbox_transform=fig.transFigure,
)

savefig(fig, 'pol-shifts')
