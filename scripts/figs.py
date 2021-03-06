#!/usr/bin/env python
from pathlib import Path
from itertools import product
from functools import partial

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from mbdvv.physics import ion_pot, vv_pol, alpha_kin, lg_cutoff2
from mbdvv.app import kcal, ev
import mbdvv.report as rp

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style='ticks', context='paper')
mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('mathtext', fontset='stix')

MBDSCAN_DATA = pd.HDFStore('data/mbd-scan-data.h5')


def savefig(fig, name, ext='pdf', **kwargs):
    fig.savefig(
        f'pub/figs/{name}.{ext}',
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
ax.axhline(1, color='black', linewidth=0.5, zorder=-1)
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
VDW_ENERGIES_SOLIDS_H5 = Path('data/vdw-energies-solids.h5')
if not SOLIDS_VV_H5.exists() or not VDW_ENERGIES_SOLIDS_H5.exists():
    aims_data_solids, solids_pts, solids_ds, alpha_vvs_solids = rp.setup_solids()
if SOLIDS_VV_H5.exists():
    solids_vv = pd.read_hdf(SOLIDS_VV_H5, 'table')
else:
    solids_vv = (
        solids_pts
        .loc[lambda x: x.rho > 0]
        .assign(
            alpha=lambda x: alpha_kin(x.rho, x.rho_grad_norm, x.kin_dens),
            ion_pot=lambda x: ion_pot(x.rho, x.rho_grad_norm),
            vvpol=lambda x: vv_pol(x.rho, x.rho_grad_norm),
        )
        .groupby('label')
        .apply(lambda x: x.assign(
            group=solid_groups[solids_ds.df.loc(0)[x.name, 1.].group]
        ))
    )
    solids_vv.to_hdf(SOLIDS_VV_H5, 'table')
if VDW_ENERGIES_SOLIDS_H5.exists():
    vdw_energies_solids = pd.read_hdf(VDW_ENERGIES_SOLIDS_H5, 'table')
else:
    vdw_energies_solids = rp.specs_to_binding_enes(
        [
            {},
            {'rpa': True, 'scs': True, 'beta': 0.83, '_label': 'MBD@rsSCS'},
            {
                'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.81, 'Rvdw_scale_vv': 'cutoff',
                'Rvdw17_base': True, 'vv_norm': 'aims', '_label': 'MBD-NL'
            },
        ],
        solids_ds,
        aims_data_solids.loc(0)[:, 1.],
        alpha_vvs_solids,
        unit=ev,
    )
    vdw_energies_solids.to_hdf(VDW_ENERGIES_SOLIDS_H5, 'table')

fig, axes = plt.subplots(
    2, 3, figsize=(3.7, 2.8), gridspec_kw=dict(hspace=0.3, wspace=0.1)
)
grouped = solids_vv.groupby('group')
for ax, group in zip(axes.flat, solid_groups.values()):
    df = grouped.get_group(group)
    im = rp.plot_ion_alpha(ax, df, norm=62)[-1]
    ax.set_title(group, fontdict={'fontsize': 8.6})
    ax.set_xticks([0, 0.5])
    ax.set_yticks([0, 1, 10])
for i, j in product(range(axes.shape[0]), range(axes.shape[1])):
    ax = axes[i, j]
    if (i, j) in {(1, 0), (1, 1), (0, 2)}:
        ax.set_xticklabels([0, 0.5])
        ax.set_xlabel(r'$I[n]/E_{\mathrm{h}}$')
    else:
        ax.set_xticklabels([])
    if j == 0:
        ax.set_yticklabels([0, 1, 10])
        ax.set_ylabel(r'$\chi[n]$')
    else:
        ax.set_yticklabels([])
fig.colorbar(im, ax=axes[-1, -1], fraction=1)
axes[0, 2].set_xlabel(r'$\hspace{2}I[n]/E_{\mathrm{h}}$')
axes[-1, -1].set_visible(False)

savefig(fig, 'solids-hists')

aims_data_s66, s66_ds, alpha_vvs_s66 = rp.setup_s66()
aims_data_x23, _, _ = rp.setup_x23()

fig, axes = plt.subplots(
    1, 3, figsize=(3.7, 1.2), gridspec_kw=dict(wspace=0.1), sharey=True
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
            ion_pot=lambda x: ion_pot(x.rho, x.rho_grad_norm),
            vvpol=lambda x: vv_pol(x.rho, x.rho_grad_norm)/nmol,
        )
    )
    rp.plot_ion_alpha(ax, df)
    ax.set_title(label)
    ax.set_xticks([0, 0.5])
    ax.set_xticklabels([0, 0.5])
    ax.set_xlabel(r'$I[n]/E_{\mathrm{h}}$')
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
                {'vv': 'lg2', 'C_vv': 0.0093, 'beta': np.nan, '_label': 'lg2VV'},
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
    .pipe(
        lambda x: (x['complex']-x['fragment-1']-x['fragment-2'])
        / (x['fragment-1']+x['fragment-2'])
    )
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

VDW_ENERGIES_S66_H5 = Path('data/vdw-energies-s66.h5')
if VDW_ENERGIES_S66_H5.exists():
    vdw_energies_s66 = pd.read_hdf(VDW_ENERGIES_S66_H5, 'table')
else:
    vdw_energies_s66 = rp.specs_to_binding_enes([
        {},
        {'scs': True, 'beta': 0.83, '_label': 'MBD@rsSCS'},
        {
            'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.81, 'Rvdw_scale_vv': 'cutoff',
            'Rvdw17_base': True, 'vv_norm': 'nonsph', '_label': 'MBD-NL'
        },
    ], s66_ds, aims_data_s66, alpha_vvs_s66, free_atoms_vv, unit=kcal)
    vdw_energies_s66.to_hdf(VDW_ENERGIES_S66_H5, 'table')

vdw_energies_s66 = vdw_energies_s66.append(
    MBDSCAN_DATA['/scf'].loc(0)['S66x8'].loc(0)[:, :, 'pbe']
    .merge(
        MBDSCAN_DATA['/vv10']
        .loc(0)['S66x8']
        .loc(0)[:, :, :, ['base', 'vdw']]['ene']
        .unstack()
        .pipe(lambda x: x['vdw'] - x['base'])
        .unstack()
        .apply(lambda x: interp1d(x.index, x), axis=1)
        .to_frame('vdw'),
        on='system dist'.split()
    )
    .rename_axis(['label', 'scale'])
    .assign(vdw=lambda df: df.apply(lambda x: float(x['vdw'](6.8)), axis=1))
    .assign(method='PBE+VV10')
    .set_index('method', append=True)
    .assign(ene=lambda x: x['ene'] + x['vdw'])
    .assign(
        delta=lambda x: x['ene'] - x['ref'],
        reldelta=lambda x: (x['ene'] - x['ref'])/abs(x['ref']),
    )
)

with sns.color_palette(list(reversed(sns.color_palette('coolwarm', 8)))):
    g = sns.catplot(
        data=vdw_energies_s66.reset_index(),
        kind='box',
        x='method',
        y='reldelta',
        hue='scale',
        order='PBE PBE+MBD@rsSCS PBE+MBD-NL PBE+VV10'.split(),
        aspect=1.6,
        height=1.75,
        margin_titles=True,
        fliersize=1,
        linewidth=0.8,
    )
g.ax.axhline(color='black', linewidth=0.5, zorder=-1)
g.set(ylim=(-.5, .5))
g.set_xticklabels(rotation=30, ha='right')
g.set_xlabels('')
g.set_ylabels(r'$\Delta E_i/E_i^\mathrm{ref}$')
g.set(yticks=[-.3, -.1, 0, .1, .3])
g.set_yticklabels([r'$-30\%$', r'$-10\%$', '0%', '10%', '30%'])
g._legend.set_title(r'$R/R_\mathrm{eq}$')
savefig(g, 's66-errors')

g = sns.catplot(
    data=vdw_energies_solids.reset_index(),
    kind='box',
    x='method',
    y='reldelta',
    hue='group',
    order='PBE PBE+MBD@rsSCS PBE+MBD-NL'.split(),
    aspect=2,
    height=1.4,
    margin_titles=True,
    fliersize=1,
    linewidth=.75,
)
g.ax.axhline(color='black', linewidth=0.5, zorder=-1)
g.set(ylim=(-.31, .21))
g.set_xticklabels(rotation=30, ha='right')
g.set_xlabels('')
g.set_ylabels(r'$\Delta E_i/E_i^\mathrm{ref}$')
g.set(yticks=[-.1, 0, .1])
g.set_yticklabels([r'$-10\%$', '0%', '10%'])
savefig(g, 'solids-errors')

results_surface = rp.setup_surface()

VDW_ENERGIES_SURFACE_H5 = Path('data/vdw-energies-surface.h5')
if VDW_ENERGIES_SURFACE_H5.exists():
    vdw_energies_surface = pd.read_hdf(VDW_ENERGIES_SURFACE_H5, 'table')
else:
    vdw_energies_surface = rp.evaluate_mbd_specs(
        [
            {},
            {
                'vv': 'lg2', 'beta': 0.83, 'Rvdw_scale_vv': 'cutoff',
                'vv_norm': 'aims', 'k_grid': (2, 2, 1), '_label': 'MBD-NL'
            },
            {
                'scs': True, 'beta': 0.83, 'vdw_ref': 'surf', 'k_grid': (2, 2, 1),
                '_label': 'MBD@rsSCS[surf]'
            },
            {
                'ts': True, 'param_a': 20, 'beta': 0.96, 'damping': 'fermi',
                'vdw_ref': 'surf', '_label': 'TS[surf]'
            },
        ],
        pd.DataFrame(results_surface).T.rename_axis('scale')
        .rename(columns={0: 'data', 1: 'gridfile'})
        .reset_index().assign(label='surface', fragment='all')
        .set_index('label scale fragment'.split())
    )
    vdw_energies_surface.to_hdf(VDW_ENERGIES_SURFACE_H5, 'table')

fig, ax = plt.subplots(figsize=(3, 2))
sns.lineplot(
    data=vdw_energies_surface.pipe(rp.ene_dft_vdw)['ene'].unstack('scale')
    .pipe(lambda df: df.apply(lambda y: y-df.iloc(1)[-1])).stack().to_frame('ene')
    .pipe(lambda df: df*ev).reset_index(),
    x='scale',
    y='ene',
    hue='method',
    ax=ax,
)
ax.axhline(color='black', linewidth=0.5, zorder=-1)
ax.set_xlim(2.7, 10)
ax.set_ylim(None, 0.1)
ax.set_xlabel(r'$d$(surface–molecule)$/\mathrm{\AA}$')
ax.set_ylabel(r'$E/\mathrm{eV}$')
ax.set_xticks([3, 9])
ax.set_yticks([-0.5, 0])
savefig(fig, 'surface')


def plot_cutoff(ax, cutoff):
    ion_pot = np.linspace(0, 1, 1000)
    alpha = np.linspace(0, 12, 1000)
    ax.set_xlabel(r'$I/E_\mathrm{h}$')
    ax.set_ylabel(r'$\chi$')
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 1, 10])
    return ax.contourf(
        ion_pot, alpha, cutoff(ion_pot, alpha[:, None]),
        np.linspace(0, 1, 10)
    )


fig, ax = plt.subplots(figsize=(3.2, 1.8))
cs = plot_cutoff(ax, lg_cutoff2)
cbar = fig.colorbar(cs)
cbar.set_ticks([0, 1])
savefig(fig, 'cutoff-func')
