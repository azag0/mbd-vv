# ::hide
# vim: set fo=croqnj com=b\:#>,b\:# spell:
from itertools import product
import warnings
from functools import partial
import importlib
from pkg_resources import resource_filename, resource_stream
from fractions import Fraction
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from mbdvv.app import kcal, ev
from vdwsets import get_x23

from mbdvv.physics import ion_pot, vv_pol, alpha_kin, nm_cutoff, \
    lg_cutoff, lg_cutoff2
import mbdvv.report as rp

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

importlib.reload(rp)

pd.options.display.max_rows = 999

# ::%config InlineBackend.figure_format = 'svg'
# ::%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 300}
sns.set(
    style='ticks',
    context='notebook', rc={
        'axes.formatter.useoffset': False,
    }
)
mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('mathtext', fontset='stix')

warnings.filterwarnings(
    'ignore',
    'Sorting because non-concatenation axis is not aligned',
    FutureWarning
)
warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)

# ::>
# ## Introduction
#
# The basic proposition of our new model, dubbed MBD@VV, is to parameterize the
# many-body dispersion (MBD) Hamiltonian with the Vydrov–Van Voorhis (VV)
# polarizability functional. The MBD Hamiltonian  describes a system of harmonic
# oscillators $\boldsymbol\xi_i$ characterized by their static polarizabilities
# $\alpha_{0,i}$ and frequencies $\omega_i$ and interacting via a long-range
# dipole potential $\mathbf T^\text{lr}$,
# $$
# H^\text{MBD}(\{\alpha_{0,i},\omega_i\})=\sum_i-\frac12\nabla_{\xi_i}^2+\sum_i\frac12\omega_i^2\xi_i^2 +\frac12\sum_{ij}\omega_i\omega_j\sqrt{\alpha_{0,i}\alpha_{0,j}}\boldsymbol{\xi}_i\cdot\mathbf T^\text{lr}_{ij}\boldsymbol{\xi}_j
# $$
# We obtain the oscillator parameters by coarse-graining the VV polarizability
# functional ($C=0.0093$),
# $$
# \alpha^\text{VV}[n](\mathrm
# iu)=\frac{n}{\frac{4\pi}3n+C\frac{|\boldsymbol\nabla n|^4}{n^4}+u^2} \tag{1}
# $$
# to Hirshfeld fragments $w_i^\text{H}$,
# $$
# \alpha_i^\text{VV}(\mathrm iu)=\int\mathrm d\mathbf r\,w_i^\text{H}(\mathbf r)\alpha^\text{VV}[n](\mathbf r,\mathrm iu)
# $$
# The effective oscillator frequencies are calculated such as to reproduce the
# same $C_6$ coefficients for the oscillators as if calculated directly from
# $\alpha_i(\mathrm iu)$ via the Casimir–Polder formula,
# $$
# \omega_i=\frac43\frac{C_{6,i}}{\alpha_i(0)^2},\qquad
#   C_{6,i}=\frac3\pi\int_0^\infty\mathrm du\,\alpha_i(\mathrm iu)^2
# $$
# The same definition of $\mathbf T^\text{lr}$ as in the
# MBD@rsSCS variant is used,
# $$
# \mathbf T_{ij}^\text{lr}=\frac1{1+\exp\left(-6\Big(\frac{|\mathbf r|}{\beta(R_i^\text{vdw}+R_j^\text{vdw})}-1\Big)\!\right)}\boldsymbol\nabla\otimes\boldsymbol\nabla\frac1{|\mathbf r|}\Bigg|_{\mathbf r=\mathbf R_j-\mathbf R_i}
# $$
# In the following sections, we will reuse the vdW radii derived from Hirshfeld
# volumes that are used in MBD@rsSCS,
# $$
# R_i^\text{vdW}=R_i^\text{vdW,free}\left(\frac{V_i^\text{H}}{V_i^\text{H,free}}\right)^\frac13
# $$
# We will investigate other options in the penultimate section.
#
# This plain combination of the MBD approach and VV polarizability functional
# already improves description of ionic systems compared to previous MBD
# parameterizations (Table 2). But when combined with DFT, it lacks in accuracy
# for covalent systems with respect to state-of-the-art vdW methods (Table 1),
# and suffers from a double-counting of electron correlation in metallic systems
# (slowly-varying electron density regions) (Table 2).
#
# On the S66x8 set, MARE of plain MBD@VV is 14% compared to 10% of MBD@rsSCS. On
# the data set of 63 solids, plain PBE underbinds semiconductors and ionic
# solids by 4% and 3%, respectively, binds main-group metals and
# transition-metal carbides and nitrides sufficiently, and overbinds transition
# metals by 4%. PBE+MBD@rsSCS fails for almost all ionic solids, all main-group
# metals, and carbides and nitrides, and half of transition metals. Rescaling
# the RPA-MBD eigenvalues as suggested by Gould et al. reduces the number of
# failures to only a half for ionic solids and a half for carbides and nitrides,
# and to zero for main-group metals and transition metals. But the calculated
# ionic solids are overbound by 11%, transition metals by 20%, and main-group
# metals by 250%. In contrast, the plain PBE+MBD@VV method does not fail for
# any of the solids systems. It binds semiconductors and ionic solids
# sufficiently. But it still overbinds main-group metals and transition metals
# by 14% and 9%, respectively.

# ::hide
aims_data_s66, s66_ds, alpha_vvs_s66 = rp.setup_s66()

# ::>

# ::hide
aims_data_solids, solids_pts, solids_ds, alpha_vvs_solids = rp.setup_solids()

# ::>

# ::hide
aims_data_x23, x23_ds, alpha_vvs_x23 = rp.setup_x23()

# ::>
# **Table 1: Performance on the S66x8 data set.** The mean relative errors (MRE)
# and mean absolute relative errors (MARE) of three methods are shown: the plain
# PBE functional, the standard PBE+MBD@rsSCS method (with optimal damping
# parameter $\beta=0.83$), and an MBD method based on the VV polarizability
# functional ($C=0.0093$), without any screening, with optimal $\beta=0.87$.

# ::hide
energies_s66_intro = rp.specs_to_binding_enes([
    {},
    {'scs': True, 'beta': 0.83},
    {'vv': True, 'C_vv': 0.0093, 'beta': 0.88},
], s66_ds, aims_data_s66, alpha_vvs_s66, unit=kcal)

# ::>

# ::hide
energies_s66_intro.groupby('method', sort=False) \
    .apply(rp.dataset_stats)[['MRE', 'MARE']].round(3)

# ::>
# **Table 2: Performance on the data set of 63 solids.** $N$ is the number of
# solids from a given group for which the method does not fail. The groups are
# ionic crystals (IC), main-group metals (MM), semiconductors (SC), transition
# metals (TM), and TM carbides and nitrides (TMCN). In addition to methods from
# the previous table, the MBD@rsSCS method is also evaluated by rescaling the
# RPA-MBD eigenvalues as suggested by Gould et al (the `rpa` flag).

# ::hide
energies_solids_intro = rp.specs_to_binding_enes(
    [
        {},
        {'scs': True, 'beta': 0.83, 'rpa': True},
        {'scs': True, 'beta': 0.83},
        {'vv': True, 'C_vv': 0.0093, 'beta': 0.88},
    ],
    solids_ds,
    aims_data_solids.loc(0)[:, 1.],
    alpha_vvs_solids,
    unit=ev,
)

# ::>

# ::hide
energies_solids_intro.groupby('group method'.split(), sort=False) \
    .apply(rp.dataset_stats).unstack('group').swaplevel(0, 1, 1) \
    .loc(1)[:, ['N', 'MRE']].sort_index(1).round(3)

# ::>
# To solve these two issues, we borrow two techniques from the VV10 nonlocal
# functional and the TS method. First, we subtract the portion of the
# polarizability that comes from metallic electron density regions. Second, we
# normalize the atomic VV polarizabilities and $C_6$ coefficients to reproduce
# the respective exact quantities for free atoms.
#
# ## MBD trivia
#
# Before proceeding with the two improvements of the plain MBD@VV approach, we
# present the performance of several trivial modifications of MBD@rsSCS on the
# S66x8 set. The sensitivity of the method to to the change of reference
# free-atom parameters is small, but not entirely negligible. Using the
# definition of free-atom vdW radii from free-atom static polarizabilities using
# the following simple formula introduces an even smaller change,
# $$
# R_i^\text{vdW,free}=\tfrac52(\alpha_{0,i}^\text{free})^\frac17
# $$
# Using just the 2nd order of the MBD energy does not decrease the overall
# accuracy, but it leads to stronger overbinding.
#
# **Table 3: Performance on the S66x8 set.** Plain PBE, PBE+MBD@rsSCS, and
# PBE+TS are shown for comparison. `vdw_ref[BG]` uses the static
# polarizabilities and $C_6$ coefficients of free atoms from Bučko and Gould.
# `17base` uses the definition of vdW radii from static polarizabilities,
# `rpa,ord2` uses just the 2nd-order energy from the many-body expansion.

# ::hide
rp.specs_to_binding_enes([
    {},
    {'scs': True, 'beta': 0.83},
    {'scs': True, 'beta': 0.83, 'vdw_ref': 'BG'},
    {'scs': True, 'beta': 0.83, 'Rvdw17_base': True},
    {'scs': True, 'beta': 0.83, 'rpa': True, 'ord2': True},
    {
        'ts': True, 'param_a': 20, 'beta': 0.96, 'damping': 'fermi',
        '_label': 'TS'
    },
], s66_ds, aims_data_s66, alpha_vvs_s66, unit=kcal) \
    .groupby('method', sort=False) \
    .apply(rp.dataset_stats)[['MRE', 'MARE']].round(3)


# ::>
# The use of the vdW radii from static polarizabilities has a substantial effect
# on the performance for main-group metals, reducing MRE from $-14\%$ to $-5\%$ for
# MBD@VV.
#
# **Table 4: Performance on the data set of 63 solids.**

# ::hide
rp.specs_to_binding_enes(
    [
        {},
        {'vv': True, 'C_vv': 0.0093, 'beta': 0.88},
        {'vv': True, 'C_vv': 0.0093, 'beta': 0.88, 'Rvdw17_base': True},
    ],
    solids_ds,
    aims_data_solids.loc(0)[:, 1.],
    alpha_vvs_solids,
    unit=ev,
).groupby('group method'.split(), sort=False) \
    .apply(rp.dataset_stats).unstack('group').swaplevel(0, 1, 1) \
    .loc(1)[:, ['N', 'MRE']].sort_index(1).round(3)


# ::>
# We also verify that the second-order expansion of MBD@rsSCS is equivalent to
# TS@rsSCS and of MBD@TS to TS with the proper choice of damping

# ::hide
rp.specs_to_binding_enes([
    {},
    {
        'ts': True, 'param_a': 20, 'beta': 0.96, 'damping': 'fermi',
        '_label': 'TS'
    },
    {
        'rpa': True, 'ord2': True, 'damping': 'sqrtfermi,dip',
        'param_a': 20, 'beta': 0.96
    },
    {
        'rpa': True, 'ord2': True, 'scs': True, 'beta': 0.83,
        '_label': 'MBD@rsSCS, 2nd order'
    },
    {
        'ts': True, 'scs': True, 'beta': 0.83, 'param_a': 6,
        'damping': 'fermi2', 'scr_damping': 'fermi,dip,gg'
    },
], s66_ds, aims_data_s66, alpha_vvs_s66, unit=kcal) \
    .groupby('method', sort=False) \
    .apply(rp.dataset_stats)[['MRE', 'MARE']].round(3)


# ::>
# ## Optimizing VV polarizability functional
#
# First we look into the parametrization of the VV polarizability functional in
# (1). The value of the parameter $C$ was original set to 0.0089 by minimizing
# MARE on a set of 17 $C_6$ coefficients, of which 7 are free atoms, 6 diatomic
# molecules, and the heaviest molecule is CS$_2$. In a later work, they
# reparametrized the functional by extending the set of reference $C_6$
# coefficients to 54, now including 25 free atoms and several smaller
# hydrocarbons and carbohydrates, up to benzene. This led to $C$ equal to
# 0.0093. Here, we extend the functional form of the original functional,
# and reparametrize the coefficients on a much larger set of 1081 $C_6$
# coefficients of dimers of 46 organic molecules.
#
# The VV polarizability functional can be rewritten as
# $$
# \alpha^\text{VV}[n](\mathrm iu)=
# \frac{n}{\frac{4\pi}3n+\left(C_1\frac{\tau_\text W[n]}n\right)^2+u^2}
# $$
# where $C_1\equiv 8\sqrt{C}\doteq3/4$ and $\tau_\text W[n]\equiv|\boldsymbol\nabla
# n|^2/8n$ is the von Weizsäcker kinetic energy density, which is the exact
# functional for single-orbital electron densities. ($\tau_\text W/n$ is
# equivalent to the “local ionization potential” discussed by VV.) Hence, the
# expression in the parentheses can be interpreted as a scaled reduced kinetic
# energy density.  This motivates extending the functional form to either of the
# following two
# $$
# \begin{aligned}
# \alpha{^\text{VV}}'[n](\mathrm iu)&=
# \frac{n}{\frac{4\pi}3n+\left(\frac{C_1\tau_\text
# W[n]+C_2\tau_\text{KS}[n]}n\right)^2+u^2} \\
# \alpha{^\text{VV}}''[n](\mathrm iu)&=
# \frac{n}{\frac{4\pi}3n+\left(C_1\frac{\tau_\text
# W[n]}n\right)^2+\left(C_2\frac{\tau_\text{KS}[n]}n\right)^2+u^2}
# \end{aligned}
# $$
# where $\tau_\text{KS}$ is the Kohn–Sham kinetic energy density.
#
# Figure 1 shows that in terms of MARE on the $C_6$ dataset, the two kinetic
# energy densities contain the same information and as a result $C_1$ and $C_2$
# are almost perfectly linearly dependent. The lowest MARE of 4.0% is achieved for
# $(C_1,C_2)=(0.73,-0.02)$, but omitting the KS kinetic energy entirely gives
# almost as good performance of 4.3% for $(C_1,C_2)=(0.82,0)$. For comparison,
# $C_1=0.82$ corresponds to $C=0.0101$, a further increase from the original
# 0.0089 followed by 0.0094. Similar results are obtained for VV$''$, with the
# only difference being that summing squares of the kinetic energies breaks
# the linear dependence of the two coefficients, but the resulting performance
# is unaffected.

# ::hide
def my_pol(n, grad, kin, C1=8*np.sqrt(0.0093), C2=0., u=0.):
    tw = grad**2/(8*n)
    return n/(4*np.pi/3*n+((C1*tw+C2*kin)/n)**2+u**2)


def my_pol_2(n, grad, kin, C1=8*np.sqrt(0.0093), C2=0., u=0.):
    tw = grad**2/(8*n)
    return n/(4*np.pi/3*n+((C1*tw)**2+(C2*kin)**2)/n**2+u**2)


# ::>

# ::hide
coords_C6, hirsh_C6, ref_C6, C6_set_pts = rp.setup_C6_set()

# ::>

# ::hide
C6_my_pol_scan = pd.concat({
    (C1, C2): rp.evaluate_func_on_C6(
        partial(my_pol, C1=C1, C2=C2), C6_set_pts, ref_C6
    ) for C1, C2 in tqdm(list(product(
        [0.75, 0.77, .79, .81, .83],
        [-.01, -.005, 0, .005, .01],
    )))
}, names='C1 C2'.split())

# ::>

# ::hide
fig, ax = plt.subplots()
rp.plot_matrix_df(
    ax,
    C6_my_pol_scan.groupby('C1 C2'.split())
    .apply(rp.dataset_stats)['MARE'].unstack('C1'),
    levels=[.05, .06, .07, .085, .1, .13, .17]
)

# ::>
# **Figure 1: MARE($C_1$, $C_2$) of VV$'$ on the $C_6$ dataset.**

# ::hide
C6_my_pol_scan.groupby('C1 C2'.split()).apply(rp.dataset_stats)[['MARE']] \
    .unstack('C2')['MARE'].pipe(rp.findmin)

# ::>

# ::hide
C6_my_pol_scan.groupby('C1 C2'.split()).apply(rp.dataset_stats)['MARE'] \
    .loc[:, 0.].pipe(rp.findmin)

# ::>

# ::hide
C6_my_pol_2_scan = pd.concat({
    (C1, C2): rp.evaluate_func_on_C6(
        partial(my_pol_2, C1=C1, C2=C2), C6_set_pts, ref_C6
    ) for C1, C2 in tqdm(list(product(
        [0.75, 0.77, .79, .81, .83],
        [0, .02, .04, .06, .08],
    )))
}, names='C1 C2'.split())

# ::>

# ::hide
fig, ax = plt.subplots()
rp.plot_matrix_df(
    ax,
    C6_my_pol_2_scan.groupby('C1 C2'.split())
    .apply(rp.dataset_stats)['MARE'].unstack('C1'),
    levels=[.05, .06, .07, .085, .1, .13, .17]
)

# ::>
# **Figure 2: MARE($C_1$, $C_2$) of VV$''$ on the $C_6$ dataset.**

# ::hide
C6_my_pol_2_scan.groupby('C1 C2'.split()).apply(rp.dataset_stats)[['MARE']] \
    .unstack('C2')['MARE'].pipe(rp.findmin)

# ::>

# ::hide
C6_my_pol_2_scan.groupby('C1 C2'.split()).apply(rp.dataset_stats)['MARE'] \
    .loc[:, 0.].pipe(rp.findmin)

# ::>
# The optimized VV functional leads to slightly improved accuracy of MBD@VV on
# the S66 set, especially in the long range, but it is still lacking behind the
# state-of-the-art MBD@rsSCS method.

# ::hide
energies_s66_vvopt = rp.specs_to_binding_enes([
    {},
    {'scs': True, 'beta': 0.83},
    {'vv': True, 'C_vv': 0.0093, 'beta': 0.88},  # optimal beta
    {'vv': True, 'C_vv': 0.0101, 'beta': 0.87},  # optimal beta
], s66_ds, aims_data_s66, alpha_vvs_s66, unit=kcal)

# ::>

# ::hide
energies_s66_vvopt.groupby('method').apply(rp.dataset_scale_stats) \
    .loc(1)[:, ['MRE', 'MARE']].round(3)


# ::>
# ## Dealing with metallic electrons
#
# Most exchange–correlation functionals are exact for the uniform electron gas
# by construction and as a result, describe accurately the electron correlation
# *within* slowly-varying density regions, which can be found in metals.
# Furthermore, the interactions *between* such regions and other regions of the
# electron density are effectively screened by the conducting electrons.  At the
# same time, these metallic-density regions contribute dominantly to the total
# polarizability (see (1) and therefore, if used
# directly in any vdW model, would result in overpolarization and overbinding of
# metallic systems.  To avoid this double-counting of the electron correlation,
# we smoothly cut off the VV polarizability functional in metallic-density
# regions.  These regions can be distinguished using the combination of two
# local electron-density descriptors: the local gap $I$ and the
# iso-orbital indicator $\alpha$,
# $$
# s[n]=\frac{|\boldsymbol\nabla n|}{2(3\pi^2)^\frac13n^\frac43},\quad
#   \alpha[n]=\frac{\tau_\text{KS}[n]-\tau_\text{W}[n]}{\tau^\text{unif}(n)}
# $$
# where $\tau^\text{unif}$ is the kinetic energy density of a uniform electron
# gas.  In particular, the metallic density is characterized by $s\rightarrow0$
# and $\alpha\sim1$.
#
# Figures 3–5 present a numerical analysis of the contributions of different
# regions of the electron density (along $s$ and $\alpha$) to the total
# polarizability, as estimated by the VV polarizability functional. Figure 3
# shows that in small organic molecules, the vast majority of the polarizability
# comes from electron density with either $s>0.1$ and the small part that comes
# from low-gradient regions has $\alpha<1$ (centers of covalent bonds).
# Figure 4 shows how intermolecular interactions change polarizability
# contributions. The intermolecular regions (low gradient, large $\alpha$) are
# contributing a significant amount of polarizability, despite the fact that the
# electron density in those regions is low. This effect is more pronounced in
# the benzene crystal. The diamond crystal shares many features with the benzene
# compounds, corresponding to the fact that much of the polarizability in all
# four systems comes from tetravalent carbon atoms.

# ::hide
s66_subset_pts = (
    pd.concat((
        pd.read_hdf(aims_data_s66.loc(0)[lbl, 1.0, frag]['gridfile'])
        for lbl, frag in [('Benzene ... AcOH', 'fragment-1'),
                          ('Cyclopentane ... Cyclopentane', 'fragment-1'),
                          ('Ethyne ... Water (CH-O)', 'fragment-2'),
                          ('Uracil ... Neopentane', 'fragment-1')]
    )).loc[lambda x: (x.rho > 0)]
    .assign(kin_dens=lambda x: x.kin_dens/2)
    .assign(
        alpha=lambda x: alpha_kin(x.rho, x.rho_grad_norm, x.kin_dens),
        ion_pot=lambda x: ion_pot(x.rho, x.rho_grad_norm),
        vvpol=lambda x: vv_pol(x.rho, x.rho_grad_norm),
    )
)

# ::>

# ::hide
fig, ax = plt.subplots(figsize=(4, 3))
*_, img = rp.plot_ion_alpha(ax, s66_subset_pts)
fig.colorbar(img)

# ::>
# **Figure 3: Histogram of $s$ and $\alpha$ weighted by $\alpha^\text{VV}$ in
# four molecules from S66.** The plotted function is
# $H(s',\alpha')=\sum_{i=1}^4\int\mathrm d\mathbf r\delta(s(\mathbf r)-s')
# \delta(\alpha(\mathbf r)-\alpha')\alpha_i^\text{VV}(\mathbf r)$, its magnitude
# is mapped to the color intensity on a log scale. The four molecules are
# benzene, cyclopentane, ethyne, and uracil.


# ::hide
def plot_benzene(axes):
    payload = zip(
        axes,
        ['molecule', 'dimer', 'crystal', 'diamond'],
        [1, 2, 4, 1/6],
        [
            aims_data_s66.loc(0)['Benzene ... AcOH', 1.0, 'fragment-1'],
            aims_data_s66.loc(0)['Benzene ... Benzene (pi-pi)', 1.0, 'complex'],
            aims_data_x23.loc(0)['Benzene', 1.0, 'crystal'],
            aims_data_solids.loc(0)['C', 1.0, 'crystal'],
        ]
    )
    for (ax, ax2), label, nmol, row in payload:
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
        ax2.hist(
            df.ion_pot, range=(0, 1), bins=100, weights=df.vvpol*df.part_weight,
            lw=0,
        )


fig, axes = plt.subplots(4, 2, figsize=(5, 10))
plot_benzene(axes)


# ::>
# **Figure 4: Histogram of $s$ and $\alpha$ weighted by $\alpha^\text{VV}$ in
# carbon-based compounds.** The plots on the left are of the same type as in
# Figure 3, the plots on the right are weighted histograms of $s$ only. The four
# systems are benzene molecule, benzene dimer, benzene crystal, and diamond
# crystal. The plots are normalized to six carbon atoms.
#
# Figure 5 shows a much richer spectrum of patterns within the set of 63 simple
# solids.  Most similar to the organic molecules is the group of semiconductors,
# with an even larger amount of polarizability coming from regions with low
# gradient and high $\alpha$ compared to the benzene crystal. In contrast, the
# vast majority of the polarizability in main-group metals comes from the
# regions of the electron density that is close to the uniform electron gas
# ($s\sim0$, $\alpha\sim1$). In transition metals, the polarizability is
# distributed over a large range of the local gap along the $\alpha\sim1$
# line, with a larger part still coming from low-gradient regions. These
# features are largely shared by the transition-metal carbides and nitrides, but
# the whole distribution is shifted to larger values of $\alpha\sim2$.
# In ionic solids, which do not feature covalent bonds, most of the
# polarizability comes from single-orbital regions ($\alpha=0$), but a
# substantial amount comes also from noncovalent orbital overlap regions.

# ::hide
solid_groups = {
    'SC': 'semiconductors',
    'MM': 'main-group metals',
    'TM': 'transition metals',
    'TMCN': 'TM carbides & nitrides',
    'IC': 'ionic crystals',
}

g = sns.FacetGrid(
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
    )),
    col='group',
    col_wrap=3,
    col_order=solid_groups.values(),
    height=2.4,
)
g.map_dataframe(
    lambda x, y, data, **kwargs: rp.plot_ion_alpha(plt, data, norm=60),
    'local gap', 'alpha'
)


# ::>
# **Figure 5: Histogram of $s$ and $\alpha$ weighted by $\alpha^\text{VV}$ in 63
# solids.**

# ::>
# To avoid the double-counting of electron correlation in metallic systems, and
# based on the numerical analysis presented above, we propose a smooth cutoff
# function of $s$ and $\alpha$ that damps contributions to polarizability from
# the nonmetallic (nm) density regions. We investigate three variants (Figure 6):
# $g_\text{nm}$ cuts off only the metallic portion of the density, $g_\text{lg}$
# cuts off all low-gradient (lg) regions except for single-orbital ($\alpha<1$)
# regions, which includes also overlaps of density-tail regions (noncovalent
# orbital overlaps). $g_\mathrm{lg2}$ cuts of more of the low-gradient regions.
# The adapted polarizability function then takes the following form,
# $$
#   \alpha^\text{nmVV}[n]=g_\text{nm}(s,\alpha)\alpha^\text{VV}[n]
# $$
# The rationale for the $g_\text{lg}$ versions is that the contributions from the
# noncovalent orbital overlaps (see Figure 3) significantly increase the
# polarizability of bound molecules compared to isolated molecules. While this
# mechanism may reflect the reality to some degree, we have no evidence at the
# moment whether the VV polarizability functional captures it correctly. For
# this reason, we will investigate all three variants. The lg and lg2 version
# differ only in the degree to which they attempt to remove this effect.

# ::hide
def plot_cutoff(ax, cutoff, title):
    ion_pot = np.linspace(0, 1, 1000)
    alpha = np.linspace(0, 12, 1000)
    ax.set_xlabel(r'$I$')
    ax.set_ylabel(r'$\alpha$')
    ax.set_title(title)
    ax.label_outer()
    return ax.contourf(
        ion_pot, alpha, cutoff(ion_pot, alpha[:, None]),
        np.linspace(0, 1, 10)
    )


fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), sharex=True, sharey=True)
plot_cutoff(axes[0], nm_cutoff, r'$g_\mathrm{nm}(s,\alpha)$')
plot_cutoff(axes[1], lg_cutoff, r'$g_\mathrm{lg}(s,\alpha)$')
plot_cutoff(axes[2], lg_cutoff2, r'$g_\mathrm{lg2}(s,\alpha)$')


# ::>
# **Figure 6: Cutoff functions of nonmetallic electron density regions.** The
# color scale encodes 0 to dark violet and 1 to beige.
#
# The exact shape of the functions is to some degree arbitrary, but mostly can
# be characterized as being zero in the largest possible neighborhood around
# $(s,\alpha)=(0,1)$ without making $\alpha^\text{nmVV}[n]$ significantly
# smaller than $\alpha^\text{VV}[n]$ for organic molecules and semiconductors.
# $\alpha^\text{lgVV}[n]$ is then a straightforward extension to regions with
# $\alpha>1$.
#
# The three versions of the cutoff behave differently when applied to the S66 set
# (Table 4). The nonmetallic cutoff has a negligible effect on the
# performance. The weaker low-gradient cutoff introduces a slight improvement in
# the performance, and also requires somewhat smaller damping. Neither of these
# two cutoffs affects the scaled-distance regime. The strong long-gradient
# cutoff performs best, improving MARE by up to 4 percent
# points. This version requires substantially less damping ($\beta=0.77$) and
# also affects the scaled-distance regime. This suggests that the effect of
# increased polarizability in the complexes is not captured correctly by the VV
# functional. The MBD@lg2VV variant has still slightly wider ranges of errors on
# the S66, the overall MARE is comparable to MBD@rsSCS. It is also clear that
# the optimized version of the VV functional (the $C$ parameter) provides
# significantly better results, especially at long range.

# ::hide
energies_s66_vvnm = rp.specs_to_binding_enes([
    {},
    {'scs': True, 'beta': 0.83},
    {'vv': True, 'beta': 0.88, 'Rvdw17_base': True},
    {'vv': True, 'C_vv': 0.0093, 'beta': 0.88, 'Rvdw17_base': True},  # optimal beta
    {'vv': 'nm', 'C_vv': 0.0093, 'beta': 0.88, 'Rvdw17_base': True},  # optimal beta
    {'vv': 'lg', 'C_vv': 0.0093, 'beta': 0.85, 'Rvdw17_base': True},  # optimal beta
    {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.77, 'Rvdw17_base': True},  # optimal beta
    {'vv': True, 'C_vv': 0.0101, 'beta': 0.87, 'Rvdw17_base': True},  # optimal beta
    {'vv': 'nm', 'C_vv': 0.0101, 'beta': 0.87, 'Rvdw17_base': True},  # optimal beta
    {'vv': 'lg', 'C_vv': 0.0101, 'beta': 0.84, 'Rvdw17_base': True},  # optimal beta
    {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.76, 'Rvdw17_base': True},  # optimal beta
], s66_ds, aims_data_s66, alpha_vvs_s66, unit=kcal)

# ::>
# **Table 4: Performance on the S66x8 set.** The $\beta$ parameter minimizes
# MARE for each individual method.

# ::hide
energies_s66_vvnm.groupby('method').apply(rp.dataset_scale_stats) \
    .loc(1)[:, ['MRE', 'MARE']].round(3)


# ::>

# ::hide
with sns.color_palette(list(reversed(sns.color_palette('coolwarm', 8)))):
    g = sns.catplot(
        data=energies_s66_vvnm.reset_index(),
        kind='box',
        x='method',
        order=energies_s66_vvnm
        .index.get_level_values('method').unique()[[0, 1, 2, -1]],
        y='reldelta',
        hue='scale',
        aspect=2,
        height=2.5,
        margin_titles=True
    )
g.set(ylim=(-.5, .5))
g.set_xticklabels(rotation=30, ha='right')
g.set_xlabels('')
g.set_ylabels(r'$\Delta E_i/E_i^\mathrm{ref}$')
g.set(yticks=[-.3, -.1, 0, .1, .3])
g.set_yticklabels([r'$-30\%$', r'$-10\%$', '0%', '10%', '30%'])
g._legend.set_title('equilibrium\ndistance scale')

# ::>
# The magnitude of the effect of the increased polarizability of complexes vs.
# monomers is shown in Figure 7 and Table 5. Whereas with TS the changes in
# polarizability are within 3%  and 0% on average (and the short-range screening
# doesn't change that), with VV or nmVV they are up to 10%, on average at 6%.
# With the milder low-gradient cutoff, the average shift is reduced to 3%, and
# with the strong low-gradient cutoff, there is on average no change in the
# polarizability when forming the complexes.
#
# **Figure 7: Histograms of relative changes in polarizability from monomers
# to complexes on S66.** The plotted quantity is
# $(\alpha(\text{complex})-\alpha({\text{monomers}}))/\alpha({\text{monomers}})$.

# ::hide
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

# ::>

# ::hide
sns.FacetGrid(
    vdw_params_s66
    .rename_axis('quantity', axis=1)['alpha_0 C6'.split()].stack()
    .unstack('i_atom').sum(axis=1).unstack('fragment')
    .pipe(lambda x: (x['complex']-x['fragment-1']-x['fragment-2'])
          / (x['fragment-1']+x['fragment-2']))
    .to_frame('diff')
    .reset_index(),
    hue='quantity',
    col='method',
    col_wrap=3,
    col_order='VV nmVV lgVV lg2VV TS rsSCS'.split(),
    height=2,
).map(plt.hist, 'diff', histtype='step', bins=np.linspace(-.02, .12, 20)) \
    .add_legend()

# ::>
# **Table 5: Relative changes in polarizability from monomers to complexes on S66.**

# ::hide
(
    vdw_params_s66
    .rename_axis('quantity', axis=1)['alpha_0 C6'.split()].stack()
    .unstack('i_atom').sum(axis=1).unstack('fragment')
    .pipe(lambda x: (x['complex']-x['fragment-1']-x['fragment-2'])
          / (x['fragment-1']+x['fragment-2']))
    .groupby('method quantity'.split()).describe(percentiles=[.5]).unstack()
    .loc(0)['VV nmVV lgVV lg2VV TS rsSCS'.split()].stack()
    .round(3)
)


# ::>
# Finally, we check how much of the polarizability in the S66 set do the cutoffs
# remove in monomers. For nmVV and lgVV, this is negligible. For lg2VV, it is
# still only a 2% reduction in both static polarizabilities and C6 coefficients.
# Clearly the effects on the performance observed above do not come from any
# changes in vdW properties of the monomers.

# ::hide
(
    vdw_params_s66
    .rename_axis('quantity', axis=1)['alpha_0 C6'.split()].stack()
    .unstack('i_atom').sum(axis=1)
    .loc(0)[:, 1, 'fragment-1 fragment-2'.split()].unstack('method')
    .pipe(lambda x: x.apply(lambda y: (y-x['VV'])/x['VV'])).stack()
    .groupby('method quantity'.split()).describe(percentiles=[.5]).unstack()
    .loc(0)['nmVV lgVV lg2VV TS rsSCS'.split()].stack()
    .round(3)
)

# ::>
# Now that we understand the behavior on the S66 set, we proceed to the solids,
# which were the initial motivation to consider the metallic cutoff in the first
# place (Table 6). First, there is no difference between the nonmetallic
# and low-gradient versions of the cutoff: the noncovalent orbital overlaps do
# not play a significant role in solids. Second, the nonmetallic cutoff
# substantially improves the performance on main-group metals (MRE from $-5\%$ to
# 0%) and to a lesser degree on transition metals (MRE from $-8\%$ to $-5\%$).

# ::hide
energies_solids_vvnm = rp.specs_to_binding_enes(
    [
        {},
        {'scs': True, 'beta': 0.83},
        {'vv': True, 'C_vv': 0.0101, 'beta': 0.87, 'Rvdw17_base': True},  # optimal beta
        {'vv': 'nm', 'C_vv': 0.0101, 'beta': 0.87, 'Rvdw17_base': True},  # optimal beta
        {'vv': 'lg', 'C_vv': 0.0101, 'beta': 0.84, 'Rvdw17_base': True},  # optimal beta
        {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.76, 'Rvdw17_base': True},  # optimal beta
    ],
    solids_ds,
    aims_data_solids.loc(0)[:, 1.],
    alpha_vvs_solids,
    unit=ev,
)

# ::>
# **Table 6: Performance on the data set of 63 solids.**

# ::hide
energies_solids_vvnm.groupby('group method'.split(), sort=False) \
    .apply(rp.dataset_stats).unstack('group').swaplevel(0, 1, 1) \
    .loc(1)[:, ['N', 'MRE']].sort_index(1).round(3)

# ::>

# ::hide
vdw_params_solids = pd.concat(
    dict(rp.evaluate_mbd_specs(
        [
            {'beta': np.nan, '_label': 'TS'},
            {'vv': True, 'C_vv': 0.0101, 'beta': np.nan, '_label': 'VV'},
            {'vv': 'nm', 'C_vv': 0.0101, 'beta': np.nan, '_label': 'nmVV'},
            {'vv': 'lg', 'C_vv': 0.0101, 'beta': np.nan, '_label': 'lgVV'},
            {'vv': 'lg2', 'C_vv': 0.0101, 'beta': np.nan, '_label': 'lg2VV'},
        ],
        aims_data_solids.loc(0)[:, 1., 'crystal'],
        alpha_vvs_solids,
        get_vdw_params=True,
    ).ene.dropna()),
    names='label scale fragment method'.split()
).xs(1, level='scale').xs('crystal', level='fragment')

# ::>
# The cause of these improvements can be most easily seen on the effect that the
# nonmetallic effects have on the atomic C6 coefficients in the metals. The most
# drastic effect can be observed on sodium, where the C6 coefficient goes from
# 290 (VV) to 5 (lg2VV). For transition metals, the resulting C6 coefficients
# are even substantially lower than the vdW surf screened coefficients. This
# can be affected either be decreasing the nonmetallic cutoff, or by normalizing
# to free atoms, which we investigate in the next section.

# ::hide
(
    vdw_params_solids
    .append(
        pd.read_csv(resource_filename('mbdvv', 'data/vdw-surf.csv'))
        .assign(method='surf', i_atom=1).set_index('label method i_atom'.split())
    )
    .sort_index()
    .pipe(
        lambda df: df.assign(
            group=solids_ds.df['group'].xs(1, level='scale')
            .loc[df.index.get_level_values('label')].values
        )
    )
    .xs(1, level='i_atom')
    .set_index('group', append=True).swaplevel(0, 2).loc(0)['MM TM'.split()]['C6']
    .unstack('method').round()
)

# ::>
# ## Normalizing to reference free atoms
#
# The VV polarizability functional is only approximate, which is manifest
# already for free-atom polarizabilities and $C_6$ coefficients, where accurate
# reference values are known.  To mitigate this error, we normalize the VV
# quantities with the ratio of the free-atom polarizabilities and $C_6$
# coefficients as calculated by the VV functional and as given by reference
# calculations,
# $$
# \alpha_{0,i}^\text{rVV}=\alpha_{0,i}^\text{VV}\frac{\alpha_{0,i}^\text{ref,free}}{\alpha_{0,i}^\text{VV,free}},\quad
#   C_{6,i}^\text{rVV}=C_{6,i}^\text{rVV}\frac{C_{6,i}^\text{ref,free}}{C_{6,i}^\text{VV,free}}
# $$
# This correction assumes that any error in $\alpha^\text{VV}[n]$ (or
# $\alpha^\text{nm/lgVV}[n]$) is at least partially transferable from free atoms to
# atoms in molecules and materials.  Alternatively, this step can be seen as a
# modification of the TS method, in which the scaling of reference free-atom
# quantities by Hirshfeld volumes is replaced with that by VV-derived
# polarizabilities and $C_6$ coefficients.

# ::>

# ::hide
free_atoms_pts = rp.free_atoms_pts()

# ::>

# ::hide
free_atoms_pts = (
    free_atoms_pts
    .reset_index('species')
    .assign(
        species=lambda x: x['species']
        .astype(pd.api.types.CategoricalDtype(
            categories=x['species'].unique().tolist()
        ))
    )
    .set_index('species', append=True)
)


# ::>
# To understand what kinds of changes this normalization introduces, we
# plot the errors in the $C_6$ coefficients of free atoms with the unnormalized
# polarizability functionals (Figure 8). First, the TS and BG reference values
# agree with each other to within 10% up to Kr, and then again for 5th-row
# *p* elements, but there are significant discrepancies for 5th- and 5th-row
# *d* elements (includes Ag, Pd, Pt, Au). Second, the VV functional is
# relatively accurate for 1*s* and *p* elements, but substantially
# underestimates the polarizability of *s* and *d* atoms that form metals.
# Third, there is no substantial difference in evaluating the VV functional on
# spherical or nonspherical free-atom densities, except for several elements,
# crucially H, C, and N (Table 7). The spherical polarizabilities are always
# larger, because the spherical densities are more diffuse, at least with the
# PBE functional.

# ::hide
free_atoms_vv = (
    pd.concat({
        (C_vv, sph): rp.free_atoms_vv(
            free_atoms_pts.loc[sph], partial(my_pol, C1=8*np.sqrt(C_vv))
        )
        for C_vv, sph in product([0.0093, 0.0101], [True, False])
    }, names='C_vv spherical'.split())
    .merge(rp.vdw_params.query('N == Z'), left_on='species', right_index=True, how='outer')
    .sort_index()
)

# ::>
# **Figure 8: $C_6$ coefficients of free atoms across periodic table.**. TS and
# BG refer to the two sets of high-level reference values. `sph` and `nonsph`
# denotes whether the VV functional was evaluated on spherical
# (spin-unpolarized) or nonspherical (spin-polarized) atomic densities.


# ::hide
def plot_species(ax, df, what, label=None, relative=False):
    df = df.sort_values('N')
    what = df[what]
    if relative:
        what /= df['C6(BG)']
    ax.plot(df['N'], what.values, label=label)


fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
plot_species(axes[0], free_atoms_vv.loc(0)[0.0101, True], 'C6(TS)', 'TS')
plot_species(axes[0], free_atoms_vv.loc(0)[0.0093, False], 'C6', 'VV[0.0093,nonsph]')
plot_species(axes[0], free_atoms_vv.loc(0)[0.0101, False], 'C6', 'VV[0.0101,nonsph]')
plot_species(axes[0], free_atoms_vv.loc(0)[0.0101, True], 'C6', 'VV[0.0101,sph]')
plot_species(axes[0], free_atoms_vv.loc(0)[0.0101, False], 'C6(BG)', 'BG')
plot_species(axes[1], free_atoms_vv.loc(0)[0.0101, True], 'C6(TS)', None, True)
plot_species(axes[1], free_atoms_vv.loc(0)[0.0093, False], 'C6', None, True)
plot_species(axes[1], free_atoms_vv.loc(0)[0.0101, False], 'C6', None, True)
plot_species(axes[1], free_atoms_vv.loc(0)[0.0101, True], 'C6', None, True)
axes[1].set_xlabel('$Z$')
axes[1].set_xticks([2, 10, 18, 31, 36, 46, 54, 81, 86])
axes[1].set_xticklabels('He Ne Ar Ga Kr Pd Xe Tl Rn'.split())
axes[0].set_yscale('log')
axes[0].set_ylabel('$C_6$')
axes[1].set_ylabel(r'$C_6/C_6(\mathrm{BG})$')

fig.legend(loc='center right', bbox_to_anchor=(1, .5))
fig.subplots_adjust(right=0.8)

# ::>
# **Table 7: Relative differences of C6 coefficients to the TS reference.**
# `diff` is the difference between the spherical and nonspherical columns. Shown
# are elements with absolute `diff` larger than 9%.

# ::hide
(
    free_atoms_vv
    .pipe(lambda x: (x['C6']-x['C6(TS)'])/x['C6(TS)'])
    .loc[0.0101]
    .unstack('spherical')
    .assign(diff=lambda x: x[False]-x[True])
    .query('abs(diff) > 0.09')
    .rename_axis('species')
)

# ::>
# We repeat the procedure of optimizing the $C$ coefficients in the VV
# functional, but now with the normalization to free-atom reference via
# nonspherical free atoms. As a result, we optimize the ability of the functional
# to predict relative changes to the atomic polarizabilities in molecules,
# rather than absolute polarizabilities. Although the functional dependence of
# MARE on $C_1$ and $C_2$ is substantially different with the normalization, the
# minima are very close, which means that a single set of $(C_1,C_2)$ optimizes
# both the absolute polarizabilities as well as the relative changes in
# polarizabilities.

# ::hide
C6_my_pol_normed_nonsph_scan = pd.concat({
    (C1, C2): rp.evaluate_func_on_C6(
        partial(my_pol, C1=C1, C2=C2), C6_set_pts, ref_C6,
        free_atoms_pts.loc[False], coords_C6.species, rescale=True
    ) for C1, C2 in tqdm(list(product(
        [0.70, 0.75, .8, .85, .9],
        [-.04, -.02, 0, .02, .04],
    )))
}, names='C1 C2'.split())

# ::>

# ::hide
fig, ax = plt.subplots()
rp.plot_matrix_df(
    ax,
    C6_my_pol_normed_nonsph_scan.groupby('C1 C2'.split())
    .apply(rp.dataset_stats)['MARE'].unstack('C1'),
    levels=[.05, .06, .07, .085, .1, .13, .17]
)

# ::>
# In contrast, this is not the case when the spherical free atoms are used. In
# that case, the optimal $C$ coefficient is pushed beyond 0.015. This is simply
# because with spherical atoms, the VV functional severely overestimates the
# polarizabilities, and the optimization attempts to artificially reduce this.
# In the following, we will consider only the nonspherical free-atom reference,
# unless explicitly stated otherwise.

# ::hide
C6_my_pol_normed_nonsph_scan \
    .groupby('C1 C2'.split()).apply(rp.dataset_stats)[['MARE']] \
    .unstack('C2')['MARE'].pipe(rp.findmin)

# ::>

# ::hide
C6_my_pol_normed_nonsph_scan \
    .groupby('C1 C2'.split()).apply(rp.dataset_stats)['MARE'] \
    .loc[:, 0.].pipe(rp.findmin)

# ::>

# ::hide
C6_my_pol_normed_sph_scan = pd.concat({
    (C1, C2): rp.evaluate_func_on_C6(
        partial(my_pol, C1=C1, C2=C2), C6_set_pts, ref_C6,
        free_atoms_pts.loc[True], coords_C6.species, rescale=True
    ) for C1, C2 in tqdm(list(product(
        [.8, .85, .9, .95, 1.],
        [-.08, -.06, -.04, -.02, 0],
    )))
}, names='C1 C2'.split())

# ::>

# ::hide
fig, ax = plt.subplots()
rp.plot_matrix_df(
    ax,
    C6_my_pol_normed_sph_scan.groupby('C1 C2'.split())
    .apply(rp.dataset_stats)['MARE'].unstack('C1'),
    levels=[.05, .06, .07, .085, .1, .13, .17]
)

# ::>

# ::hide
C6_my_pol_normed_sph_scan \
    .groupby('C1 C2'.split()).apply(rp.dataset_stats)[['MARE']] \
    .unstack('C2')['MARE'].pipe(rp.findmin)

# ::>

# ::hide
C6_my_pol_normed_sph_scan \
    .groupby('C1 C2'.split()).apply(rp.dataset_stats)['MARE'] \
    .loc[:, 0.].pipe(rp.findmin)

# ::>
# We also plot the distribution of relative errors in the C6 coefficients on the
# data set of C6 coefficients. Since this set contains mostly organic compounds,
# and the optimized VV functional is quite accurate for atoms H, C, N, O, the
# normalization improves the errors only slightly.


C6s_from_methods = pd.concat([
    rp.evaluate_ts_scs_on_C6(coords_C6, hirsh_C6, ref_C6),
    pd.concat({
        'VV': rp.evaluate_func_on_C6(my_pol, C6_set_pts, ref_C6),
        'VV(opt)': rp.evaluate_func_on_C6(
            partial(my_pol, C1=0.805), C6_set_pts, ref_C6
        ),
        'VV(normed)': rp.evaluate_func_on_C6(
            my_pol, C6_set_pts, ref_C6,
            free_atoms_pts.loc[False], coords_C6.species, rescale=True
        ),
        'VV(opt,normed)': rp.evaluate_func_on_C6(
            partial(my_pol, C1=0.805), C6_set_pts, ref_C6,
            free_atoms_pts.loc[False], coords_C6.species, rescale=True
        ),
    }, names=['method']),
])

g = sns.FacetGrid(
    C6s_from_methods.reset_index(), col='method', col_wrap=3, height=2.2
)
g.map(plt.scatter, 'ref', 'reldelta')

# ::>
# Next, we check what is the effect of normalization on the binding energies. On
# the S66 set, the normalization has hardly any effect.

# ::hide
energies_s66_vvnorm = rp.specs_to_binding_enes([
    {},
    {'scs': True, 'beta': 0.83},
    {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.76, 'Rvdw17_base': True},
    {'vv': 'lg2', 'vv_norm': 'nonsph', 'C_vv': 0.0101, 'beta': 0.76, 'Rvdw17_base': True},
], s66_ds, aims_data_s66, alpha_vvs_s66, free_atoms_vv, unit=kcal)

# ::>

# ::hide
energies_s66_vvnorm.groupby('method').apply(rp.dataset_scale_stats) \
    .loc(1)[:, ['MRE', 'MARE']].round(3)

# ::>
# As a diversion, here is a curiosity. Through experimentation, one can find a
# particular combination of choices that make the resulting method quite
# accurate, but this is most likely via cancellation of errors. This combination
# is the unoptimized VV functional without any cutoffs and the normalization to
# *spherical* atoms. As discussed above, the absence of any cutoffs makes the
# $C_6$ coefficients of the complex larger, increasing the binding energies. But
# the normalization to spherical atoms artificially decreases the $C_6$
# coefficients substantially, hence decreasing the binding energies. In this
# particular case, the cancellation of these two effects is perfect.

# ::hide
energies_s66_vverr = rp.specs_to_binding_enes([
    {},
    {'vv': True, 'C_vv': 0.0093, 'beta': 0.87, 'Rvdw17_base': True},
    {'vv': True, 'C_vv': 0.0093, 'beta': 0.8, 'Rvdw17_base': True},
    {'vv': True, 'C_vv': 0.0093, 'vv_norm': 'sph', 'beta': 0.8, 'Rvdw17_base': True},
], s66_ds, aims_data_s66, alpha_vvs_s66, free_atoms_vv, unit=kcal)

# ::>

# ::hide
energies_s66_vverr.groupby('method').apply(rp.dataset_scale_stats) \
    .loc(1)[:, ['MRE', 'MARE']].round(3)

# ::>
# The effect of normalization can be further seen on the distribution of
# relative differences of atomic $C_6$ coefficients to the TS Hirshfeld-scaled
# coefficients over the monomers in the S66 set.

# ::hide
vdw_params_vvnorm_s66 = pd.concat(
    dict(rp.evaluate_mbd_specs(
        [
            {'beta': np.nan, '_label': 'TS'},
            {'scs': True, 'beta': 0.83, '_label': 'rsSCS'},
            {'vv': True, 'C_vv': 0.0101, 'beta': np.nan, '_label': 'VV'},
            {
                'vv': True, 'vv_norm': 'nonsph', 'C_vv': 0.0101, 'beta': np.nan,
                '_label': 'VV[nonsph]'
            },
            {
                'vv': True, 'vv_norm': 'sph', 'C_vv': 0.0101, 'beta': np.nan,
                '_label': 'VV[sph]'
            },
        ],
        aims_data_s66,
        alpha_vvs_s66,
        free_atoms_vv,
        get_vdw_params=True,
    ).ene),
    names='label scale fragment method'.split()
)

# ::>

# ::hide
sns.FacetGrid(
    vdw_params_vvnorm_s66
    .set_index('species', append=True)
    .rename_axis('quantity', axis=1)['alpha_0 C6'.split()].stack()
    .loc(0)[:, 1., 'fragment-1 fragment-2'.split()]
    .unstack('method')
    .pipe(lambda x: x.apply(lambda y: (y-x['TS'])/x['TS'])).stack()
    .unstack('quantity')[['C6']]
    .reset_index(),
    hue='species',
    col='method',
    col_wrap=2,
    col_order='VV VV[sph] VV[nonsph] rsSCS'.split(),
).map(plt.hist, 'C6', histtype='step').add_legend()


# ::>
# On the solids data set, the normalization affects just the transition metals,
# as expected. The $C_6$ coefficients of alkali metals are affected too, but
# since they are small, the vdW contribution is negligible in any case. For
# transition metals, the normalization increases the $C_6$ coefficients, and
# hence increases the overbinding. This is more pronounced when using the BG
# reference values, which are mostly larger than TS for the *d* elements.

# ::hide
energies_solids_vvnorm = rp.specs_to_binding_enes(
    [
        {},
        {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.76, 'Rvdw17_base': True},
        {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.76, 'Rvdw17_base': True, 'vv_norm': 'aims'},
        {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.76, 'Rvdw17_base': True, 'vv_norm': 'aims', 'vdw_ref': 'BG'},
    ],
    solids_ds,
    aims_data_solids.loc(0)[:, 1.],
    alpha_vvs_solids,
    unit=ev,
)

# ::>

# ::hide
energies_solids_vvnorm.groupby('group method'.split(), sort=False) \
    .apply(rp.dataset_stats).unstack('group').swaplevel(0, 1, 1) \
    .loc(1)[:, ['N', 'MRE']].sort_index(1).round(3)

# ::>
# However, with the normalization and when using the TS references values, the
# transition-metal $C_6$ coefficients from VV are now very close to the values
# obtained by the vdW-surf approach (Table 8).

# ::hide
vdw_params_solids_vvnorm = pd.concat(
    dict(rp.evaluate_mbd_specs(
        [
            {
                'vv': 'lg2', 'C_vv': 0.0101, 'beta': np.nan,
                'Rvdw17_base': True, 'vv_norm': 'aims', '_label': 'rlg2VV'
            },
            {
                'vv': 'lg2', 'C_vv': 0.0101, 'beta': np.nan, 'vdw_ref': 'BG',
                'Rvdw17_base': True, 'vv_norm': 'aims', '_label': 'rlg2VV[BG]'
            },
        ],
        aims_data_solids.loc(0)[:, 1., 'crystal'],
        alpha_vvs_solids,
        get_vdw_params=True,
    ).ene.dropna()),
    names='label scale fragment method'.split()
).xs(1, level='scale').xs('crystal', level='fragment')

# ::>
# **Table 8: $C_6$ coefficients of main-group and transition metals.**

# ::hide
(
    vdw_params_solids_vvnorm
    .append(
        pd.read_csv(resource_filename('mbdvv', 'data/vdw-surf.csv'))
        .assign(method='surf', i_atom=1).set_index('label method i_atom'.split())
    )
    .sort_index()
    .pipe(
        lambda df: df.assign(
            group=solids_ds.df['group'].xs(1, level='scale')
            .loc[df.index.get_level_values('label')].values
        )
    )
    .xs(1, level='i_atom')
    .set_index('group', append=True).swaplevel(0, 2).loc(0)['MM TM'.split()]['C6']
    .unstack('method').round()
)

# ::>
# ## Van der Waals radii
#
# Up to now, the damping in all the MBD variants tested was always based on the
# Hirshfeld-scaled reference vdW radii. In this section, we will investigate
# other options.

# ::hide
energies_s66_vdw = rp.specs_to_binding_enes([
    {},
    {'scs': True, 'beta': 0.83},
    {'scs': True, 'beta': 0.78, 'no_vdwscs': True},
    {'scs': True, 'beta': 0.82, 'Rvdw_vol_scale': Fraction('1/7')},
    {'scs': True, 'beta': 0.82, 'Rvdw_vol_scale': Fraction('1/7'), 'Rvdw17_base': True},
    {'scs': True, 'beta': 0.82, 'Rvdw17': True},
    {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.79, 'Rvdw17': True},
    {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.77, 'Rvdw17': True, 'vv_norm': 'nonsph'},
    {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.81, 'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, 'vv_norm': 'nonsph'},
    {'scs': True, 'beta': 0.85},
    {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.83, 'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, 'vv_norm': 'nonsph'},
], s66_ds, aims_data_s66, alpha_vvs_s66, free_atoms_vv, unit=kcal)


# ::>
# In MBD@rsSCS, the vdW radii used in the MBD Hamiltonian are not only
# Hirshfeld-scaled, but also further modified by the short-range screening.
# Here, we find that this modification is not necessary if the $\beta$ parameter
# is adjusted accordingly.

# ::hide
energies_s66_vdw.groupby('method').apply(rp.dataset_scale_stats) \
    .loc(1)[:, ['MRE', 'MARE']].round(3).iloc[[1, 2]]

# ::>
# MBD@rsSCS scales the free-atom vdW radii with a cubic root of the ratio of the
# Hirshfeld volumes. Here, we find that this can be modified to the power of
# 1/7. As a result, we can use the formula for vdW radii from static
# polarizabilities not only for free atoms, but also for the atoms in molecules.

# ::hide
energies_s66_vdw.groupby('method').apply(rp.dataset_scale_stats) \
    .loc(1)[:, ['MRE', 'MARE']].round(3).iloc[[3, 4, 5]]

# ::>
# This approach works also for MBD@VV. Note that the method in the first row of
# the table does not use any free-atom reference data.

# ::hide
energies_s66_vdw.groupby('method').apply(rp.dataset_scale_stats) \
    .loc(1)[:, ['MRE', 'MARE']].round(3).iloc[[6, 7, 8]]

# ::>
# For reasons explained below, we may still need to use the 1/3 scaling rather
# than 1/7 scaling. This works equally well for MBD@VV on the S66 set. The
# range of errors is still somewhat larger than with MBD@rsSCS on the compressed
# geometries, but except for that the behavior of this version of MB@VV is
# equivalent to MBD@rsSCS. At compressed geometries, the short-range screening
# is probably able to better capture the increased polarizability than the VV
# functional.

# ::hide
energies_s66_vdw.groupby('method').apply(rp.dataset_scale_stats) \
    .loc(1)[:, ['MRE', 'MARE']].round(3).iloc[[8]]

# ::>

# ::hide
with sns.color_palette(list(reversed(sns.color_palette('coolwarm', 8)))):
    g = sns.catplot(
        data=energies_s66_vdw.reset_index(),
        kind='box',
        x='method',
        y='reldelta',
        hue='scale',
        aspect=2,
        height=2.5,
        margin_titles=True
    )
g.ax.axhline(color='black', linewidth=0.5, zorder=-1)
g.set(ylim=(-.5, .5))
g.set_xticklabels(rotation=30, ha='right')
g.set_xlabels('')
g.set_ylabels(r'$\Delta E_i/E_i^\mathrm{ref}$')
g.set(yticks=[-.3, -.1, 0, .1, .3])
g.set_yticklabels([r'$-30\%$', r'$-10\%$', '0%', '10%', '30%'])
g._legend.set_title('equilibrium\ndistance scale')

# ::>
# Let's compare with results obtained with the VV10 functional.

mbdscan_data = pd.HDFStore('data/mbd-scan-data.h5')

# ::>

# ::hide
energies_s66_vv10 = (
    mbdscan_data['/scf'].loc(0)['S66x8'].loc(0)[:, :, 'pbe']
    .merge(
        mbdscan_data['/vv10']
        .loc(0)['S66x8']
        .loc(0)[:, :, :, ['base', 'vdw']]['ene']
        .unstack()
        .pipe(lambda x: x['vdw'] - x['base'])
        .unstack()
        .apply(lambda x: interp1d(x.index, x), axis=1)
        .to_frame('vdw'),
        on='system dist'.split()
    )
    .assign(vdw=lambda df: df.apply(lambda x: float(x['vdw'](6.8)), axis=1))
    .assign(method='PBE+VV10')
    .assign(ene=lambda x: x['ene'] + x['vdw'])
    .assign(
        delta=lambda x: x['ene'] - x['ref'],
        reldelta=lambda x: (x['ene'] - x['ref'])/abs(x['ref']),
    )
)

# ::>

# ::hide
with sns.color_palette(list(reversed(sns.color_palette('coolwarm', 8)))):
    g = sns.catplot(
        data=energies_s66_vv10.reset_index(),
        kind='box',
        x='method',
        y='reldelta',
        hue='dist',
        aspect=2,
        height=2.5,
        margin_titles=True
    )
g.ax.axhline(color='black', linewidth=0.5, zorder=-1)
g.set(ylim=(-.5, .5))
g.set_xticklabels(rotation=30, ha='right')
g.set_xlabels('')
g.set_ylabels(r'$\Delta E_i/E_i^\mathrm{ref}$')
g.set(yticks=[-.3, -.1, 0, .1, .3])
g.set_yticklabels([r'$-30\%$', r'$-10\%$', '0%', '10%', '30%'])
g._legend.set_title('equilibrium\ndistance scale')

# ::>

energies_s66_vdw_bare = rp.evaluate_mbd_specs([
    {'scs': True, 'beta': 0.83},
    {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.81, 'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, 'vv_norm': 'nonsph'},
], aims_data_s66, alpha_vvs_s66, free_atoms_vv)


# ::>

(
    pd.merge(
        mbdscan_data['/scf'].loc(0)['S66x8', :, :, ['pbe', 'pbe0'], False].reset_index()
        .assign(system=lambda x: x['system'].str.replace('  ', ' ')),
        energies_s66_vdw.reset_index(),
        left_on=('system', 'dist'),
        right_on=('label', 'scale'),
        suffixes=('_dft', '_vdw'),
    )
    .drop('system dist cp natoms name group delta reldelta'.split(), axis=1)
    .set_index('label scale xc method'.split())
    .groupby('label scale xc'.split()).apply(lambda x: x.assign(ene_vdw=x['ene_vdw']-x['ene_vdw'].xs('PBE', level='method').iloc[0]))
    .assign(ene=lambda x: x['ene_dft']+x['ene_vdw'])
    .assign(delta=lambda x: x['ref']-x['ene'])
    .assign(reldelta=lambda x: x['delta']/x['ref'])
    .groupby('xc method'.split())
    .apply(rp.dataset_scale_stats)
    .loc(1)[:, 'N MRE MARE'.split()].round(3)
)

# ::>

# ::hide
energies_solids_vdw = rp.specs_to_binding_enes(
    [
        {},
        {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.79, 'Rvdw17': True},  # optimal beta
        {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.77, 'Rvdw17': True, 'vv_norm': 'aims'},  # optimal beta
        {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.77, 'Rvdw17': True, 'vv_norm': 'aims', 'vdw_ref': 'BG'},
        {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.83, 'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, 'vv_norm': 'aims'},
    ],
    solids_ds,
    aims_data_solids.loc(0)[:, 1.],
    alpha_vvs_solids,
    unit=ev,
)

# ::>
# In the solids, the use of any vdW radii based on VV polarizabilities leads to smaller vdW
# radii and hence stronger binding, resulting in improved performance for the ionic solids,
# and decreased performance for the metals and carbides and nitrides. This may
# seem like something we might want to avoid, but it may be necessary if we want
# to achieve the same performance as the surf-family of methods for hybrid
# interfaces (see below).

# ::hide
energies_solids_vdw.groupby('group method'.split(), sort=False) \
    .apply(rp.dataset_stats).unstack('group').swaplevel(0, 1, 1) \
    .loc(1)[:, ['N', 'MRE']].sort_index(1).round(3)


# ::>

g = sns.catplot(
    data=energies_solids_vdw.reset_index(),
    kind='box',
    x='method',
    y='reldelta',
    hue='group',
    aspect=2,
    height=2.5,
    margin_titles=True
)
g.ax.axhline(color='black', linewidth=0.5, zorder=-1)
g.set(ylim=(-.35, .35))
g.set_xticklabels(rotation=30, ha='right')

# ::>
# Finally, we show the vdW radii of atoms in metals obtained by various
# approaches. The only one that gives small enough radii comparable to vdW-surf is
# using the cubic-root scaling with rlg2VV polarizabilities and the TS reference
# radii. Either of the following makes the vdW radii substantially larger: using
# the 1/7-scaling or using free-atom vdW radii from the 1/7 formula.

# ::hide
vdw_params_solids_vdw = pd.concat(
    dict(rp.evaluate_mbd_specs(
        [
            {
                'vv': 'lg2', 'C_vv': 0.0101, 'beta': np.nan, 'vv_norm': 'aims',
                'Rvdw_scale_vv': 'cutoff', '_label': 'rlg2VV[1/3]'
            },
            {
                'vv': 'lg2', 'C_vv': 0.0101, 'beta': np.nan, 'vv_norm': 'aims',
                'Rvdw17': True, '_label': 'rlg2VV[1/7]'
            },
            {
                'vv': 'lg2', 'C_vv': 0.0101, 'beta': np.nan, 'vv_norm': 'aims',
                'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, '_label': 'rlg2VV[1/3,17base]'
            },
            {'beta': np.nan, '_label': 'TS'},
            {'beta': np.nan, 'Rvdw17_base': True, '_label': 'TS[17base]'},
        ],
        aims_data_solids.loc(0)[:, 1., 'crystal'],
        alpha_vvs_solids,
        get_vdw_params=True,
    ).ene.dropna()),
    names='label scale fragment method'.split()
).xs(1, level='scale').xs('crystal', level='fragment')

# ::>

# ::hide
(
    vdw_params_solids_vdw
    .append(
        pd.read_csv(resource_filename('mbdvv', 'data/vdw-surf.csv'))
        .assign(method='surf', i_atom=1).set_index('label method i_atom'.split())
    )
    .sort_index()
    .pipe(
        lambda df: df.assign(
            group=solids_ds.df['group'].xs(1, level='scale')
            .loc[df.index.get_level_values('label')].values
        )
    )
    .xs(1, level='i_atom')
    .set_index('group', append=True).swaplevel(0, 2).loc(0)['MM TM'.split()]['R_vdw']
    .unstack('method').round(2)
)

# ::>
# ## Further tests
#
# Here, we check performance on other systems besides S66 and the data set of
# solids investigated above.
#
# On X23, the performance is comparable to MBD@rsSCS. Rather then overbinding,
# the MBD@VV variants somewhat underbind, but the overall performance is the
# same.

# ::>

# ::hide
energies_x23_all = rp.specs_to_binding_enes(
    [
        {},
        {'scs': True, 'beta': 0.83, 'kdensity': 0.8},
        {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.77, 'Rvdw17': True, 'vv_norm': 'nonsph', 'kdensity': 0.8},
        {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.81, 'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, 'vv_norm': 'nonsph', 'kdensity': 0.8},
        {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.83, 'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, 'vv_norm': 'nonsph', 'kdensity': 0.8},
    ], x23_ds, aims_data_x23, alpha_vvs_x23, free_atoms_vv, unit=kcal)

# ::>

x23_translator = {x.geomname: x.Index[0] for x in get_x23().df.itertuples() if x.geomname != 'CO2'}
x23_translator

# ::>

# ::hide
energies_x23_all.groupby('method', sort=False) \
    .apply(rp.dataset_stats)[['MRE', 'MARE']].round(3)


# ::>

(
    pd.merge(
        mbdscan_data['/scf'].loc(0)['X23', :, :, ['pbe', 'pbe0'], False].reset_index()
        .replace({'system': x23_translator}),
        energies_x23_all.reset_index(),
        left_on=('system', 'dist'),
        right_on=('label', 'scale'),
        suffixes=('_dft', '_vdw'),
    )
    .drop('system dist cp natoms name group delta reldelta'.split(), axis=1)
    .set_index('label scale xc method'.split())
    .groupby('label scale xc'.split()).apply(lambda x: x.assign(ene_vdw=x['ene_vdw']-x['ene_vdw'].xs('PBE', level='method').iloc[0]))
    .assign(ene=lambda x: x['ene_dft']+x['ene_vdw'])
    .assign(delta=lambda x: x['ref']-x['ene'])
    .assign(reldelta=lambda x: x['delta']/x['ref'])
    .groupby('xc method'.split())
    .apply(rp.dataset_stats)
    .loc(1)['N MRE MARE'.split()].round(3)
)

# ::>
# On S12L, the performance is somewhat worse than with MBD@rsSCS, going from 5%
# to 10%. In particular, the pi-pi complexes are underbound by 10% to 20%. We can understand
# the origin of this underbinding by comparing between nmVV, lgVV, and lg2VV.
# For pi-pi complexes, the difference between these three is up to 30% for the
# buckyball catcher, whereas it is within 5% for the other complexes.

# ::hide
aims_data_s12l, s12l_ds, alpha_vvs_s12l = rp.setup_s12l()

# ::>

# ::hide
energies_s12l_all = rp.specs_to_binding_enes(
    [
        {},
        {'scs': True, 'beta': 0.83},
        {'vv': 'nm', 'C_vv': 0.0101, 'beta': 0.87, 'Rvdw17_base': True},
        {'vv': 'lg', 'C_vv': 0.0101, 'beta': 0.84, 'Rvdw17_base': True},
        {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.76, 'Rvdw17_base': True},
        {'vv': 'lg2', 'C_vv': 0.0101, 'beta': 0.77, 'Rvdw17': True, 'vv_norm': 'nonsph'},
        {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.81, 'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, 'vv_norm': 'nonsph'},
        {'vv': 'lg2', 'C_vv': 0.0093, 'beta': 0.83, 'Rvdw_scale_vv': 'cutoff', 'Rvdw17_base': True, 'vv_norm': 'nonsph'},
    ], s12l_ds, aims_data_s12l, alpha_vvs_s12l, free_atoms_vv, unit=kcal, refname='energy')

# ::>

# ::hide
energies_s12l_all.groupby('method', sort=False) \
    .apply(rp.dataset_stats)[['MRE', 'MARE']].round(3)

# ::>

# ::hide
energies_s12l_all['reldelta'].unstack('label').round(2)

# ::>

(
    pd.merge(
        mbdscan_data['/scf'].loc(0)['S12L', :, :, ['pbe', 'pbe0'], False].reset_index(),
        energies_s12l_all.reset_index(),
        left_on=('system', 'dist'),
        right_on=('label', 'scale'),
        suffixes=('_dft', '_vdw'),
    )
    .drop('system dist cp natoms name group delta reldelta'.split(), axis=1)
    .set_index('label scale xc method'.split())
    .groupby('label scale xc'.split()).apply(lambda x: x.assign(ene_vdw=x['ene_vdw']-x['ene_vdw'].xs('PBE', level='method').iloc[0]))
    .assign(ene=lambda x: x['ene_dft']+x['ene_vdw'])
    .assign(delta=lambda x: x['ref']-x['ene'])
    .assign(reldelta=lambda x: x['delta']/x['ref'])
    # ['reldelta'].unstack('label').round(2)
    .groupby('xc method'.split())
    .apply(rp.dataset_stats)
    .loc(1)['N MRE MARE'.split()].round(3)
)

# ::>
# ### Hybrid interface
#
# Here we consider an adsorption of a benzene molecule on a silver surface. The
# surface is modeled with six layers of silver atoms and the periodic images of
# the benzene molecules are well separated.

# ::hide
results_surface = rp.setup_surface()

# ::>
# Here we plot the atomic $C_6$ coefficients in a cross section of the layer. The
# surface metallic atoms have significantly larger $C_6$ coefficients than the
# bulk atoms.


# ::hide
def plot_surface_C6(ax):
    payload = [('vv_pols', 'VV'), ('vv_pols_lg2', 'lg2VV')]
    for vv_label, label in payload:
        ax.scatter(
            results_surface[3.3][0]['coords']['value'][2, :],
            3/np.pi*np.sum(
                results_surface[3.3][0][vv_label]**2
                * results_surface[3.3][0]['omega_grid_w'][:, None], 0
            ),
            label=label,
        )
    ax.legend()


fig, ax = plt.subplots()
plot_surface_C6(ax)

# ::>


# ::hide
def tmp():
    aims_data = pd.DataFrame(results_surface).T.rename_axis('scale') \
        .rename(columns={0: 'data', 1: 'gridfile'}) \
        .reset_index().assign(label='surface', fragment='all') \
        .set_index('label scale fragment'.split())
    return rp.evaluate_mbd_specs([
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
    ], aims_data)


energies_surface_all = tmp()


# ::>
# And finally we calculate the binding energy curves. With the use of the
# cubic root scaling of vdW radii, we achieve roughly the same vdW parameters as
# in the vdW-surf approach, and also the resulting binding energies are very
# similar to MBD@rsSCS[surf]. This is not the case when other approaches to the
# vdW radii are used, which leads to weaker binding.

# ::hide
fig, ax = plt.subplots(figsize=(4, 3))
sns.lineplot(
    data=energies_surface_all.pipe(rp.ene_dft_vdw)['ene'].unstack('scale')
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
ax.set_xlabel(r'$\mathrm{Distance}/\mathrm{\AA}$')
ax.set_ylabel(r'$\mathrm{Energy}/\mathrm{eV}$')

# ::>

results_layered = rp.setup_layered()

# ::>


def _get_layered_dataset():
    idxs = []
    data_vars = defaultdict(list)
    for label, shift, xc, data in results_layered:
        if not data:
            continue
        idxs.append((label, shift))
        data_vars['coords'].append(data['coords']['value'])
        data_vars['lattice_vector'].append(data['lattice_vector']['value'])
        data_vars['elems'].append(data['elems']['atom'])
        data_vars['energy'].append(data['energy'][0]['value'][0])
        data_vars['energy_vdw'].append(data['energy'][1]['value'][0])
        data_vars['volumes'].append(data['volumes'])
        data_vars['vv_pols'].append(data['vv_pols'])
        data_vars['vv_pols_nm'].append(data['vv_pols_nm'])
        data_vars['C6_vv_nm'].append(data['C6_vv_nm'])
        data_vars['alpha_0_vv_nm'].append(data['alpha_0_vv_nm'])
        data_vars['volumes_free'].append(data['free_atoms']['volumes'])
        data_vars['C6_vv_free'].append(data['free_atoms']['C6_vv'])
        data_vars['vv_pols_free'].append(data['free_atoms']['vv_pols'])
        data_vars['alpha_0_vv_free'].append(data['free_atoms']['alpha_0_vv'])
        data_vars['species'].append(data['free_atoms']['species'])
    idx = pd.MultiIndex.from_tuples(idxs, names=['label', 'shift'])
    return pd.DataFrame(data_vars, index=idx)


df = _get_layered_dataset()

# ::>

(
    df[['energy', 'energy_vdw']].to_xarray()
    .pipe(lambda ds: xr.concat(
        [ds['energy']-ds['energy_vdw'], ds['energy']],
        pd.Index(['PBE', 'PBE+MBD'], name='method')
    ))
    .pipe(lambda x: x-x.sel(shift=40))
    .to_dataframe('energy')
    .reset_index()
    .pipe(
        lambda x: sns.relplot(
            data=x,
            kind='line',
            x='shift',
            y='energy',
            col='method',
            hue='label',
            height=3,
        )
        .set(xlim=(-.4, .7), ylim=(None, 0.002))
    )
)

# ::>

layered_enes = (
    pd.concat([
        df['energy'].to_xarray()
        .pipe(lambda x: x-x.sel(shift=40))
        .sel(shift=slice(None, 4))
        .to_dataframe()
        .reset_index()
        .groupby('label')
        .apply(
            lambda x: pd.Series(minimize(
                interp1d(x['shift'].values, x['energy'].values, kind='cubic'),
                [0],
                bounds=[(-.4, .7)],
            ))
        )[['fun', 'x']]
        .applymap(lambda x: x[0])
        .rename(columns={'fun': 'energy_uc', 'x': 'c_shift'}),
        df['lattice_vector']
        .xs(0, level='shift')
        .pipe(lambda x: x*0.5291)
        .apply(lambda x: np.linalg.det(x)/x[2, 2])
        .to_frame('area'),
        pd.read_csv(
            resource_stream('mbdvv', 'data/layered.csv'),
            index_col='label scale'.split()
        )
        .xs(1, level='scale')
        [['c', 'energy']]
        .rename(columns={'c': 'c_ref', 'energy': 'energy_ref'}),
    ], axis=1)
    .assign(n_layer=lambda x: np.where((x['c_ref'] > 10) | (x['area'] < 6), 2, 1))
    .assign(energy=lambda x: -x['energy_uc']*ev*1e3/x['area']/x['n_layer'])
)
layered_enes

# ::>

layered_enes.loc[['BN', 'graphite'], 'energy_uc']*ev*1e3/4

# ::>

(
    layered_enes
    .assign(err=lambda x: (x['energy']-x['energy_ref'])/x['energy_ref'])
    .assign(aerr=lambda x: abs(x['err']))
    [['err', 'aerr']].describe()
)

# ::>

df.keys()

# ::>

(
    df
    .assign(vv_shave=lambda x: (x['vv_pols_nm']-x['vv_pols'])/x['vv_pols'])
    .groupby('label shift'.split())
    .apply(lambda x: pd.DataFrame({
        'vv_shave': x['vv_shave'].iloc[0][0],
        'vv_pols': x['vv_pols'].iloc[0][0],
        'vv_pols_nm': x['vv_pols_nm'].iloc[0][0],
        'elem': x['elems'].iloc[0]
    }))
    .reset_index()
    .groupby('elem shift'.split()).apply(lambda x: x['vv_shave'].describe())
    .loc(0)[:, [-.4, 0, 40]]
)
