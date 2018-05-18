import numpy as np
from scipy.special import erf
import pandas as pd
from functools import partial


def reduced_grad(x):
    return x.rho_grad_norm/(2*(3*np.pi**2)**(1/3)*x.rho*(4/3))


def alpha_kin(x):
    return (
        (x.kin_dens-x.rho_grad_norm**2/(8*x.rho)) /
        (3/10*(3*np.pi**2)**(2/3)*x.rho**(5/3))
    )


def terf(x, *, k, x0):
    return 0.5*(erf(k*(x+x0))+erf(k*(x-x0)))


rgrad_cutoff = partial(terf, k=60, x0=0.07)


def vv_pol(n, grad, C=0.0093, u=0.):
    return n/(4*np.pi/3*n+C*(grad/n)**4+u**2)


def calc_vvpol(x, freq, rgrad_cutoff):
    idx = x.index
    n = x.rho.values
    grad = x.rho_grad_norm.values
    w = x.part_weight.values
    cutoff = rgrad_cutoff(reduced_grad(x).values)
    try:
        freq = freq[:, None]
    except TypeError:
        pass
    x = pd.concat(dict(
        vvpol=pd.DataFrame((vv_pol(n, grad, u=freq)*w).T),
        vvpol_nm=pd.DataFrame((vv_pol(n, grad, u=freq)*(w*cutoff)).T),
    ), axis=1)
    x.index = idx
    return x


def bin_alpha_vv(df, bins):
    prefix = df.index.names
    if prefix == [None]:
        prefix = []

    def binidx(x):
        return np.digitize(
            reduced_grad(x).clip(bins[0]+1e-10, bins[-1]-1e-10),
            bins
        )
    subsums = (
        df
        .assign(binidx=binidx)
        .set_index('binidx', append=True)
        .pipe(calc_vvpol, 0, rgrad_cutoff).stack().reset_index(-1, drop=True)
        .groupby(prefix + ['binidx'])
        .apply(lambda x: pd.Series({
            'vv_pol': x.vvpol.sum(),
            'vv_pol_nm': x.vvpol_nm.sum()
        }))
    )
    return subsums
