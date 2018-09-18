import numpy as np
from scipy.special import erf


def reduced_grad(n, grad):
    return grad/(2*(3*np.pi**2)**(1/3)*n*(4/3))


def alpha_kin(n, grad, kin):
    return (
        (kin - grad**2/(8*n)) /
        (3/10*(3*np.pi**2)**(2/3)*n**(5/3))
    )


def beta_kin(n, grad, kin):
    return (
        (kin - grad**2/(8*n)) /
        (kin + 3/10*(3*np.pi**2)**(2/3)*n**(5/3))
    )


def terf(x, *, k, x0):
    return np.where(x > 0, 0.5*(erf(k*(x+x0))+erf(k*(x-x0))), 0)


# rgrad_cutoff = partial(terf, k=60, x0=0.07)


def vv_pol(n, grad, C=0.0093, u=0.):
    return n/(4*np.pi/3*n+C*(grad/n)**4+u**2)


def nm_cutoff(rgrad, alpha):
    return 1 - (
        (1-terf(rgrad, k=60, x0=0.12)) *  # high gradient
        terf(alpha-10*rgrad, k=6, x0=0.7) *  # low alpha, tilted
        (1-terf(alpha, k=1.5, x0=5))  # high alpha
    )


def lg_cutoff(rgrad, alpha):
    return 1 - (
        (1-terf(rgrad, k=60, x0=0.12)) *  # high gradient
        terf(alpha-10*rgrad, k=6, x0=0.7)  # low alpha, tilted
    )


def lg_cutoff2(rgrad, alpha):
    return 1 - (
        (1-terf(rgrad, k=40, x0=0.18)) *  # high gradient
        terf(alpha-10*rgrad, k=6, x0=0.7)  # low alpha, tilted
    )
