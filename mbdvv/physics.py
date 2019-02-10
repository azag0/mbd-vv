import numpy as np
from scipy.special import erf, expit


eV = 1/27.211385


def ion_pot(n, grad):
    return (grad/n)**2/8


def alpha_kin(n, grad, kin):
    return (
        (kin - grad**2/(8*n))
        / (3/10*(3*np.pi**2)**(2/3)*n**(5/3))
    )


def beta_kin(n, grad, kin):
    return (
        (kin - grad**2/(8*n))
        / (kin + 3/10*(3*np.pi**2)**(2/3)*n**(5/3))
    )


def terf(x, *, k, x0):
    return np.where(x > 0, 0.5*(erf(k*(x+x0))+erf(k*(x-x0))), 0)


def logistic(x, *, w, x0):
    return expit(4/w*(x-x0))


def scanintp(x, *, c):
    return np.where(x < 0, 1, np.where(x > 1, 0, np.exp(-c*x/(1-x))))


def vv_pol(n, grad, C=0.0093, u=0.):
    return n/(4*np.pi/3*n+C*(grad/n)**4+u**2)


def nm_cutoff(ion, alpha):
    grad_normed = np.sqrt(2*ion)/((3*np.pi**2)**(1/3)*4/3)
    return 1 - (
        (1 - terf(grad_normed, k=60, x0=0.12))  # high gradient
        * terf(alpha-10*grad_normed, k=6, x0=0.7)  # low alpha, tilted
        * (1-terf(alpha, k=1.5, x0=5))  # high alpha
    )


def lg_cutoff(ion, alpha):
    grad_normed = np.sqrt(2*ion)/((3*np.pi**2)**(1/3)*4/3)
    return 1 - (
        (1-terf(grad_normed, k=60, x0=0.12))  # high gradient
        * terf(alpha-10*grad_normed, k=6, x0=0.7)  # low alpha, tilted
    )


def lg_cutoff2(ion, alpha):
    return 1 - (
        (1 - logistic(ion, w=1*eV, x0=5*eV))
        * (1 - scanintp(alpha-3*np.sqrt(ion), c=0.1))
    )
