import numpy as np
from scipy.special import erf


def reduced_grad(x):
    return x.rho_grad_norm/(2*(3*np.pi**2)**(1/3)*x.rho*(4/3))


def alpha_kin(x):
    return (
        (x.kin_dens-x.rho_grad_norm**2/(8*x.rho)) /
        (3/10*(3*np.pi**2)**(2/3)*x.rho**(5/3))
    )


def terf(x, *, k, x0):
    return 0.5*(erf(k*(x+x0))+erf(k*(x-x0)))


def vv_pol(n, grad, C=0.0093, u=0.):
    return n/(4*np.pi/3*n+C*(grad/n)**4+u**2)
