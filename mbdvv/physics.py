import numpy as np


def reduced_grad(x):
    return x.rho_grad_norm/(2*(3*np.pi**2)**(1/3)*x.rho*(4/3))


def alpha_kin(x):
    return (
        (x.kin_dens-x.rho_grad_norm**2/(8*x.rho)) /
        (3/10*(3*np.pi**2)**(2/3)*x.rho**(5/3))
    )
