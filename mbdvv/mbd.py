import numpy as np
from pymbd import get_kgrid
from scipy.special import erf


class MBDException(Exception):
    pass


class NegativeEigs(MBDException):
    pass


class NegativeAlpha(MBDException):
    pass


def scaled_eigs(x):
    return np.where(x >= 0, x, -erf(np.sqrt(np.pi)/2*x**4)**(1/4))


def mbd_rsscs(mbd_calc, coords, alpha_0, C6, R_vdw, beta, lattice=None,
              k_grid=None, rpa=False, scale_eigs=True):
    def _array(obj, *args, **kwargs):
        if obj is not None:
            return np.array(obj, *args, **kwargs)

    coords = _array(coords, dtype=float, order='F')
    alpha_0 = _array(alpha_0, dtype=float)
    C6 = _array(C6, dtype=float)
    R_vdw = _array(R_vdw, dtype=float)
    freq, freq_w = mbd_calc.omega_grid
    omega = 4./3*C6/alpha_0**2
    alpha_dyn = alpha_0/(1+(freq[:, None]/omega)**2)
    alpha_dyn_rsscs = np.empty_like(alpha_dyn)
    for a, a_scr in zip(alpha_dyn, alpha_dyn_rsscs):
        sigma = (np.sqrt(2./np.pi)*a/3)**(1./3)
        a_nlc = np.linalg.inv(
            np.diag(np.repeat(1./a, 3)) + mbd_calc.dipole_matrix(
                coords, 'fermi,dip,gg', sigma=sigma, R_vdw=R_vdw,
                beta=beta, lattice=lattice,
            )
        )
        a_scr[:] = sum(a_nlc[i::3, i::3].sum(1) for i in range(3))/3
    alpha_0_rsscs = alpha_dyn_rsscs[0, :]
    if np.any(alpha_0_rsscs <= 0):
        raise NegativeAlpha(alpha_0_rsscs)
    C6_rsscs = 3./np.pi*np.sum(freq_w[:, None]*alpha_dyn_rsscs**2, 0)
    R_vdw_rsscs = R_vdw*(alpha_0_rsscs/alpha_0)**(1./3)
    omega_rsscs = 4./3*C6_rsscs/alpha_0_rsscs**2
    pre = np.repeat(omega_rsscs*np.sqrt(alpha_0_rsscs), 3)
    if lattice is None:
        k_grid = [None]
    else:
        assert k_grid is not None
        k_grid = get_kgrid(lattice, k_grid)
    ene = 0
    for k_point in k_grid:
        T = mbd_calc.dipole_matrix(
            coords, 'fermi,dip', R_vdw=R_vdw_rsscs, beta=beta,
            lattice=lattice, k_point=k_point
        )
        if rpa:
            for u, uw in zip(freq[1:], freq_w[1:]):
                A = np.diag(np.repeat(alpha_0_rsscs/(1+(u/omega_rsscs)**2), 3))
                eigs = np.linalg.eigvals(A@T)
                eigs = np.real(eigs)
                if scale_eigs:
                    eigs = scaled_eigs(eigs)
                if np.any(eigs <= -1):
                    raise NegativeEigs(k_point, u, eigs)
                if not scale_eigs:
                    log_eigs = np.log(1+eigs)
                else:
                    log_eigs = np.log(1+eigs)-eigs
                ene += 1/(2*np.pi)*np.sum(log_eigs)*uw
        else:
            eigs = np.linalg.eigvalsh(
                np.diag(np.repeat(omega_rsscs**2, 3))+np.outer(pre, pre)*T
            )
            if np.any(eigs < 0):
                raise NegativeEigs(k_point, eigs)
            ene += np.sum(np.sqrt(eigs))/2-3*np.sum(omega_rsscs)/2
    ene /= len(k_grid)
    return ene
