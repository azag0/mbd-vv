from numpy import pi, sqrt


def k_F(n):
    return (3*pi**2*n)**(1/3)


beta = 0.066725
mu = beta*(pi**2/3)


def S(n, grad):
    return grad/(2*k_F(n)*n)


def P(n, grad):
    return S(n, grad)**2


def Z(n, grad, kin):
    return tW(n, grad)/kin


def tW(n, grad):
    return 1/8*grad**2/n


kappa = 0.804


def Alpha(p, z):
    return (5*p/3)*(1/z-1)


def F_Xpbe(s):
    return 1+kappa-kappa/(1+mu*s**2/kappa)


def epsilon_Xunif(n):
    return -3*k_F(n)/(4*pi)


def epsilon_Xpbe(n, grad):
    return epsilon_Xunif(n)*F_Xpbe(S(n, grad))


def epsilon_Xtpss(n, grad, kin):
    return epsilon_Xunif(n)*F_Xtpss(P(n, grad), Z(n, grad, kin))


def F_Xtpss(p, z):
    return 1+kappa-kappa/(1+x_tpss(p, z)/kappa)


def x_tpss(p, z):
    return (
        (10/81+c*z**2/(1+z**2)**2)*p +
        146/2025*q_b(p, z)**2 -
        73/405*q_b(p, z)*sqrt(1/2*(3/5*z)**2+1/2*p**2) +
        1/kappa*(10/81)**2*p**2 +
        2*sqrt(e)*10/81*(3/5*z)**2 +
        e*mu*p**3
    ) / (1+sqrt(e)*p)**2


e = 1.537
c = 1.59096


def q_b(p, z):
    return (9/20)*(Alpha(p, z)-1)/sqrt(1+B*Alpha(p, z)*(Alpha(p, z)-1)) + 2*p/3


B = 0.40
