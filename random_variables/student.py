#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

# STL

# External
import numpy as np
import scipy.special
import scipy.stats
from scipy import special
# import scipy.special.cython_special

# Project
from random_variables import RandomVariable

import log_bessel_k


def student_t(df, loc=0., scale=1., functor=None) -> 'RandomVariable':
    """
        only for df >=5
    """
    functor = functor or _StudentFunctor
    student = functor(df)

    if df <= 0:
        raise ValueError("Degree of freedom df should be positive")

    if df <= 2:
        kurtosis = np.nan
    elif df <= 4:
        kurtosis = np.inf
    else:
        kurtosis = 3. + 6. / (df - 4.)

    if df <= 1:
        variance = np.nan
    elif df <= 2:
        variance = np.inf
    else:
        variance = df / (df - 2.)

    rv = RandomVariable(
        cf=student.cf,
        cf_prime=student.cf_prime,
        rvs=student.rvs,
        mean=0.,
        variance=variance,
        skewness=0. if df > 3 else np.nan,
        kurtosis=kurtosis,
        symmetric=True
    )
    rv *= scale
    rv += loc
    return rv


def _modified_bessel_second_kind(n):
    if (n % 1.) == 0:
        return lambda x: scipy.special.kn(int(n), x)
    else:
        return lambda x: scipy.special.kv(n, x)
        # return lambda x: scipy.special.cython_special.kv(n, x)


def _modified_bessel_second_kind_prime(n):
    if (n % 1.) == 0:
        # return lambda x: -(n/x)*scipy.special.kn(n, x)-scipy.special.kn(n+1, x)
        return lambda x: -.5*(scipy.special.kn(n-1, x)+scipy.special.kn(n+1, x))
    else:
        # return lambda x: (n/x)*scipy.special.kv(n, x)-scipy.special.kv(n+1, x)
        return lambda x: -.5*(scipy.special.kv(n-1, x)+scipy.special.kv(n+1, x))


class _StudentFunctor:
    TAYLOR_EXP_THRESHOLD = 6

    def __init__(self, df, display=False):
        self._df = df
        self._display = display

    def cf(self, t):
        """
            at z = 0:
            \forall \nu >0, K_{\nu}(z) \sim \frac{\Gamma(\nu)}{2} \left({\dfrac {2}{z}}\right)^{\nu}

        :param df:
        :param t:
        :return:
        """
        if not np.isscalar(t):
            return np.array([self._cf(x) for x in t])

        else:
            return self._cf(t)

    def rvs(self, *args, **kwargs):
        return scipy.stats.t.rvs(df=self._df, *args, **kwargs)

    def _cf(self, t):
        # if t <= 10 ** (-self.TAYLOR_EXP_THRESHOLD / self._df / 2.):
        #     # logging.debug("Use Taylor second order expansion")
        #     # y = 1 - t * t / (4 * (self._df / 2. - 1))
        #     nu = self._df
        #     y = 1 - .5*t * t * (nu/(nu-2))
        #     return y
        x = np.sqrt(self._df)*np.abs(t)

        return 2*self._psi(self._df / 2., x)

    def cf_prime(self, t):
        """
            for nu > 2
        """
        x = np.sqrt(self._df) * np.abs(t)
        nu = self._df
        return (-nu*t/(nu/2.-1.)) * self._psi(nu/2. - 1, x)

    def _psi(self, nu, z):
        # return ((z/2)**nu)*scipy.special.kv(nu, z)/scipy.special.gamma(nu)
        k = np.round(nu)
        nu0 = nu-k
        if nu0 == -.5:
            k -= 1
            nu0 = .5

        if k < 0:
            raise ValueError("nu should be greater or equal to -1/2")
        zk0, zk1 = self._zk0(nu0, z)

        psi1 = zk1/scipy.special.gamma(nu0+1)
        if k == 0:
            psi = nu0*((z/2)**(nu0))*scipy.special.kv(nu0, z)/scipy.special.gamma(nu0+1)
        elif k == 1:
            psi = psi1
        else:
            zk2 = (nu0+1)*zk1 + ((z/2)**(nu0+1))*zk0
            psi2 = zk2/scipy.special.gamma(nu0+2)
            psi = psi2

        nu_k = 2 + nu0
        while nu > nu_k:
            psi = psi2 + (z*z)/(4*nu_k*(nu_k-1))*psi1
            psi1 = psi2
            psi2 = psi
            nu_k += 1

        if self._display:
            print(f"nu: {nu}, z: {z}, psi: {psi}, zk1 {zk1}, zk2: {zk2}")
        return psi

    def _zk0(self, nu, z):
        """
        """
        if not -1/2 <= nu <= 1/2:
            raise ValueError("nu should be comprised in -0.5 0.5")

        if (nu == -1 / 2.) or (nu == 1 / 2):
            zk0 = .5*np.sqrt(np.pi / 2) * np.exp(-z) * np.sqrt(z)
            zk1 = .25*np.sqrt(np.pi) * np.exp(-z) * (z + 1)
            # else:
            #     # if np.abs(nu) < .01:  # FIXME MAGIC NUMBER
            #     #     zk0, zk1=0, 0
            #     # else:
            #
        elif z < 1:
            zk0, zk1 = self._zk0_small_z_large_nu(nu, z)
        else:
            zk0, zk1 = self._zk0_large_z(nu, z)
        return zk0, zk1

    @staticmethod
    def _zk0_small_z_large_nu(nu, z, nb_factors=5):
        k = 0
        if nu == 0:
            f = (z/2)*scipy.special.digamma(1) - scipy.special.xlogy(z/2, z/2)
            p = .25*z
            q = .25*z
        else:
            gamma_p = scipy.special.gamma(1 + nu)
            gamma_m = scipy.special.gamma(1 - nu)
            if np.abs(nu < .1):
                gamma1 = (1/gamma_m - 1/gamma_p)/(2*nu)
                gamma2 = (1 / gamma_m + 1 / gamma_p) / 2

                mu = scipy.special.xlogy(nu, 2/z)

                # f = 1./np.sinc(nu)*(gamma1*np.cosh(mu)*z/2 - gamma2 * np.sinh(mu)/mu*scipy.special.xlogy(z/2, z/2))

                f = 1. / np.sinc(nu) * (
                    gamma1 * np.cosh(mu) * z / 2 - gamma2 * .5 * (
                        scipy.special.exprel(mu)+scipy.special.exprel(-mu)) * scipy.special.xlogy(z / 2, z / 2))
            else:
                f = 1/(2*nu*np.sinc(nu))
                f *= ((z/2)**(1-nu))/gamma_m - ((z/2)**(1+nu))/gamma_p

            p = .5 * ((z/2)**(1-nu))*gamma_p
            q = .5 * ((z/2)**(1+nu))*gamma_m

        c = 1
        zk0 = c * f
        zk1 = scipy.special.gamma(1+nu)/2.
        c2 = (z/2)**(1+nu)
        z = z * z
        for _ in range(nb_factors):
            k += 1
            f = (k * f + p + q) / (k * k-nu*nu)
            p = p / (k-nu)
            q = q / (k+nu)
            c = c * z / 4 / k
            zk0 += c * f
            zk1 += c2*(p-k*f)
            c2 = c2 * z / 4 / (k+1)

        return zk0, zk1

    @staticmethod
    def _zk0_large_z(nu, z):
        zk0 = (z/2)*scipy.special.kv(nu, z)
        zk1 = (z/2)**(nu+1) * scipy.special.kv(nu+1, z)
        return zk0, zk1


class StudentFunctorRec(_StudentFunctor):
    def _psi_log(self, nu, z):
        # return ((z/2)**nu)*scipy.special.kv(nu, z)/scipy.special.gamma(nu)
        # log_k = np.log(scipy.special.kv(nu, z))
        log_k = log_bessel_k.log_bessel_forward_rec(nu, z)
        res = nu*np.log(z/2) + log_k - special.loggamma(nu)
        return res

    def _psi(self, nu, z):
        return np.exp(self._psi_log(nu, z))


class StudentFunctorDirect(_StudentFunctor):
    def _psi(self, nu, z):
        # return ((z/2)**nu)*scipy.special.kv(nu, z)/scipy.special.gamma(nu)
        res = nu*np.log(z/2) + np.log(scipy.special.kv(nu, z)) - special.loggamma(nu)
        return np.exp(res)


def student_functor_rec(dtype=np.float32):
    class _StudentFunctorRec(_StudentFunctor):
        def _psi(self, nu, z):
            # return ((z/2)**nu)*scipy.special.kv(nu, z)/scipy.special.gamma(nu)
            # log_k = np.log(scipy.special.kv(nu, z))
            res = np.zeros(1, dtype=dtype)
            z = np.asarray(z, dtype=dtype)
            nu = np.asarray(nu, dtype=dtype)
            log_k = log_bessel_k.log_bessel_forward_rec(nu, z, dtype)
            res += nu * np.log(z / 2)
            res += log_k
            res -= np.asarray(special.loggamma(nu), dtype=dtype)
            return np.exp(res)
    return _StudentFunctorRec


def student_functor_rec_from_ratio(dtype=np.float32):
    class _StudentFunctorRec(_StudentFunctor):
        def _psi(self, nu, z):
            # return ((z/2)**nu)*scipy.special.kv(nu, z)/scipy.special.gamma(nu)
            # log_k = np.log(scipy.special.kv(nu, z))
            res = np.zeros(1, dtype=dtype)
            z = np.asarray(z, dtype=dtype)
            nu = np.asarray(nu, dtype=dtype)
            log_k = log_bessel_k.log_bessel_from_ratio(nu, z, dtype)
            res += nu * np.log(z / 2)
            res += log_k
            res -= np.asarray(special.loggamma(nu), dtype=dtype)
            return np.exp(res)
    return _StudentFunctorRec


def student_functor_direct_log(dtype=np.float32):
    class _StudentFunctorDirect(_StudentFunctor):
        def _psi(self, nu, z):
            # return ((z/2)**nu)*scipy.special.kv(nu, z)/scipy.special.gamma(nu)
            res = np.zeros(1, dtype=dtype)
            z = np.asarray(z, dtype=dtype)
            nu = np.asarray(nu, dtype=dtype)
            res += nu * np.log(z / 2)
            res += np.log(np.asarray(scipy.special.kve(nu, z), dtype=dtype)) - z
            res -= np.asarray(special.loggamma(nu), dtype=dtype)
            return np.exp(res)
    return _StudentFunctorDirect


def student_functor_direct(dtype=np.float32):
    class _StudentFunctorDirect(_StudentFunctor):
        def _psi(self, nu, z):
            # return ((z/2)**nu)*scipy.special.kv(nu, z)/scipy.special.gamma(nu)
            res = np.zeros(1, dtype=dtype)
            z = np.asarray(z, dtype=dtype)
            nu = np.asarray(nu, dtype=dtype)
            # res += (z / 2)**nu
            # res /= np.asarray(special.gamma(nu), dtype=dtype)
            res += np.exp(nu*np.log(z / 2)-np.asarray(special.loggamma(nu), dtype=dtype))
            res *= np.asarray(scipy.special.kv(nu, z), dtype=dtype)
            return res
    return _StudentFunctorDirect


