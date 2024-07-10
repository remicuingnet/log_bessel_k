#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

# External
from enum import IntEnum

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.fft

from . import RandomVariable


class TAYLOR_CORRECTION(IntEnum):
    NONE = -1
    ORDER_0 = 0
    FIRST_ORDER = 1
    SECOND_ORDER = 2


def cdf_from_cf(x: float,
                rv: 'RandomVariable',
                taylor_correction: TAYLOR_CORRECTION = TAYLOR_CORRECTION.SECOND_ORDER,
                **kwargs):
    """
    """

    if rv.symmetric:
        def fun_for_cdf(t: float) -> float:
            if t == 0:
                u = -x  # * rv.cf(0)
                # raise ValueError()
            else:
                cf = rv.cf(t)

                if (np.isnan(np.abs(cf)) or np.isinf(np.abs(cf))) and taylor_correction > TAYLOR_CORRECTION.NONE:
                    cf = 1
                    # if taylor_correction >= TAYLOR_CORRECTION.FIRST_ORDER:
                    #     cf += 1j * rv.mean * t
                    if taylor_correction >= TAYLOR_CORRECTION.SECOND_ORDER:
                        cf += - .5 * t * t * (rv.var + rv.mean)

                u = -np.sinc(x*t/np.pi)*cf*x
            return np.real(u)
    else:
        def fun_for_cdf(t: float) -> float:
            if t == 0:
                u = -np.imag(x * rv.cf(0))
                # raise ValueError()
            else:
                u = np.imag(np.exp(-1j * x * t) * rv.cf(t) / t)
            return u

    y2, *_ = scipy.integrate.quad(fun_for_cdf, 0, np.inf, **kwargs)

    cdf = .5 - y2 / np.pi
    return cdf


def pdf_from_cf(x: float,
                rv: 'RandomVariable',
                taylor_correction: TAYLOR_CORRECTION = TAYLOR_CORRECTION.SECOND_ORDER,
                dtype=np.float64,
                **kwargs):
    """
     symmetric: symmetric random variable around zero
    """
    if rv.symmetric:
        def fun_for_pdf(t):
            cf = rv.cf(t)

            if (np.isnan(np.abs(cf)) or np.isinf(np.abs(cf))) and taylor_correction > TAYLOR_CORRECTION.NONE:
                cf = 1
                if taylor_correction >= TAYLOR_CORRECTION.FIRST_ORDER:
                    cf += 1j * rv.mean * t
                if taylor_correction >= TAYLOR_CORRECTION.SECOND_ORDER:
                    cf += - .5 * t * t * (rv.var + rv.mean)

            return dtype(np.real(np.cos(x * t) * cf))
    else:
        def fun_for_pdf(t):
            cf = rv.cf(t)

            if np.isnan(np.abs(cf)) and taylor_correction > TAYLOR_CORRECTION.NONE:
                cf = 1
                if taylor_correction >= TAYLOR_CORRECTION.FIRST_ORDER:
                    cf += 1j * rv.mean * t
                if taylor_correction >= TAYLOR_CORRECTION.SECOND_ORDER:
                    cf += - .5 * t * t * (rv.var + rv.mean)

            return dtype(np.real(np.exp(-1j*x*t) * cf))

    y = scipy.integrate.quad(fun_for_pdf, 0, np.inf, **kwargs)[0]
    y = np.real(y)/np.pi
    pdf = y
    return pdf


def pdf_from_cf_trapezoidal(x: float,
                            rv: 'RandomVariable',
                            taylor_correction: TAYLOR_CORRECTION = TAYLOR_CORRECTION.SECOND_ORDER,
                            dtype=np.float64,
                            t_max=100,
                            n_points=2000,
                            **kwargs):
    """
     symmetric: symmetric random variable around zero
    """
    if rv.symmetric:
        def fun_for_pdf(t):
            if t == 0:
                return 1
            cf = rv.cf(t)

            if (np.isnan(np.abs(cf)) or np.isinf(np.abs(cf))) and taylor_correction > TAYLOR_CORRECTION.NONE:
                cf = 1
                if taylor_correction >= TAYLOR_CORRECTION.FIRST_ORDER:
                    cf += 1j * rv.mean * t
                if taylor_correction >= TAYLOR_CORRECTION.SECOND_ORDER:
                    cf += - .5 * t * t * (rv.var + rv.mean)

            return np.real(np.cos(x * t) * cf)
    else:
        def fun_for_pdf(t):
            cf = rv.cf(t)

            if np.isnan(np.abs(cf)) and taylor_correction > TAYLOR_CORRECTION.NONE:
                cf = 1
                if taylor_correction >= TAYLOR_CORRECTION.FIRST_ORDER:
                    cf += 1j * rv.mean * t
                if taylor_correction >= TAYLOR_CORRECTION.SECOND_ORDER:
                    cf += - .5 * t * t * (rv.var + rv.mean)

            return np.real(np.exp(-1j*x*t) * cf)

    t_vec = np.linspace(0, t_max, n_points)

    integrand = [fun_for_pdf(t) for t in t_vec]
    integrand[0] = .5
    integrand[-1] *= .5
    integrand = np.asarray(integrand, dtype=dtype)
    integrand = integrand[::-1]
    y = np.sum(integrand)
    y *= t_max
    y /= n_points
    # y, *_ = scipy.integrate.quad(fun_for_pdf, 0, np.inf, **kwargs)

    y /= np.pi
    pdf = y
    return pdf


def pdf_from_cf_fft(x: float,
                    rv: 'RandomVariable',
                    taylor_correction: TAYLOR_CORRECTION = TAYLOR_CORRECTION.SECOND_ORDER,
                    dtype=np.float64,
                    n=50000,
                    t_max=2000,
                    **kwargs):
    """
     symmetric: symmetric random variable around zero
    """

    d = np.arange(n)
    dt = t_max/n
    x_min = -n * np.pi / t_max
    x_max = -x_min
    x_vec = x_min + (x_max - x_min) * d / n

    u = np.zeros((n,))
    u[0] = .5  # trapezoidal (ignore last term .5*x[N])
    for i in range(1, n):

        t = i*dt

        cf = rv.cf(np.abs(t))[0]
        if (np.isnan(np.abs(cf)) or np.isinf(np.abs(cf))) and taylor_correction > TAYLOR_CORRECTION.NONE:
            cf = 1
            if taylor_correction >= TAYLOR_CORRECTION.FIRST_ORDER:
                cf += 1j * rv.mean * t
            if taylor_correction >= TAYLOR_CORRECTION.SECOND_ORDER:
                cf += - .5 * t * t * (rv.var + rv.mean)

        u[i] = cf*np.exp(-1j*x_min*t)  # TODO check offset

    u[-1] *= .5  # #trapezoidal (ignore last term .5*x[N])
    pdf = scipy.fft.fft(u)
    pdf = np.real(pdf)*dt/np.pi
    idx = np.argmin(np.abs(x_vec-x))
    return pdf[idx], x_vec[idx]


def pdf_from_cf_fft1(x: float,
                     rv: 'RandomVariable',
                     taylor_correction: TAYLOR_CORRECTION = TAYLOR_CORRECTION.SECOND_ORDER,
                     **kwargs):
    """
     symmetric: symmetric random variable around zero
    """
    m = 30000
    a = 10000
    beta = a/m

    ind = np.arange(m)

    tj = (ind-m/2)*beta
    xk = 2*np.pi*(ind-m/2)/a
    u = np.zeros((m,))
    for i, t in zip(ind, tj):
        if t == 0:
            cf = 1
        else:
            cf = rv.cf(np.abs(t))[0]
        if (np.isnan(np.abs(cf)) or np.isinf(np.abs(cf))) and taylor_correction > TAYLOR_CORRECTION.NONE:
            cf = 1
            if taylor_correction >= TAYLOR_CORRECTION.FIRST_ORDER:
                cf += 1j * rv.mean * t
            if taylor_correction >= TAYLOR_CORRECTION.SECOND_ORDER:
                cf += - .5 * t * t * (rv.var + rv.mean)
        if i % 2 == 0:
            u[i] = cf
        else:
            u[i] = -cf
    pdf = scipy.fft.fft(u)
    pdf = ((-1)**ind)*beta*pdf
    pdf = np.real(pdf)/2 / np.pi

    idx = np.argmin(np.abs(xk-x))
    return pdf[idx]


def pdf_from_cf_fft2(x: float,
                     rv: 'RandomVariable',
                     taylor_correction: TAYLOR_CORRECTION = TAYLOR_CORRECTION.SECOND_ORDER,
                     **kwargs):

    mt = 20000

    t_vec = np.linspace(-1000, 1000, mt)
    dt = (t_vec[-1]-t_vec[0])/mt

    def cf_fun(t):
        if t == 0:
            cf = 1
        else:
            cf = rv.cf(np.abs(t))[0]
        return cf

    cf = np.array([cf_fun(t) for t in t_vec])
    cf[0] *= .5
    cf[-1] *= .5
    pdf = np.real(np.sum(np.exp(-1j*t_vec*x)*cf/2/np.pi))*dt
    return pdf


def pdf_from_cf_fft_frt(x: float,
                        rv: 'RandomVariable',
                        taylor_correction: TAYLOR_CORRECTION = TAYLOR_CORRECTION.SECOND_ORDER,
                        **kwargs):
    """
     symmetric: symmetric random variable around zero
    """
    m = 3000
    a = 100
    g = 20

    beta = a/m
    gamma = g/m
    delta = beta*gamma/2/np.pi

    ind = np.arange(m)

    tj = (ind-m/2)*beta
    xk = (ind-m/2)*gamma
    u = np.zeros((m,), dtype=np.complex)
    for i, t in zip(ind, tj):
        if t == 0:
            cf = 1
        else:
            cf = rv.cf(np.abs(t))[0]
        if (np.isnan(np.abs(cf)) or np.isinf(np.abs(cf))) and taylor_correction > TAYLOR_CORRECTION.NONE:
            cf = 1
            if taylor_correction >= TAYLOR_CORRECTION.FIRST_ORDER:
                cf += 1j * rv.mean * t
            if taylor_correction >= TAYLOR_CORRECTION.SECOND_ORDER:
                cf += - .5 * t * t * (rv.var + rv.mean)
        u[i] = cf *np.exp(np.pi*i*1j*m*delta)

    # G
    y = np.zeros((2*m,))
    z = np.zeros((2*m,))
    y[:m] = u*np.exp(-np.pi*1j*ind*ind*delta)
    z[:m] = np.exp(np.pi*1j*ind*ind*delta)
    z[m:] = np.exp(np.pi*1j*((ind-2*m)**2)*delta)

    # G
    pdf = scipy.fft.ifft(scipy.fft.fft(y)*scipy.fft.fft(z))
    pdf = np.exp(-np.pi*1j*ind*ind*delta)*pdf[:m]
    pdf = beta*pdf*np.exp(np.pi*1j*(ind-m/2)*m*delta)
    pdf = np.real(pdf)/2 / np.pi
    #
    return pdf


def quantile_from_cf(alpha, rv: 'RandomVariable', q0=None, newton=True, **kwargs):
    # TODO FIND A WAY TO INITIALIZE q0
    q0 = q0 or rv.mean

    def fun(x):
        return cdf_from_cf(x, rv) - alpha

    def fprime(x):
        return pdf_from_cf(x, rv)

    if newton:
        q = scipy.optimize.newton(fun, q0, fprime=fprime, **kwargs)
    else:
        q = scipy.optimize.newton(fun, q0, fprime=None, **kwargs)
    return q


def r_quantile_from_cf(alpha,
                       rv_numerator: 'RandomVariable',
                       rv_denominator: 'RandomVariable',
                       r0=None, newton=False, **kwargs):
    """
        approximate the quantile Q_W(alpha) of W:= rv1/rv2 with r such that P(rv1 - r * rv2 <= 0) = alpha
        return: quantile approximation and lower and upper bound on the cdf F_W(r).
    """
    r0 = r0 or (rv_numerator.mean / rv_denominator.mean)

    def fun(r: float):
        rv = rv_numerator + rv_denominator * (-r)
        return cdf_from_cf(0, rv) - alpha

    def fprime(r: float):
        # to check
        def _fun(t):
            return np.imag(rv_numerator.cf(t) * rv_denominator.cf_prime(-r*t))/np.pi

        y2, *_ = scipy.integrate.quad(_fun, 0, np.inf, **kwargs)
        return y2

    if newton:
        q = scipy.optimize.newton(fun, r0, fprime=fprime, **kwargs)
    else:
        q = scipy.optimize.newton(fun, r0, fprime=None, **kwargs)

    epsilon_2 = cdf_from_cf(0, rv_denominator)
    alpha_up = alpha + epsilon_2
    if q > 0:
        epsilon_1 = cdf_from_cf(0, rv_numerator)
        alpha_low = alpha + epsilon_2 * (1 - 2*epsilon_1)
    else:
        alpha_low = alpha - epsilon_2

    return q, (alpha_low, alpha_up)


def r_quantile_lower_bound_from_cf(alpha,
                                   rv_numerator: 'RandomVariable',
                                   rv_denominator: 'RandomVariable',
                                   r0=None, newton=False, **kwargs):

    epsilon_2 = cdf_from_cf(0, rv_denominator)
    return r_quantile_from_cf(alpha-epsilon_2,
                              rv_numerator,
                              rv_denominator, r0, newton, **kwargs)


def r_quantile_upper_bound_from_cf(alpha,
                                   rv_numerator: 'RandomVariable',
                                   rv_denominator: 'RandomVariable',
                                   r0=None, newton=False, **kwargs):

    epsilon_1 = cdf_from_cf(0, rv_numerator)
    epsilon_2 = cdf_from_cf(0, rv_denominator)
    q, bounds = r_quantile_from_cf(alpha - epsilon_2 + 2*epsilon_1*epsilon_2,
                                   rv_numerator,
                                   rv_denominator, r0, newton, **kwargs)
    if q < 0:
        q, bounds = r_quantile_from_cf(alpha + epsilon_2,
                                       rv_numerator,
                                       rv_denominator, r0, newton, **kwargs)
    return q, bounds
