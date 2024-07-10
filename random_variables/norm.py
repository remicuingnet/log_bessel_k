#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

import numpy as np
from scipy import stats

from . import RandomVariable, DistributionMoments


def _characteristic_function(t):
    """ return characteristic function for centered standard normal distribution"""
    return np.exp(-.5*t * t)


def _characteristic_function_prime(t):
    """ return first derivative of the characteristic function for centered standard normal distribution"""
    return -t*np.exp(-.5*t * t)


def _random_variable_sample(*args, **kwargs):
    """ return random samples drawn according to centered standard normal distribution"""
    return stats.norm.rvs(*args, **kwargs)


def norm(loc=0., scale=1.) -> 'RandomVariable':
    """ Return the random variable loc + scale * X where X ~N(0,1) is a normally distributed random variable """
    rv = RandomVariable(
        cf=_characteristic_function,
        cf_prime=_characteristic_function_prime,
        rvs=_random_variable_sample,
        mean=0,
        variance=1,
        skewness=0,
        kurtosis=3,
        symmetric=True
    )
    rv *= scale
    rv += loc
    return rv


def r_quantile_for_normal(alpha,
                          rv_numerator: 'RandomVariable',
                          rv_denominator: 'RandomVariable'):
    """
    Ratio of two independent normal random variables
    """
    # FIXME missing bound on the cumulative function  + option to center the interval etc.
    x = stats.norm.ppf(alpha)
    return _PsiFunctor(rv_numerator.moments, rv_denominator.moments)(x)


class _PsiFunctor:
    def __init__(self,
                 distribution_numerator: 'DistributionMoments',
                 distribution_denominator: 'DistributionMoments'):
        self.distribution_numerator = distribution_numerator
        self.distribution_denominator = distribution_denominator

    def __call__(self, x, **kwargs):
        """
            psi function defined in
            Ratio of two independent normal random variables
        """
        mu1 = np.array(self.distribution_numerator.mean).reshape(-1)
        std1 = np.sqrt(np.array(self.distribution_numerator.var).reshape(-1))

        mu2 = np.array(self.distribution_denominator.mean).reshape(-1)
        std2 = np.sqrt(np.array(self.distribution_denominator.var).reshape(-1))

        y = x*std2 / mu2

        res = 1 + ((std1 * mu2 / (std2 * mu1)) ** 2) * (1 - y * y)
        ind = np.isnan(res)  # TODO FIND OUT WHEN THIS MAY HAPPEN
        res[ind] = 0
        ind = res < 1
        res[ind] = np.nan

        ind = ~ind
        res_0 = res[ind]
        y = y[ind]
        mu1 = mu1[ind]
        mu2 = mu2[ind]

        res_0 = 1 + y * np.sqrt(res_0)
        res_0 *= mu1 / mu2
        res_0 /= 1-y*y
        res[ind] = res_0
        return res
