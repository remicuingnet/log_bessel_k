#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

# STL
from copy import deepcopy
from typing import Callable, Optional, Union

# Externals
import numpy as np


class DistributionMoments(object):
    """
        Class to compute the first fourth moments of linear combination of distribution with known first fourth moments
    """
    def __init__(self,
                 mean: float = 0,
                 variance: float = 1.,
                 skewness: float = 0.,
                 kurtosis: float = 0.):
        """

        :param mean: mean E[X]
        :param variance: variance E[(X-E[X])^2]
        :param skewness: skewness E[(X-E[X])^3]/variance^(3/2)
               https://en.wikipedia.org/wiki/Skewness#Pearson's_moment_coefficient_of_skewness
        :param kurtosis: E[(X-E[X])^3]/variance^(2)
               https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments
        """
        # mean: E[X]
        self.mean = np.float64(mean)
        # variance: E[(X-E[X])^2]
        self.var = np.float64(variance)
        # skewness: E[(X-E[X])^3]/variance^(3/2)
        self.skewness = np.float64(skewness)
        # kurtosis:  E[(X-E[X])^3]/variance^(3/2)
        self.kurtosis = np.float64(kurtosis)

    @property
    def excess_kurtosis(self):
        return self.kurtosis - 3

    @property
    def std(self):
        return np.sqrt(self.var)

    def __iadd__(self, other: 'DistributionMoments') -> 'DistributionMoments':
        """
            moments of the sum of two distributions
        :param other: instance of DistributionMoments
        :return: self
        """
        if not isinstance(other, DistributionMoments):
            raise TypeError("other should be of type {} instead of {}".format(type(self), type(other)))

        skewness = self.skewness*(self.var**(3./2.))
        skewness += other.skewness*(other.var**(3./2.))

        kurtosis = self.kurtosis*self.var*self.var
        kurtosis += other.kurtosis*other.var*other.var
        kurtosis += 6*self.var*other.var
        kurtosis /= (self.var+other.var)**2

        self.kurtosis = kurtosis
        self.skewness = skewness / (self.var + other.var) ** (3. / 2.)
        self.mean += other.mean
        self.var += other.var

        return self

    def __imul__(self, other: float) -> 'DistributionMoments':
        """
            Moments of a distribution multiplied by a scalar
        :param other: scalar
        :return: self
        """
        if not (np.isscalar(other) or isinstance(other, np.ndarray)):
            raise TypeError("other should be of a scalar instead of {}".format(type(other)))
        self.mean *= other
        self.var *= other*other
        self.skewness *= np.sign(other)
        if np.count_nonzero(other == 0) or np.count_nonzero(np.isnan(other)):
            print("zeros: {}, nan{}".format(np.count_nonzero(other == 0), np.count_nonzero(np.isnan(other))))
        return self

    def __add__(self, other: 'DistributionMoments') -> 'DistributionMoments':
        """
            moments of the sum of two distributions
        :param other: instance of DistributionMoments
        :return: new DistributionMoments instance
        """
        x = deepcopy(self)
        x.__iadd__(other)
        return x

    def __mul__(self, other: float) -> 'DistributionMoments':
        """
            Moments of a distribution multiplied by a scalar
        :param other: scalar
        :return: new DistributionMoments
        """
        x = deepcopy(self)
        x.__imul__(other)
        return x


class RandomVariable:
    def __init__(self,
                 cf: Optional[Callable[[float], float]] = None,
                 cf_prime: Optional[Callable[[float], float]] = None,
                 rvs: Optional[Callable[[], float]] = None,
                 mean: float = 0,
                 variance: float = 1.,
                 skewness: float = 0.,
                 kurtosis: float = 0.,
                 symmetric: bool = False,
                 **kwargs):
        """
            :param symmetric: True iff the random variable is symmetric around the mean
        """
        self._offset = _Constant(0.)
        self._cfs = [self._offset.cf, cf]  # characteristic functions
        self._cf_primes = [self._offset.cf_prime, cf_prime]  # derivative of characteristic functions
        self._rvs = [self._offset.rvs, rvs]  # random variable generator
        self._scales = [1., 1.]
        self._moments = DistributionMoments(mean=0,  # mean is coded in the offset (to avoid double definition)
                                            variance=variance,
                                            skewness=skewness,
                                            kurtosis=kurtosis)

        self._symmetric = symmetric  # if symmetric around the mean

    @property
    def moments(self) -> 'DistributionMoments':
        """ return Distribution Moments"""
        moments = deepcopy(self._moments)
        moments.mean += self._offset.value
        return moments

    @property
    def mean(self) -> float:
        return self._offset.value

    @property
    def var(self) -> float:
        return self._moments.var

    @property
    def std(self) -> float:
        return self._moments.std

    @property
    def skewness(self) -> float:
        return self._moments.skewness

    @property
    def kurtosis(self) -> float:
        return self._moments.kurtosis

    @property
    def excess_kurtosis(self) -> float:
        return self._moments.excess_kurtosis

    @property
    def symmetric(self):
        return self._symmetric and (self._offset.value == 0)

    def copy(self):
        return deepcopy(self)

    def cf(self, t):
        return self._cf(t, skip=-1)

    def _cf(self, t, skip=-1):
        """ characteristic function """
        y = 1
        for index, (cf, scale) in enumerate(zip(self._cfs, self._scales)):
            if index == skip:
                continue
            if cf is None:
                y = None
                break

            x = t * scale
            y = y * cf(x)
        return y

    def cf_prime(self, t):
        y = 0
        for index, (cf_prime, scale) in enumerate(zip(self._cf_primes, self._scales)):
            if cf_prime is None:
                y = None
                break
            y_part = scale*cf_prime(scale*t)
            y_part = y_part*self._cf(t, skip=index)
            y = y + y_part
        return y

    def rvs(self, *args, **kwargs):
        y = 0
        for rvs, scale in zip(self._rvs, self._scales):
            if rvs is None:
                y = None
                break

            t = rvs(*args, **kwargs)
            y = y + t * scale
        return y

    def __imul__(self, other: float) -> 'RandomVariable':
        """
            Moments of a distribution multiplied by a scalar
        :param other: scalar
        :return: self
        """
        if not (np.isscalar(other) or isinstance(other, np.ndarray)):
            raise TypeError("other should be of a scalar instead of {}".format(type(other)))

        # update scales for characteristic function computation
        for i, scale in enumerate(self._scales):
            self._scales[i] = scale*other

        # update offset
        self._scales[0] = 1.
        self._offset *= other

        # update moments
        self._moments *= other

        return self

    def __iadd__(self, other: Union[float, 'RandomVariable']) -> 'RandomVariable':

        if isinstance(other, RandomVariable):
            other = deepcopy(other)
            self._moments += other._moments
            self._scales += other._scales[1:]
            self._cfs += other._cfs[1:]
            self._cf_primes += other._cf_primes[1:]
            self._rvs += other._rvs[1:]
            self._offset += other._offset
            self._symmetric = self._symmetric and other._symmetric
        else:
            self._offset += other
        return self

    def __add__(self, other: 'RandomVariable') -> 'RandomVariable':
        x = deepcopy(self)
        x.__iadd__(other)
        return x

    def __mul__(self, other: float) -> 'RandomVariable':

        x = deepcopy(self)
        x.__imul__(other)
        return x


class _Constant:
    def __init__(self, value: float = 0.):
        self.value = value

    def __iadd__(self, other: Union[float, '_Constant']) -> '_Constant':
        if isinstance(other, _Constant):
            self.value += other.value
        else:
            self.value += other
        return self

    def __imul__(self, other: Union[float, '_Constant']) -> '_Constant':
        if isinstance(other, _Constant):
            self.value *= other.value
        else:
            self.value *= other
        return self

    def __add__(self, other: Union[float, '_Constant']) -> '_Constant':
        x = deepcopy(self)
        x.__iadd__(other)
        return x

    def __mul__(self, other: Union[float, '_Constant']) -> '_Constant':
        x = deepcopy(self)
        x.__imul__(other)
        return x

    def cf(self, t):
        if self.value is not None:
            y = np.exp(1j * self.value * t)
        else:
            y = np.ones_like(t)
        return y

    def cf_prime(self, t):
        if self.value is not None:
            y = 1j * self.value * np.exp(1j * self.value * t)
        else:
            y = np.zeros_like(t)
        return y

    def rvs(self, *args, **kwargs):
        return self.value
