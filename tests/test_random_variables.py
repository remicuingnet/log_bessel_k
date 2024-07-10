#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

# STL
import unittest

# External
from scipy import stats
import numpy as np

# Project
import random_variables


class RandomVariable(unittest.TestCase):
    NB_TESTS = 10
    SCALE = 10
    NB_LINEAR_COMBINATIONS = 10
    DELTA = 1e-6

    def test_normal(self):
        for _ in range(self.NB_TESTS):
            loc = np.random.randn() * self.SCALE
            scale = np.random.rand() * self.SCALE
            rv = random_variables.norm(loc=loc, scale=scale)

            mu, var, skewness, excess_kurtosis = stats.norm.stats(
                loc=loc,
                scale=scale,
                moments="mvsk")

            self.assertAlmostEqual(rv.mean, mu, delta=1e-6)
            self.assertAlmostEqual(rv.var, var, delta=1e-6)
            self.assertAlmostEqual(rv.std, np.sqrt(var), delta=1e-6)
            self.assertAlmostEqual(rv.skewness, skewness, delta=1e-6)
            self.assertAlmostEqual(rv.excess_kurtosis, excess_kurtosis, delta=1e-6)
            self.assertAlmostEqual(rv.kurtosis, 3+excess_kurtosis, delta=1e-6)

    def test_normal_lc(self):
        for _ in range(self.NB_TESTS):
            lc_rv = None
            mu = 0
            var = 0
            for _ in range(self.NB_LINEAR_COMBINATIONS):
                loc = np.random.randn() * self.SCALE
                scale = np.random.rand() * self.SCALE
                scale2 = np.random.rand() * self.SCALE
                rv = random_variables.norm(loc=loc, scale=scale)
                if lc_rv is None:
                    lc_rv = rv*scale2
                else:
                    lc_rv = lc_rv + rv*scale2
                mu += scale2*loc
                var += scale2*scale2*scale*scale

            mu, var, skewness, excess_kurtosis = stats.norm.stats(
                loc=mu,
                scale=np.sqrt(var),
                moments="mvsk")

            self.assertAlmostEqual(lc_rv.mean, mu, delta=1e-6)
            self.assertAlmostEqual(lc_rv.var, var, delta=1e-6)
            self.assertAlmostEqual(lc_rv.std, np.sqrt(var), delta=1e-6)
            self.assertAlmostEqual(lc_rv.skewness, skewness, delta=1e-6)
            self.assertAlmostEqual(lc_rv.excess_kurtosis, excess_kurtosis, delta=1e-6)
            self.assertAlmostEqual(lc_rv.kurtosis, 3+excess_kurtosis, delta=1e-6)

    def test_normal_ratio(self):
        for _ in range(self.NB_TESTS):
            lc_rv = [None, None]

            for _ in range(self.NB_LINEAR_COMBINATIONS):
                for i in range(2):
                    loc = np.random.rand() * self.SCALE  # positive mean
                    loc += np.random.rand() * self.SCALE  # positive mean
                    scale = np.random.rand() * self.SCALE
                    scale2 = np.random.rand() * self.SCALE
                    rv = random_variables.norm(loc=loc, scale=scale)
                    if lc_rv[i] is None:
                        lc_rv[i] = rv * scale2
                    else:
                        lc_rv[i] = lc_rv[i] + rv * scale2

            rv_numerator = lc_rv[0]
            rv_denominator = lc_rv[1]

            for _ in range(self.NB_TESTS):
                alpha = np.random.rand()
                q1 = random_variables.r_quantile_for_normal(alpha, rv_numerator, rv_denominator)
                q2, _ = random_variables.r_quantile_from_cf(alpha, rv_numerator, rv_denominator, newton=True)
                q3, _ = random_variables.r_quantile_from_cf(alpha, rv_numerator, rv_denominator, newton=False)

                if not (np.isnan(q1) and q2 < 0):
                    self.assertAlmostEqual(q1, q2, delta=self.DELTA)
                self.assertAlmostEqual(q2, q3, delta=self.DELTA)

    def test_normal_lc_pdf_cdf(self):
        for _ in range(self.NB_TESTS):
            lc_rv = None
            mu = 0
            var = 0
            for _ in range(self.NB_LINEAR_COMBINATIONS):
                loc = np.random.randn() * self.SCALE
                scale = np.random.rand() * self.SCALE
                scale2 = np.random.rand() * self.SCALE
                rv = random_variables.norm(loc=loc, scale=scale)
                if lc_rv is None:
                    lc_rv = rv*scale2
                else:
                    lc_rv = lc_rv + rv*scale2
                mu += scale2*loc
                var += scale2*scale2*scale*scale

            for _ in range(self.NB_TESTS):
                stats.norm.stats()
                x = stats.norm.rvs(loc=mu, scale=np.sqrt(var))
                pdf1 = stats.norm.pdf(x, loc=mu, scale=np.sqrt(var))
                pdf2 = random_variables.pdf_from_cf(x, lc_rv)

                cdf1 = stats.norm.cdf(x, loc=mu, scale=np.sqrt(var))
                cdf2 = random_variables.cdf_from_cf(x, lc_rv)

                ppf1 = stats.norm.ppf(cdf1, loc=mu, scale=np.sqrt(var))
                ppf2 = random_variables.quantile_from_cf(cdf1, lc_rv)

                # print(p1, p2)
                self.assertAlmostEqual(pdf1, pdf2, delta=self.DELTA)
                self.assertAlmostEqual(cdf1, cdf2, delta=self.DELTA)
                self.assertAlmostEqual(ppf1, ppf2, delta=self.DELTA)

    def test_student(self):
        for _ in range(self.NB_TESTS):
            loc = np.random.randn() * self.SCALE
            scale = np.random.rand() * self.SCALE
            df = np.random.randint(5, 20)
            rv = random_variables.student_t(df=df, loc=loc, scale=scale)

            mu, var, skewness, excess_kurtosis = stats.t.stats(
                df=df,
                loc=loc,
                scale=scale,
                moments="mvsk")

            self.assertAlmostEqual(rv.mean, mu, delta=self.DELTA)
            self.assertAlmostEqual(rv.var, var, delta=self.DELTA)
            self.assertAlmostEqual(rv.std, np.sqrt(var), delta=self.DELTA)
            self.assertAlmostEqual(rv.skewness, skewness, delta=self.DELTA)
            self.assertAlmostEqual(rv.excess_kurtosis, excess_kurtosis, delta=self.DELTA)
            self.assertAlmostEqual(rv.kurtosis, 3+excess_kurtosis, delta=self.DELTA)

    def test_student_pdf_cdf(self):
        for _ in range(self.NB_TESTS):

            df = np.random.randint(5, 20)

            loc = np.random.randn() * self.SCALE
            scale = np.random.rand() * self.SCALE
            rv = random_variables.student_t(df=df, loc=loc, scale=scale,
                                            functor=random_variables.student_functor_rec(np.float64))

            for _ in range(self.NB_TESTS):
                stats.norm.stats()
                x = stats.t.rvs(df=df, loc=loc, scale=scale)
                pdf1 = stats.t.pdf(x, df=df, loc=loc, scale=scale)
                pdf2 = random_variables.pdf_from_cf(x, rv)

                cdf1 = stats.t.cdf(x, df=df, loc=loc, scale=scale)
                cdf2 = random_variables.cdf_from_cf(x, rv)

                ppf1 = stats.t.ppf(cdf1, df=df, loc=loc, scale=scale)
                ppf2 = random_variables.quantile_from_cf(cdf1, rv)

                # print(p1, p2)
                self.assertAlmostEqual(pdf1, pdf2, delta=1e-6)
                self.assertAlmostEqual(cdf1, cdf2, delta=1e-6)
                self.assertAlmostEqual(ppf1, ppf2, delta=1e-6)

    def test_student_cl_ratio(self):
        for _ in range(self.NB_TESTS):
            lc_rv = [None, None]

            for _ in range(self.NB_LINEAR_COMBINATIONS):
                for i in range(2):
                    df = np.random.randint(5, 20)
                    loc = np.random.rand() * self.SCALE  # positive mean
                    loc += np.random.rand() * self.SCALE  # positive mean
                    scale = np.random.rand() * self.SCALE
                    scale2 = np.random.rand() * self.SCALE
                    rv = random_variables.student_t(df=df, loc=loc, scale=scale)
                    if lc_rv[i] is None:
                        lc_rv[i] = rv * scale2
                    else:
                        lc_rv[i] = lc_rv[i] + rv * scale2

            rv_numerator = lc_rv[0]
            rv_denominator = lc_rv[1]

            for _ in range(self.NB_TESTS):
                alpha = np.random.rand()
                q2, _ = random_variables.r_quantile_from_cf(alpha, rv_numerator, rv_denominator, newton=False)
                q1, _ = random_variables.r_quantile_from_cf(alpha, rv_numerator, rv_denominator, newton=True)
                self.assertAlmostEqual(q1, q2, delta=self.DELTA)


def test():
    nb_samples = 10000
    rv1 = random_variables.student_t(df=6, loc=1, scale=2)
    rv2 = random_variables.norm(loc=10, scale=1)
    rv3 = rv1 + rv2

    for rv in [rv1, rv2, rv3]:

        print("mean", rv.mean)
        print("std", rv.std)
        print("skewness", rv.skewness)
        print("kurtosis", rv.kurtosis)
        print("excess_kurtosis", rv.excess_kurtosis)
        print()
        print("empirical mean", np.mean(rv.rvs(size=nb_samples)))


if __name__ == "__main__":
    # test()
    unittest.main()
