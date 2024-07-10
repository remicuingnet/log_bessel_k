#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

import numpy as np
from scipy import special


def log_bessel_forward_rec(nu, z, dtype=np.float64):
    nu = np.asarray(nu, dtype=dtype).copy()
    z = np.asarray(z, dtype=dtype).copy()
    k = np.round(nu)
    nu0 = nu - k
    if nu0 == -0.5:  # FIXME USE PAPER EQ to get the proper rounding
        k -= 1
        nu0 = 0.5

    k0 = np.log(special.kve(nu0, z))-z
    k1 = np.log(special.kve(nu0 + 1, z))-z
    if k > 0:
        nuk = nu0 + 1
        while nuk < nu:
            if k0 < 0:
                k0 = np.log(special.kve(nuk, z))-z
                k1 = np.log(special.kve(nuk + 1, z))-z
            else:
                buff = k1
                k1 = k0 + np.log(1 + 2*nuk/z*np.exp(k1-k0))
                k0 = buff
            nuk += 1
        res = k1
    else:
        res = k0
    return res


def log_bessel_from_ratio(nu, z, dtype=np.float64):
    res = 0
    dtype = dtype
    k = np.round(nu)
    nu0 = nu - k
    if nu0 == -0.5:

        k -= 1
        nu0 = 0.5

    k0 = dtype(special.kve(nu0, z))
    k1 = dtype(special.kve(nu0 + 1, z))
    v1 = k1 / k0
    v1 *= 1
    res0 = np.log(k0)-z
    if k > 0:

        res = np.log(v1)

        nuk = nu0 + 1
        while nuk < nu:
            x = dtype(2 * nuk / z)
            v1 = dtype(1 / v1 + x)
            nuk += dtype(1)
            res += np.log(v1)
    return res + res0


# if __name__ == "__main__":
#     nu0 = 100
#     z = 0.1
#     print(np.log(special.kv(nu0, z)))
#     print(log_bessel_forward_rec(nu0, z))
#     print(log_bessel_from_ratio(nu0, z))

