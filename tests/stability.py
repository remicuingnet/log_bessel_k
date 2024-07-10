#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

dtype = np.float32


def rec(x, n, z):
    return 1/x + 2*n/z


def seq_log(nu, z, k0, k1, nb=100):
    n_vec = []
    x_vec = []

    n_vec.append(0+nu)
    n_vec.append(1+nu)
    x_vec.append(k0)
    x_vec.append(k1)
    k0 = dtype(k0)
    k1 = dtype(k1)
    nuk = dtype(nu + 1)
    for _ in range(2, nb):
        buff = dtype(k1)
        k1 = dtype(k0 + np.log(dtype(1) + 2*nuk/z*np.exp(k1-k0)))
        k0 = buff
        nuk += 1
        x_vec.append(k1)
        n_vec.append(nuk)
    return n_vec, x_vec


def seq(x, n, z, nb=100):
    n_vec = []
    x_vec = []
    x = dtype(x)
    n = dtype(n)
    for _ in range(nb):
        x_vec.append(x)
        n_vec.append(n)
        print(n, x)
        x = dtype(1/x + 2*n/z)

        n += 1

    return n_vec, x_vec


def main():
    plt.figure()
    ax2 = plt.subplot(111)
    n0 = .5

    for z, color in zip([1e2, 1e3, 1e4], plt.rcParams['axes.prop_cycle'].by_key()['color']):
        k0 = special.kve(n0, z)
        k1 = special.kve(n0 + 1, z)

        k0 = k0 * .1
        k1 = k1 * .1

        x0 = k1/k0

        n, y2 = seq(x0, n0+1, z, 50)
        y2 = np.array(y2)
        n = np.array(n)
        yt = np.log(special.kve(n, z)) - z

        y2 = np.cumsum(np.log(y2)) + np.log(k0) - z

        y1 = yt
        y = np.abs(np.abs(y1 - y2) / y1)

        msk = y1 > 0

        ax2.plot(n, y, ":+", color=color)
        ax2.plot(n[msk], y[msk], "x-", label=f"z= {z}, ", color=color)

        n, y2 = seq_log(n0, z, np.log(k0)-z, np.log(k1)-z, 50)
        n = np.array(n)
        yt = np.log(special.kve(n, z))-z
        y2 = np.array(y2)

        y1 = yt

        y = np.abs((yt - y2) / yt)

        ax2.plot(n, y, "-", color=color)
        msk = y1 > 0

        ax2.plot(n[msk], y[msk], ".-", label=f"z= {z}, ", color=color)
        # ax4.plot(n, y1, "o-", label=f"x = {x}, z= {z}, ")

    for ax in [ax2]:
        ax.grid(True)
        ax.semilogy()
        ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
