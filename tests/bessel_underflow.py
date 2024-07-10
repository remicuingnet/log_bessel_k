#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

from scipy import special
import numpy as np
import matplotlib.pyplot as plt


def inv_(nu, l, l_pow10):
    log_l = np.log(l)+np.log(10)*l_pow10
    log_x = (2*nu-1)*np.log(2) + np.log(np.pi) - 2*log_l
    return .5*(log_x - np.log(log_x) + np.e/(np.e-1)*np.log(log_x)/log_x)


def inv_2(nu, l, l_pow10):
    log_l = np.log(l)+np.log(10)*l_pow10
    log_x = (2*nu-1)*np.log(2) + np.log(np.pi) - 2*log_l
    print(np.log(2), (np.log(np.pi) - 2*log_l-np.log(2))/2)
    return .5 * (log_x)


def inv_up(p, p_pow10, alpha=1):
    log_p = np.log(p)+p_pow10*np.log(10)
    log_a = .5*np.log(np.pi/(2*alpha))-2/(alpha*np.e)  # + np.log(.5)
    print("loga, ", log_a)
    return np.real(.5 + (log_p-log_a)/special.lambertw(alpha*(log_p-log_a), 0))


def inv_up2(p, p_pow10, z):
    z = np.array(z)
    log_p = np.log(p)+p_pow10*np.log(10)
    log_a = -z + .5*np.log(np.pi/np.e/z)
    print("loga, ", log_a)
    return np.real(.5 + (log_p-log_a)/special.lambertw((2/np.e/z)*(log_p-log_a), 0))


def inv_up2_log2(p, p_pow2,z):
    z = np.array(z)
    log_p = np.log(p)+p_pow2*np.log(2)
    log_a = -z + .5*np.log(np.pi/np.e/z)
    return np.real(.5 + (log_p-log_a)/special.lambertw((2/np.e/z)*(log_p-log_a), 0))


def inv_up3_log2(p, p_pow2,z):
    e = np.e
    z = np.array(z)
    log_p = np.log(p)+p_pow2*np.log(2)
    log_a = -z + .5*np.log(np.pi/e/z)
    return np.real(.5 + (log_p-log_a)/special.lambertw((2/e/z)*(log_p-log_a), 0))


# def inv_low2(p, p_pow10, nu):
#     log_x0 = (2*nu-1)*np.log(2) + np.log(np.pi)-2*(p_pow10*np.log(10) + np.log(p))
#     return log_x0/2


def inv_low2(p, p_pow2, nu):
    log_x0 = (2*nu-1)*np.log(2) + np.log(np.pi)-2*(p_pow2*np.log(2) + np.log(p))
    return log_x0/2


def inv_low2_bis(p, p_pow2, nu):
    log_x = (2*nu-1)*np.log(2) + np.log(np.pi)-2*(p_pow2*np.log(2) + np.log(p))
    return .5*(log_x - np.log(log_x) + np.e/(np.e-1)*np.log(log_x)/log_x)


def inv_low_ub(p, p_pow2, nu):

    log_x0 = (0*nu+1)*np.log(np.pi)-2*(p_pow2*np.log(2) + np.log(p))  # - .5*np.log(2)
    return (log_x0 - np.log(log_x0))/2


def inv_up2_pow2(p, p_pow2, nu):
    log_p = np.log(p)+p_pow2*np.log(2)
    log_x =-(np.log(2) + log_p - special.gammaln(nu))/nu + np.log(2/nu)
    res  = (log_x - np.log(log_x))*nu
    msk = np.isnan(res)
    res[msk] = inv_low_ub(p, p_pow2, nu[msk])
    return res




def dicho_search(upper_test, lo=0, up=0, factor=2, offset=1, eps=1e-6):
    if upper_test(lo):
        # lo is too large:
        lo = lo/factor - offset
        return dicho_search(upper_test, lo, up, factor, offset, eps)
    if not upper_test(up):
        # up is too large
        up = up*factor + offset
        return dicho_search(upper_test, lo, up, factor, offset, eps)

    mid = (lo+up)/2
    if upper_test(mid):
        up = mid
    else:
        lo = mid
    if 2*(up-lo)/(up+lo+eps) < eps:
        return lo, up
    return dicho_search(upper_test, lo, up, factor, offset, eps)


class TestOverFlow:
    def __init__(self, z, t=np.float32):
        self.z = z
        self.t = t

    def __call__(self, nu):
        res = special.kv(nu, self.z)
        res = self.t(res)
        return np.isinf(res)


class TestUnderFlowDenormal:
    def __init__(self, nu, t=np.float32):
        self.nu = nu
        self.t = t

    def __call__(self, z):
        res = special.kv(self.nu, z)
        res = self.t(res)
        return res == 0


class TestUnderFlowNormal:
    def __init__(self, nu, log2_lim=-126):
        self.nu = nu
        self.log2_lim = log2_lim

    def __call__(self, z):
        res = special.kv(self.nu, z)

        return res == 0 or np.log2(res) < self.log2_lim


p_float_up = 3.403, 38
p_float_lo = 1.175, -38
p_float_lo = 1.174, -38


def low(p, p_pow10, z):
    # sufficient condition for non overflow
    log_p = np.log(p)+np.log(10)*p_pow10
    log_k = -z+.5*np.log(np.pi/2/z)
    log_x0 = (.5-np.e*z/2)*np.log(1+2/np.e/z)
    log_x0 = log_p-log_k+log_x0

    log_x0 *= 2/np.e/z

    alpha = np.real(log_x0/special.lambertw(log_x0))
    return (alpha-1)/(2/np.e/z)

    # y = log_p + z - np.log(np.pi/2/z)
    y = log_p -np.log((special.kv(.5, z)))

    ez = 2/(np.e*z)

    return (ez * y/(np.log(ez) + np.log(y)) -1)/ez


def low_log2(p, p_pow2, z):
    # sufficient condition for non overflow
    log_p = np.log(p)+np.log(2)*p_pow2
    # log_k = np.log(special.kv(.5, z))
    log_k = -z+.5*np.log(np.pi/2/z)
    log_x0 = (.5-np.e*z/2)*np.log(1+2/np.e/z)
    log_x0 = log_p-log_k+log_x0

    log_x0 *= 2/np.e/z

    alpha = np.real(log_x0/special.lambertw(log_x0))
    return (alpha-1)/(2/np.e/z)

    # y = log_p + z - np.log(np.pi/2/z)
    y = log_p -np.log((special.kv(.5, z)))

    ez = 2/(np.e*z)

    return (ez * y/(np.log(ez) + np.log(y)) -1)/ez


def low_up(p, p_pow10, z):
    # sufficient condition for non underflow
    log_p = np.log(p)+np.log(10)*p_pow10
    #log_k = np.log(special.kv(.5, z))
    log_k = -z+.5*np.log(np.pi/2/z)

    log_x0 = (2*z+.5)*np.log(1+1/2/z)
    log_x0 = log_p+log_x0-log_k
    log_x0 *= 2*z+.5

    alpha = np.real(log_x0/special.lambertw(log_x0))

    return (alpha-1)*2*z


def main():
    nu = 20
    step = 1e-6
    nb_steps = 5000

    z_float = inv_(nu, 1.175, -38)
    z_float2 = inv_2(nu, 1.175, -38)
    z_double = inv_(nu, 2.225, -308)
    z_double2 = inv_2(nu, 2.225, -308)
    print("float", z_float)
    print("float2", z_float2)
    print(special.kv(nu, z_float))
    print(np.float32(special.kv(nu, z_float)))
    print(np.float32(special.kv(nu, z_float)) == 0)

    print("double", z_double)
    print("double2", z_double2)
    print(special.kv(nu, z_double))

    print()
    alpha = 1
    z = 2 / np.e / alpha
    nu_float = inv_up(3.403, 38, alpha)
    nu_float2 = inv_up2(3.403, 38, z)

    nu_double = inv_up(1.798, 308, alpha)

    print("nu float", nu_float)
    print("nu float2", nu_float2)
    print(special.kv(nu_float, z))

    print("nu double", nu_double)
    print(special.kv(nu_double, z))

    z = np.logspace(np.log10(step), np.log10(2000), nb_steps)
    nu = np.linspace(1, 1800, nb_steps)

    plt.figure(figsize=(6.4, 2.4))
    plt.semilogx()
    plt.gca().set_yscale("symlog")
    plt.grid(True)
    color = "tab:orange"
    plt.plot(z, inv_up2_log2((1-2**-23), 128, z), '--', color=color)
    plt.plot(z, inv_up3_log2((1-2**-23), 128, z), '--', color="tab:red")
    plt.plot(z, [dicho_search(TestOverFlow(float(z_), np.float32))[1] for z_ in z], '-', color=color)
    plt.plot(z, low_log2((1-2**-23), 128, z), ':', color=color)


    color="tab:blue"
    plt.plot([dicho_search(TestUnderFlowNormal(float(nu_), -126))[1] for nu_ in nu], nu, '-', color=color)
    plt.plot(inv_low2(1, -126, nu),  nu, "--", color=color)
    plt.plot(inv_low_ub(1, -126, nu), nu, ":", color=color)
    plt.autoscale(enable=True, tight=True)

    plt.xlabel(r"$z$")
    plt.ylabel(r"$\nu$")
    plt.ylim((None, 400))
    plt.title("single-precision")
    plt.tight_layout()

    z = np.logspace(np.log10(step), np.log10(2000), nb_steps)
    nu = np.linspace(1, 1800, nb_steps)
    plt.semilogx()
    plt.gca().set_yscale("symlog")
    plt.grid(True)
    color = "tab:brown"
    plt.plot(z, inv_up2_log2((1-2**-52), 1024, z), '--', color=color)
    plt.plot(z, [dicho_search(TestOverFlow(z_, np.float64))[1] for z_ in z], '-', color=color)
    plt.plot(z, low_log2((1-2**-52), 1024, z), ':', color=color)

    color = "tab:grey"
    plt.plot([dicho_search(TestUnderFlowNormal(float(nu_), -1022))[1] for nu_ in nu], nu, '-', color=color)
    plt.plot(inv_low2(1, -1022, nu),  nu, "--", color=color)
    plt.plot(inv_low_ub(1, -1022, nu), nu, ":", color=color)
    plt.autoscale(enable=True, tight=True)

    plt.xlabel(r"$z$")
    plt.ylabel(r"$\nu$")
    plt.text(.01, 500, "overflow", ha="center")
    plt.text(1200, 50, "underflow", ha="center", va="center_baseline", rotation="vertical")
    plt.text(.01, 2, "positive normal floating-point",ha="center")
    plt.title("double-precision")
    plt.title(None)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
