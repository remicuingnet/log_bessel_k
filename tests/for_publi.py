#
# Copyright (c) 2023 Remi Cuingnet. All rights reserved.
#
# See LICENSE file in the project root for full license information.
#

# STL
from dataclasses import dataclass, field

# External
from matplotlib import gridspec, ticker
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

# Project
import random_variables


def draw_figure(nb_points=250, figsize=(8, 9), dtype=np.float64):

    fig_pdf = plt.figure("pdf", figsize=figsize)

    gs = gridspec.GridSpec(4, 1, wspace=.2, hspace=.4)
    ax_pdf0 = fig_pdf.add_subplot(gs[-1])
    colors = ['k'] + [plt.rcParams['axes.prop_cycle'].by_key()['color'][c] for c in [0, 3, 4]]
    for i, (df, color) in enumerate(zip([5, 50, 500], colors)):
        ax_pdf = fig_pdf.add_subplot(gs[i])

        rvs = [
            {
                "name": "Recursion",
                "rv": random_variables.student_t(df=df, loc=0, scale=1,
                                                 functor=random_variables.student_functor_rec(dtype)),
                "line": "-"
            },
            {
                "name": "FromRatio",
                "rv": random_variables.student_t(df=df, loc=0, scale=1,
                                                 functor=random_variables.student_functor_rec_from_ratio(dtype)),
                "line": "-."
            },
            {
                "name": "Direct",
                "rv": random_variables.student_t(df=df, loc=0, scale=1,
                                                 functor=random_variables.student_functor_direct(dtype)),
                "line": ":"
            },
            {
                "name": "LogDirect",
                "rv": random_variables.student_t(df=df, loc=0, scale=1,
                                                 functor=random_variables.student_functor_direct_log(dtype)),
                "line": "--"
            }
        ]

        @dataclass
        class Result:
            name: str = None
            rv: random_variables.RandomVariable = None
            taylor_correction: random_variables.TAYLOR_CORRECTION = -1
            x: list = field(default_factory=list)
            pdf_err: list = field(default_factory=list)
            pdf_abs_err: list = field(default_factory=list)
            pdf: list = field(default_factory=list)
            pdf0: list = field(default_factory=list)
            method: str = None
            line: str = None

        result: Result
        results = []
        for method in ["quad"]:
            for taylor_correction in [random_variables.TAYLOR_CORRECTION.ORDER_0]:
                for rv in rvs:
                    results.append(Result(**rv, taylor_correction=taylor_correction, method=method))

        for x in np.linspace(0, 5, nb_points):
            pdf0 = stats.t.pdf(x, df=df, loc=0, scale=1)
            # cdf0 = stats.t.cdf(x, df=df, loc=0, scale=1)
            for result in results:
                if result.method == "quad":
                    pdf = random_variables.pdf_from_cf(x, result.rv,
                                                       taylor_correction=result.taylor_correction,
                                                       limit=500,
                                                       dtype=dtype)
                    result.x.append(x)
                    result.pdf.append(pdf)
                    result.pdf_err.append(np.abs((np.float64(pdf)-pdf0)/pdf0))
                    result.pdf_abs_err.append(np.abs((np.float64(pdf)-pdf0)))
                    result.pdf0.append(pdf0)

        for result in results:
            name = f"{result.name}"
            print()
            print(name)
            print(result.taylor_correction)
            print("pdf0", result.pdf0)
            print("pdf", result.pdf)
            print("pdf err", result.pdf_err)

            ax_pdf.plot(result.x, result.pdf_err, result.line, color=color, label=name)
            ax_pdf.set_title(f"df = {df}")

        ax_pdf0.plot(result.x, result.pdf0, '-', color=color, label=f"df = {df}")
        ax_pdf.legend(loc=2)
        ax_pdf0.legend(loc=3)
        ax_pdf0.set_title("pdf")

        for ax in [ax_pdf, ax_pdf0]:
            ticker.LogLocator.MAXTICKS = 5000

            ax.semilogy()
            ax.set_xlim(0, 5)

            ax.xaxis.grid(True, 'major', lw=.25,  linestyle='-')
            ax.yaxis.grid(True, 'minor', lw=.25,  linestyle='-')
            ax.yaxis.grid(True, 'major', lw=1, linestyle='-')

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    draw_figure()
