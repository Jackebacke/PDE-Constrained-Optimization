import pathlib

import matplotlib.pyplot as plt
import numpy as np
import poisson_discretization_DpDm as pd


def get_grid():
    """Builds a 21x21 grid in [-1,1] x [-1, 1]"""

    # Domain and discretization parameters
    lim_x = [-1, 1]
    lim_y = [-1, 1]
    mx = 21
    my = 21

    # Create grid
    grid = pd.create_grid(mx, my, lim_x, lim_y)

    return grid


def plot_data_proj1():
    """Starting point for Part 1. Shows how to load:
    data (y_data)
    source (u)
    thermal diffusivity (b)"""
    path = pathlib.Path(__file__).parent

    # Load data from file
    y_data = np.load(f"{path}/y_data.npy")

    # Build 21 x 21 grid
    grid = get_grid()
    X = grid["X"]
    Y = grid["Y"]
    Xv = grid["Xv"]
    Yv = grid["Yv"]

    # Source u
    u = u_fun(Xv, Yv)

    # Thermal diffusivity b
    b = b_fun(Xv, Yv)

    # Initial guess for a: a=1 everywhere
    a_w = 0.0 * grid["Xw"] + 1.0
    a_s = 0.0 * grid["Xs"] + 1.0
    a_e = 0.0 * grid["Xe"] + 1.0
    a_n = 0.0 * grid["Xn"] + 1.0
    a = np.concatenate([a_w, a_s, a_e, a_n])  # Note order: W-S-E-N

    # Specify Robin BC with parameter a
    bc_opts = {"type": "robin", "a": a}

    # Build SBP discretization matrices
    order = 5
    sbp_ops = pd.assemble_matrices(grid, order, b, bc_opts)
    y_init = sbp_ops["poisson_solver"](u)

    # Plot y_data and source u
    fig, axes = subplots_2d(grid["X"], grid["Y"], [y_data, u, b, y_init], sz=(2, 2))
    axes[0].set_title("y_data")
    axes[1].set_title("source u")
    axes[2].set_title("Thermal diffusivity b")
    axes[3].set_title("State y with a=1 everywhere")
    plt.tight_layout()
    plt.show()


def u_fun(X, Y):
    """Source function"""
    x0 = 0.5
    y0 = 0
    width = 0.2
    u = -1e3 * pd.shifted_gaussian(X, Y, x0, y0, width) - 1e3 * pd.shifted_gaussian(
        X, Y, -x0, y0, width
    )
    return u


def b_fun(X, Y):
    """Thermal diffusivity b"""
    return 10.0 + 2 * np.sin(np.pi * X) + 2 * np.cos(np.pi * Y)


def subplots_2d(X, Y, fields, sz=None):
    """Helper function that plots several 2d plots in one figure.
    X, Y: Coordinate arrays (from np.meshgrid)
    fields: list of fields to plot. One subplot per field.
    """
    fields = [np.reshape(f, X.shape) for f in fields]
    n_fields = len(fields)
    if sz is None:
        fig, axes = plt.subplots(n_fields, 1)
    else:
        fig, axes = plt.subplots(*sz)
        axes = axes.flatten()
    im = []
    for i in range(n_fields):
        im.append(axes[i].pcolormesh(X, Y, fields[i], shading="auto"))
        fig.colorbar(im[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
    return fig, axes


def main():
    plot_data_proj1()


if __name__ == "__main__":
    main()
