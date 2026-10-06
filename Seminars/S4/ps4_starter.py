import pathlib

import matplotlib.pyplot as plt
import numpy as np
import poisson_discretization_DpDm as pd
import ps4_setup as setup
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplg
from matplotlib.colors import LogNorm, SymLogNorm

path = pathlib.Path(__file__).parent.resolve()


def q_to_y(q, config_dict):
    """Maps the parameter q to the state y."""
    A = system_matrix(q, config_dict)
    u = config_dict["u"]
    y = spsplg.spsolve(A, u)
    return y


def system_matrix(q, config_dict):
    """Return the system matrix A = D + diag(q),
    where D approximates the Laplacian + BC."""
    sbp_ops = config_dict["sbp_ops"]
    A = sbp_ops["Laplace_with_BC"] + spsp.diags_array(q)
    return A


def plot_setup():
    """Loads the setup and plots stuff"""

    # Load dictionary with setup
    config_dict = setup.setup()

    # Extract things
    y_data = config_dict["y_data"]
    q = config_dict["q"]
    q0 = config_dict["q0"]
    grid = config_dict["grid"]

    # Compute background state y0
    y0 = q_to_y(q0, config_dict)

    # Plot
    fig, axes = subplots_2d(grid, [q0, q, y0, y0 - y_data], (2, 2))
    axes[0].set_title("q0")
    axes[1].set_title("q")
    axes[2].set_title("y0")
    axes[3].set_title("y0 - y_d")
    plt.tight_layout()


def subplots_2d(grid, fields, sz=None, figsize=None, norms=None):
    """Helper function that plots several 2d plots in one figure.
    grid:   from pd.create_grid()
    fields: list of fields to plot. One subplot per field.
    sz:     subplot arrangement, e.g. (2, 3) for 2 rows, three columns.
    """
    X = grid["X"]
    Y = grid["Y"]
    fields = [np.reshape(f, X.shape) for f in fields]
    n_fields = len(fields)
    if sz is None:
        fig, axes = plt.subplots(n_fields, 1, figsize=figsize)
    else:
        fig, axes = plt.subplots(*sz, figsize=figsize)
        axes = axes.flatten()
    im = []
    for i in range(n_fields):
        norm = None if norms is None else norms[i]
        im.append(axes[i].pcolormesh(X, Y, fields[i], shading="auto", norm=norm))
        fig.colorbar(im[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
    return fig, axes


def main():
    plot_setup()
    plt.show()


def q_sc_exact(q0, alpha, config_dict):
    """Solves for q_sc exactly"""
    H = config_dict["sbp_ops"]["H"]
    y_d = config_dict["y_data"]
    y0 = q_to_y(q0, config_dict)

    A = system_matrix(q0, config_dict)
    Y0 = spsp.diags_array(y0)
    B = -spsplg.spsolve(A, Y0.toarray())
    d = y_d - y0

    normal_matrix = B.T @ H @ B + alpha * H
    return spsplg.spsolve(spsp.csc_matrix(normal_matrix), B.T @ H @ d)


def exercise_7():
    """Exercise 7: Solve for q_sc exactly"""
    config_dict = setup.setup()
    # Extract things
    y_data = config_dict["y_data"]
    q = config_dict["q"]
    grid = config_dict["grid"]
    q0 = config_dict["q0"]
    alpha = 1e-3
    q_sc = q_sc_exact(q0, alpha, config_dict)

    # Compute background state y0
    y0 = q_to_y(q0, config_dict)
    y_sc = q_to_y(q_sc, config_dict)
    y = q_to_y(q, config_dict)
    # Plot
    minq = min(np.min(q0), np.min(q), np.min(q_sc))
    maxq = max(np.max(q0), np.max(q), np.max(q_sc))
    miny = min(np.min(y0), np.min(y), np.min(y_sc))
    maxy = max(np.max(y0), np.max(y), np.max(y_sc))

    q_norm = SymLogNorm(linthresh=1e-4, vmin=minq, vmax=maxq)
    y_norm = LogNorm(vmin=miny, vmax=maxy)
    norms = [q_norm, q_norm, q_norm, y_norm, y_norm, y_norm]
    fig, axes = subplots_2d(
        grid, [q0, q, q_sc, y0, y, y_sc], (2, 3), figsize=(12, 8), norms=norms
    )
    axes[0].set_title("q0")
    axes[1].set_title("q")
    axes[2].set_title("q_sc")
    axes[3].set_title("y0")
    axes[4].set_title("y")
    axes[5].set_title("y_sc")

    plt.tight_layout()


def exercise_8(A=0.5, alpha=1e-3):
    """Exercise 8: Solve for q_sc exactly"""
    config_dict = setup.setup()
    # Extract things
    grid = config_dict["grid"]
    q0 = config_dict["q0"]

    # True q: three Gaussian patches
    Xv = grid["Xv"]
    Yv = grid["Yv"]
    q_sc_true = A * (
        pd.shifted_gaussian(Xv, Yv, 0.5, 0.4, 0.1)
        + pd.shifted_gaussian(Xv, Yv, 0.5, -0.4, 0.1)
        + pd.shifted_gaussian(Xv, Yv, -0.5, 0.0, 0.1)
    )
    q = 1 + q_sc_true
    q_sc = q_sc_exact(q0, alpha, config_dict)

    # Compute background state y0
    y_sc_true = q_to_y(q_sc_true, config_dict)
    y_sc = q_to_y(q_sc, config_dict)

    # Plot
    diff = q_sc_true - q_sc
    abs_diff = np.abs(diff)
    minq = min(np.min(q_sc_true), np.min(q_sc))
    maxq = max(np.max(q_sc_true), np.max(q_sc))
    miny = min(np.min(y_sc_true), np.min(y_sc))
    maxy = max(np.max(y_sc_true), np.max(y_sc))

    fig, axes = subplots_2d(
        grid,
        [q_sc_true, q_sc, diff, y_sc_true, y_sc, abs_diff],
        (2, 3),
        figsize=(12, 8),
    )
    fig.suptitle(f"Exercise 8: A={A}, alpha={alpha}")
    axes[0].set_title("q_sc_true")
    axes[1].set_title("q_sc")
    axes[2].set_title("diff")
    axes[3].set_title("y_sc_true")
    axes[4].set_title("y_sc")
    axes[5].set_title("abs_diff")

    # Set q and y to same color scale for comparison
    for i in range(2):
        axes[i].collections[0].set_clim(minq, maxq)
    for i in range(3, 5):
        axes[i].collections[0].set_clim(miny, maxy)
    plt.tight_layout()
    plt.savefig(f"{path}/Figures/Exercise8/exercise_8_A{A}_alpha{alpha}.png", dpi=300)


def exercise_8_sweep(A_values=[0.1, 0.5, 1.0], alpha_values=[1e-3, 1e-2, 1e-1]):
    """Exercise 8: Sweep over A and alpha values"""
    config_dict = setup.setup()
    # Extract things
    grid = config_dict["grid"]
    q0 = config_dict["q0"]
    Xv = grid["Xv"]
    Yv = grid["Yv"]
    H = config_dict["sbp_ops"]["H"]

    y_errors = np.zeros((len(A_values), len(alpha_values)))
    q_errors = np.zeros((len(A_values), len(alpha_values)))
    for A in A_values:
        for alpha in alpha_values:
            print(f"Computing for A={A}, alpha={alpha}", flush=True)
            q_sc_true = A * (
                pd.shifted_gaussian(Xv, Yv, 0.5, 0.4, 0.1)
                + pd.shifted_gaussian(Xv, Yv, 0.5, -0.4, 0.1)
                + pd.shifted_gaussian(Xv, Yv, -0.5, 0.0, 0.1)
            )
            q = 1 + q_sc_true
            q_sc = q_sc_exact(q0, alpha, config_dict)

            # Compute background state y0
            y_sc_true = q_to_y(q_sc_true, config_dict)
            y_sc = q_to_y(q_sc, config_dict)

            # Plot
            diff_qsc = q_sc_true - q_sc
            diff_y = y_sc_true - y_sc

            norm_qsc = np.sqrt(diff_qsc.T @ H @ diff_qsc)
            norm_y = np.sqrt(diff_y.T @ H @ diff_y)

            q_errors[A_values.index(A), alpha_values.index(alpha)] = norm_qsc
            y_errors[A_values.index(A), alpha_values.index(alpha)] = norm_y

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Plot q errors (labeled by A and alpha)
    vmin_q_error = np.min(q_errors)
    vmax_q_error = np.max(q_errors)
    im0 = axes[0].imshow(
        q_errors,
        cmap="viridis",
        origin="lower",
        norm=LogNorm(vmin=vmin_q_error, vmax=vmax_q_error),
    )
    axes[0].set_xticks(np.arange(len(alpha_values)))
    axes[0].set_xticklabels([f"{alpha:.1e}" for alpha in alpha_values], rotation=45)
    axes[0].set_yticks(np.arange(len(A_values)))
    axes[0].set_yticklabels([f"{A:.3f}" for A in A_values])
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("A")
    axes[0].set_title("q_sc error (norm)")
    fig.colorbar(im0, ax=axes[0])

    # Plot y errors (labeled by A and alpha)
    vmin_y_error = np.min(y_errors)
    vmax_y_error = np.max(y_errors)
    im1 = axes[1].imshow(
        y_errors,
        cmap="viridis",
        origin="lower",
        norm=LogNorm(vmin=vmin_y_error, vmax=vmax_y_error),
    )
    axes[1].set_xticks(np.arange(len(alpha_values)))
    axes[1].set_xticklabels([f"{alpha:.1e}" for alpha in alpha_values], rotation=45)
    axes[1].set_yticks(np.arange(len(A_values)))
    axes[1].set_yticklabels([f"{A:.3f}" for A in A_values])
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("A")
    axes[1].set_title("y_sc error (norm)")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(f"{path}/Figures/Exercise8/exercise_8_errors.png", dpi=300)


if __name__ == "__main__":
    # main()
    # exercise_7()

    # for A in [0.1, 0.5, 1.0]:
    #     for alpha in [1e-3, 1e-2, 1e-1]:
    #         exercise_8(A=A, alpha=alpha)

    N = 10
    A_vals = list(np.logspace(-3, 2, N))
    alpha_vals = list(np.logspace(-4, 2, N))
    exercise_8_sweep(A_values=A_vals, alpha_values=alpha_vals)
    plt.show()
