import pathlib

import matplotlib.pyplot as plt
import numpy as np
import poisson_discretization_DpDm as pd
import sbp_operators as sbp
from proj1_starter import b_fun, get_grid, subplots_2d, u_fun

# Load data
path = pathlib.Path(__file__).parent
Y_d = np.load(path / "y_data.npy")  # Load data from file
# Build 21 x 21 grid
grid = get_grid()
X = grid["X"]
Y = grid["Y"]
Xv = grid["Xv"]
Yv = grid["Yv"]


b = b_fun(Xv, Yv)  # Thermal diffusivity b
u = u_fun(Xv, Yv)  # Source u

# Initial guess for a: a=1 everywhere
a_w = 0.0 * grid["Xw"] + 1.0
a_s = 0.0 * grid["Xs"] + 1.0
a_e = 0.0 * grid["Xe"] + 1.0
a_n = 0.0 * grid["Xn"] + 1.0
a0 = np.concatenate([a_w, a_s, a_e, a_n])  # Order: W-S-E-N

# Solve for y_init using SBP discretization matrices
bc_opts = {"type": "robin", "a": a0}
ORDER = 5
sbp_ops = pd.assemble_matrices(grid, ORDER, b, bc_opts)
y0 = sbp_ops["poisson_solver"](u)


def investigate_operators():
    """
    Investigate the SBP operators and their properties.
    """
    # Check size of the operators
    check_ops = {"a": a0, "b": b, "u": u, "y": y0, **sbp_ops}

    for name, op in check_ops.items():
        print(f"{name}: shape = {op.shape if hasattr(op, 'shape') else 'N/A'}")
        
    print("a:", a0)


def loss(a, y_d, sbp_ops):
    """
    Compute the loss function J(a) = ||y(a) - y_d||^2.

    Parameters:
    - a: Thermal diffusivity coefficients (array of shape (4*N,))
    - y_d: Observed data (array of shape (N,))
    - sbp_ops: Dictionary containing SBP operators and solver

    Returns:
    - L: Loss value
    """
    # Update boundary conditions with new a
    bc_opts = {"type": "robin", "a": a}
    sbp_ops = pd.assemble_matrices(grid, ORDER, b, bc_opts)
    y = sbp_ops["poisson_solver"](u)  # Solve for y(a)

    # Compute loss
    H = sbp_ops["H"]  # SBP norm matrix
    J = 0.5 * (y - y_d).T @ H @ (y - y_d)  # Loss function J(a) = 1/2*||y - y_d||^2

    return J


def FD_gradient(a, y_d, sbp_ops, delta=1e-6):
    """
    Compute the gradient of the loss function J(a) with respect to a using finite differences.

    Parameters:
    - a: Thermal diffusivity coefficients (array of shape (4*N,))
    - y_d: Observed data (array of shape (N,))
    - sbp_ops: Dictionary containing SBP operators and solver

    Returns:
    - grad_J: Gradient of the loss function with respect to a
    """

    def cost(a):
        return loss(a, y_d, sbp_ops)

    # Compute gradient of J with respect to a
    grad_J = np.zeros_like(a)
    for i in range(len(a)):
        a_perturbed = a.copy()
        a_perturbed[i] += delta  # Perturb a_i by a small amount
        grad_J[i] = (
            cost(a_perturbed) - cost(a)
        ) / delta  # Finite difference approximation

    return grad_J


def boundary_indices(grid):
    """Map each boundary point (order W-S-E-N) to its index in the flattened grid."""
    xb = np.concatenate([grid["Xw"], grid["Xs"], grid["Xe"], grid["Xn"]])
    yb = np.concatenate([grid["Yw"], grid["Ys"], grid["Ye"], grid["Yn"]])
    xv, yv = np.ravel(grid["Xv"]), np.ravel(grid["Yv"])
    return np.array([np.argmin((xv - x) ** 2 + (yv - y) ** 2) for x, y in zip(xb, yb)])


def adjoint_gradient(a, y_d, sbp_ops):
    """
    Compute the gradient of the loss function J(a) with respect to a using the adjoint method.

    Parameters:
    - a: Thermal diffusivity coefficients (array of shape (4*N,))
    - y_d: Observed data (array of shape (N,))
    - sbp_ops: Dictionary containing SBP operators and solver

    Returns:
    - grad_J: Gradient of the loss function with respect to a
    """
    # Update boundary conditions with new a
    bc_opts = {"type": "robin", "a": a}
    sbp_ops = pd.assemble_matrices(grid, ORDER, b, bc_opts)
    y = sbp_ops["poisson_solver"](u)  # Solve for y(a)

    y_dagger = sbp_ops["poisson_solver"](y_d - y)  # Solve adjoint problem

    BIDX = boundary_indices(grid)

    y_b = np.ravel(y)[BIDX]
    yd_b = np.ravel(y_dagger)[BIDX]
    H_bnd = sbp_ops["H_bnd"]
    return -H_bnd * y_b * yd_b  # TODO: Hur???????


def investigate_gradients():
    """
    Plot the initial gradients of the loss function with respect to a using both finite difference and adjoint methods.
    """
    grad_J_FD = FD_gradient(a0, Y_d, sbp_ops)
    grad_J_adjoint = adjoint_gradient(a0, Y_d, sbp_ops)

    # Gradients to investigate
    fields = {
        "Finite Difference Gradient": grad_J_FD,
        "Adjoint Gradient": grad_J_adjoint,
    }
    for title, grad in fields.items():
        print("Investigating:", title)
        print("Gradient shape:", grad.shape)
        print(
            "Mean:",
            np.mean(grad),
            "Std:",
            np.std(grad),
            "Max:",
            np.max(grad),
            "Min:",
            np.min(grad),
        )

    diff = np.abs(grad_J_FD - grad_J_adjoint)
    print("Max difference between FD and adjoint gradients:", np.max(diff))


def reparametrization(a, a0=1e-6, variable="a"):
    """
    Reparameterize the thermal diffusivity coefficients a to ensure positivity.
    """
    if variable == "a":
        return a0 + a**2
    elif variable == "a_l":
        return np.sqrt(a - a0)
    else:
        raise ValueError("Invalid variable type. Use 'a' or 'a_l'.")


def reparametrized_gradient(a_l, y_d, sbp_ops):
    a = reparametrization(a_l, variable="a")
    grad_J = adjoint_gradient(a, y_d, sbp_ops)
    grad_J_l = 2 * a_l * grad_J  # Chain rule for reparameterization
    return grad_J_l


def regularization():
    pass


if __name__ == "__main__":
    investigate_operators()
    # investigate_gradients()
    plt.show()
