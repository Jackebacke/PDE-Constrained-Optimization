import pathlib

import matplotlib.pyplot as plt
import numpy as np
import poisson_discretization_DpDm as pd
from proj1_starter import b_fun, get_grid, subplots_2d, u_fun
from scipy.optimize import minimize

################ Constants and Setup ################
# Load data
path = pathlib.Path(__file__).parent
figures_path = path / "Figures"
Y_d = np.load(path / "y_data_v2.npy")  # Load data from file
# Build 21 x 21 grid
grid = get_grid()
X = grid["X"]
Y = grid["Y"]
Xv = grid["Xv"]
Yv = grid["Yv"]


b = b_fun(Xv, Yv)  # Thermal diffusivity b
u = u_fun(Xv, Yv)  # Source u

# Initial guess for a: a=1 everywhere
a_w, a_s, a_e, a_n = (np.ones_like(grid[f"X{side}"]) for side in "wsen")
a_init = np.concatenate((a_w, a_s, a_e, a_n))  # Order: W-S-E-N

# Solve for y_init using SBP discretization matrices
ORDER = 5
bc_opts = {"type": "robin", "a": a_init}
sbp_ops = pd.assemble_matrices(grid, ORDER, b, bc_opts)
y_init = sbp_ops["poisson_solver"](u)


def investigate_operators():
    """
    Investigate the SBP operators and their properties.
    """
    # Check size of the operators
    check_ops = {"a": a_init, "b": b, "u": u, "y": y_init, **sbp_ops}

    for name, op in check_ops.items():
        print(f"{name}: shape = {op.shape if hasattr(op, 'shape') else 'N/A'}")

    print("Initial a:", a_init)
    plot_boundary_values([a_init], title="Initial a values")
    plt.savefig(figures_path / "initial_a_values.png")


def plot_boundary_values(
    plotting_values: list,
    legends: list | None = None,
    title="Boundary Values",
    figsize=(8, 6),
):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    titles = ["West Boundary", "South Boundary", "East Boundary", "North Boundary"]
    Xw, Xs, Xe, Xn = grid["Xw"], grid["Xs"], grid["Xe"], grid["Xn"]
    Yw, _, Ye, _ = grid["Yw"], grid["Ys"], grid["Ye"], grid["Yn"]
    boundary = [Yw, Xs, Ye, Xn]
    xlabels = ["y", "x", "y", "x"]

    for values in plotting_values:
        values_split = [
            values[: len(Xw)],
            values[len(Xw) : len(Xw) + len(Xs)],
            values[len(Xw) + len(Xs) : len(Xw) + len(Xs) + len(Xe)],
            values[len(Xw) + len(Xs) + len(Xe) :],
        ]
        for i in range(4):
            axes[i].plot(boundary[i], values_split[i])

    for i in range(4):
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabels[i])
        axes[i].set_ylabel("Value")
        if legends is not None:
            axes[i].legend(legends)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()


####################### Gradients ########################


def cost(a, y_d):
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


def FD_gradient(a, y_d, delta=1e-6):
    """
    Compute the gradient of the loss function J(a) with respect to a using finite differences.

    Parameters:
    - a: Thermal diffusivity coefficients (array of shape (4*N,))
    - y_d: Observed data (array of shape (N,))
    - sbp_ops: Dictionary containing SBP operators and solver

    Returns:
    - grad_J: Gradient of the loss function with respect to a
    """

    def cost_fn(a):
        return cost(a, y_d)

    # Compute gradient of J with respect to a
    grad_J = np.zeros_like(a)
    for i in range(len(a)):
        a_perturbed = a.copy()
        a_perturbed[i] += delta  # Perturb a_i by a small amount
        grad_J[i] = (
            cost_fn(a_perturbed) - cost_fn(a)
        ) / delta  # Finite difference approximation

    return grad_J


def adjoint_gradient(a, y_d):
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

    # Solve forward problem
    y = sbp_ops["poisson_solver"](u)  # Solve for y(a)
    # Solve adjoint problem
    y_dagger = sbp_ops["poisson_solver"](y_d - y)  # Solve adjoint problem

    H_bnd = sbp_ops["H_bnd"]
    e_bnd = sbp_ops["e_bnd"]
    return -H_bnd @ (e_bnd.T @ y_dagger) * (e_bnd.T @ y)


def investigate_gradients():
    """
    Plot the initial gradients of the loss function with respect to a using both finite difference and adjoint methods.
    """
    grad_J_FD = FD_gradient(a_init, Y_d)
    grad_J_adjoint = adjoint_gradient(a_init, Y_d)

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
    plot_boundary_values(
        [grad_J_FD, grad_J_adjoint],
        title="Gradient Comparison",
        legends=["FD Gradient", "Adjoint Gradient"],
    )
    plt.savefig(figures_path / "gradient_comparison.png")
    plot_boundary_values(
        [diff], title="Absolute Difference between FD and Adjoint Gradients"
    )
    plt.savefig(figures_path / "gradient_difference.png")


def BFGS(cost_fn, grad_fn, a_init, y_d):
    def cost_function(a):
        return cost_fn(a, y_d)

    def jacobian(a):
        return grad_fn(a, y_d)

    iteration = 0

    def callback(a):
        nonlocal iteration
        iteration += 1
        current_cost = cost_function(a)
        print(f"Iteration {iteration}: cost = {current_cost:.6e}", end="\r")

    opts = {"disp": True, "maxiter": 1000, "return_all": True}

    result = minimize(
        fun=cost_function,
        x0=a_init,
        jac=jacobian,
        method="BFGS",
        callback=callback,
        options=opts,
    )

    return result


def investigate_optimization(
    result, title="Optimized State y", filename="optimized.png"
):
    # Get optimized a values
    a_solution = result.x
    print("a>0:", np.all(a_solution > 0))

    print("Success:", result.success)
    print("Message:", result.message)
    print("Final cost:", result.fun)

    bc_opts = {"type": "robin", "a": a_solution}
    sbp_ops = pd.assemble_matrices(grid, ORDER, b, bc_opts)
    y_solution = sbp_ops["poisson_solver"](u)

    # Plot a solution
    plot_boundary_values([a_solution], title=f"{title} Thermal Diffusivity a")
    plt.savefig(figures_path / f"{filename}_a.png", dpi=600, bbox_inches="tight")

    # Plot y solution
    fig, ax = plt.subplots(figsize=(6, 5))
    Y_sol = np.reshape(y_solution, X.shape)
    im = ax.pcolormesh(X, Y, Y_sol, shading="auto")
    ax.set_title(f"{title} state y", fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    plt.savefig(figures_path / f"{filename}_y.png", dpi=600, bbox_inches="tight")

    # Plot loss vs iteration
    if hasattr(result, "allvecs"):
        losses = [cost(a, Y_d) for a in result.allvecs]
        plt.figure(figsize=(6, 4))
        plt.semilogy(losses)
        plt.title("Loss vs Iteration", fontsize=12, fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Loss J(a)")
        plt.grid()
        plt.savefig(figures_path / f"{filename}_loss.png", dpi=600, bbox_inches="tight")


def initial_minimization():
    """
    Minimize and plot results.
    """
    # Initialize the optimization
    result = BFGS(cost, adjoint_gradient, a_init, Y_d)
    investigate_optimization(
        result, title="Initial optimization:", filename="initial_optimization"
    )

    return result


############## Reparameterization #############


def reparametrization(a, a0=1e-6, variable_out="a"):
    """
    Reparameterize the thermal diffusivity coefficients a to ensure positivity.
    """
    if variable_out == "a":
        return a0 + a**2
    elif variable_out == "a_l":
        return np.sqrt(a - a0)
    else:
        raise ValueError("Invalid variable type to receive. Use 'a' or 'a_l'.")


def reparametrized_adjoint_gradient(a_l, y_d, gradient_fn=adjoint_gradient):
    a = reparametrization(a_l, variable_out="a")
    grad_J = gradient_fn(a, y_d)
    grad_J_l = 2 * a_l * grad_J  # Chain rule for reparameterization
    return grad_J_l


def reparametrized_cost(a_l, y_d, cost_fn=cost):
    a = reparametrization(a_l, variable_out="a")
    return cost_fn(a, y_d)


def reparametrized_minimization():
    a_l_init = reparametrization(a_init, variable_out="a_l")
    result = BFGS(reparametrized_cost, reparametrized_adjoint_gradient, a_l_init, Y_d)

    # Convert back to a
    result.x = reparametrization(result.x, variable_out="a")
    result.allvecs = [
        reparametrization(a_l, variable_out="a") for a_l in result.allvecs
    ]

    # Plot results
    investigate_optimization(
        result,
        title="Reparameterized optimization",
        filename="reparam_optimization",
    )
    return result


################# Regularization ################
def TV_regularization(a, gamma=1e-6):
    bc_opts = {"type": "robin", "a": a}
    sbp_ops = pd.assemble_matrices(grid, ORDER, b, bc_opts)
    H_bnd = sbp_ops["H_bnd"]
    Dt = sbp_ops["Dt"]
    one = np.ones_like(a)
    W = (Dt @ a) ** 2 + gamma**2
    R = one.T @ H_bnd @ np.sqrt(W)
    # TODO: Check formula
    return R


def TV_gradient(a, gamma=1e-6):
    bc_opts = {"type": "robin", "a": a}
    sbp_ops = pd.assemble_matrices(grid, ORDER, b, bc_opts)
    H_bnd = sbp_ops["H_bnd"]
    Dt = sbp_ops["Dt"]
    one = np.ones_like(a)

    W = (Dt @ a) ** 2 + gamma**2

    grad_W = np.diag(2 * (Dt @ a)) @ Dt
    grad_R = 0.5 * one.T @ H_bnd @ np.diag(W ** (-0.5)) @ grad_W
    # TODO: REDO derivation to check this formula
    # TODO: Check chatGPT change of Dt

    return grad_R


def investigate_reg_gradients():
    """
    Plot the initial gradients of the regularization term with respect to a.
    """

    rng = np.random.default_rng(seed=42)
    a_test = 0.7 + 0.6 * rng.random(len(a_init))

    def cost_fn(a):
        return TV_regularization(a, gamma=1e-6)

    # FD gradient
    grad_FD = np.zeros_like(a_init)
    delta = 1e-6
    for i in range(len(a_init)):
        a_perturbed = a_test.copy()
        a_perturbed[i] += delta  # Perturb a_i by a small amount
        grad_FD[i] = (
            cost_fn(a_perturbed) - cost_fn(a_test)
        ) / delta  # Finite difference approximation
    # Exact gradient
    grad_R = TV_gradient(a_test)

    # Gradients to investigate
    fields = {
        "TV Regularization Gradient": grad_R,
        "Finite Difference Gradient": grad_FD,
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

    plot_boundary_values(
        [grad_R, grad_FD],
        title="TV Regularization Gradient",
        legends=["TV Gradient", "Finite Difference Gradient"],
    )
    plt.savefig(figures_path / "TV_gradient_comparison.png")

    diff = np.abs(grad_R - grad_FD)
    plot_boundary_values(
        [diff], title="TV Regularization Gradient Difference |FD - Exact|"
    )
    plt.savefig(figures_path / "TV_gradient_difference.png")


def regularized_reparametrized_cost(
    a_l,
    y_d,
    epsilon,
    gamma=1e-6,
):
    a = reparametrization(a_l, variable_out="a")
    return cost(a, y_d) + epsilon * TV_regularization(a, gamma)


def regularized_reparametrized_adjoint_gradient(
    a_l,
    y_d,
    epsilon,
    gamma=1e-6,
):
    a = reparametrization(a_l, variable_out="a")
    grad_J = adjoint_gradient(a, y_d)
    grad_R = TV_gradient(a, gamma)
    grad_J_l = (
        2 * a_l * (grad_J + epsilon * grad_R)
    )  # Chain rule for reparameterization
    return grad_J_l


def regularized_minimization(epsilon=1e-3, gamma=1e-6, plot=True):
    a_l_init = reparametrization(a_init, variable_out="a_l")

    result = BFGS(
        lambda a_l, y_d: regularized_reparametrized_cost(a_l, y_d, epsilon, gamma),
        lambda a_l, y_d: regularized_reparametrized_adjoint_gradient(
            a_l, y_d, epsilon, gamma
        ),
        a_l_init,
        Y_d,
    )

    # Convert back to a
    result.x = reparametrization(result.x, variable_out="a")
    result.allvecs = [
        reparametrization(a_l, variable_out="a") for a_l in result.allvecs
    ]

    # Plot results
    if plot:
        investigate_optimization(
            result,
            title=f"Regularized & Reparameterized \noptimization (epsilon={epsilon}):",
            filename=f"regularized_optimization_epsilon_{epsilon}",
        )
    return result


def sweep_epsilon_values(epsilon_values, gamma=1e-6):
    """
    Sweep over different epsilon values for regularization and plot results.
    """

    results = {}
    for epsilon in epsilon_values:
        print(f"Running regularized minimization with epsilon={epsilon}")
        result = regularized_minimization(epsilon=epsilon, gamma=gamma, plot=False)
        results[epsilon] = result

    print("Plotting results for all epsilon values.")

    # Plot loss vs iteration for each epsilon
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    for epsilon, result in results.items():
        print(f"Plotting results for epsilon={epsilon}")
        if hasattr(result, "allvecs"):
            losses = [cost(a, Y_d) for a in result.allvecs]
            objective_loss = [
                cost(a, Y_d) + epsilon * TV_regularization(a) for a in result.allvecs
            ]
            axes[0].semilogy(losses, label=f"epsilon={epsilon}")
            axes[1].semilogy(objective_loss, label=f"epsilon={epsilon}")

    axes[0].set_title("Data misfit loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss J(a)")
    axes[0].legend()
    axes[0].grid()

    axes[1].set_title("Objective loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss J(a)")
    axes[1].legend()
    axes[1].grid()

    plt.savefig(
        figures_path / "loss_vs_iteration_sweep_epsilon.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    #### Initial Minimization and Investigation ####
    # Look at the operators and their properties
    # investigate_operators()

    # Investigate gradient
    # investigate_gradients()

    # Perform initial minimization
    # result = initial_minimization()

    #### Reparameterization ####
    # Perform reparameterized minimization
    # result_reparam = reparametrized_minimization()

    #### Regularization ####
    # investigate_reg_gradients()

    # result_reg = regularized_minimization(epsilon=1e-3, gamma=1e-6)

    epsilon_values = [10 ** (i) for i in range(-6, 3)]
    sweep_epsilon_values(epsilon_values, gamma=1e-6)
    plt.show()
