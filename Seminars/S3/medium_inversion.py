import matplotlib.pyplot as plt
import numpy as np
import poisson_discretization_DpDm as pd
import setup_medium_inversion as setup
from scipy.optimize import minimize


def plot_setup():
    # Setup problem
    grid, bc_opts, initial_parameters, true_parameters, order = setup.problem_setup()
    # Data and initial guess
    b0 = initial_parameters["b"]
    y_d = true_parameters["y_data"]
    u = true_parameters["u"]

    # Assemble operators for initial guess
    sbp_discr = pd.assemble_matrices(grid, order, b0, bc_opts)

    # Calculate the state corresponding to the initial guess
    y0 = sbp_discr["poisson_solver"](u)

    # Get grid coordinates and reshape values for plotting
    X, Y = grid["X"], grid["Y"]
    Y_d = np.reshape(y_d, X.shape)
    B0 = np.reshape(b0, X.shape)
    Y0 = np.reshape(y0, X.shape)

    # Plot data y_d, initial guess b0, and state y corresponding to initial guess
    fig, axes = plt.subplots(3, 1)
    im1 = axes[0].pcolormesh(X, Y, Y_d, shading="auto")
    axes[0].set_title("Data y_d")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(X, Y, B0, shading="auto")
    axes[1].set_title("Initial guess b0")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].pcolormesh(X, Y, Y0, shading="auto")
    axes[2].set_title("State y corresponding to initial guess")
    axes[2].set_xlabel("$x_1$")
    axes[2].set_ylabel("$x_2$")
    fig.colorbar(im3, ax=axes[2])

    plt.suptitle("Exercise 19: Setup of the medium inversion problem")
    plt.tight_layout()


def compute_gradient_using_adjoint(b, y_d, u, grid, order, bc_opts):
    """Returns the gradient, obtained by solving the adjoint equation."""
    current_discr = pd.assemble_matrices(grid, order, b, bc_opts)
    # Solve the forward Poisson equation for y
    y = current_discr["poisson_solver"](u)
    # Then solve adjoint equation for y_dagger
    y_dagger = current_discr["poisson_solver"](y_d - y)

    H = current_discr["H"]
    Dx_m = current_discr["Dx_m"]
    Dy_m = current_discr["Dy_m"]
    # Compute the gradient using the adjoint solution
    return -H @ ((Dx_m @ y_dagger) * (Dx_m @ y)) - H @ ((Dy_m @ y_dagger) * (Dy_m @ y))


def plot_gradient_adjoint(mx=41, my=41):
    """Plots the gradient of the objective function with respect to b."""
    # Setup problem
    grid, bc_opts, initial_parameters, true_parameters, order = setup.problem_setup(
        mx=mx, my=my
    )
    # Data and initial guess
    y_d = true_parameters["y_data"]
    b0 = initial_parameters["b"]
    u = true_parameters["u"]

    gradient = compute_gradient_using_adjoint(b0, y_d, u, grid, order, bc_opts)
    X, Y = grid["X"], grid["Y"]
    grad = np.reshape(gradient, X.shape)

    plt.figure()
    im = plt.pcolormesh(X, Y, grad, shading="auto")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Exercise 20: Adjoint Gradient of objective function w.r.t b")
    plt.colorbar(im)
    plt.tight_layout()


def cost_function(b, y_d, u, grid, order, bc_opts):
    current_discr = pd.assemble_matrices(grid, order, b, bc_opts)
    y = current_discr["poisson_solver"](u)
    H = current_discr["H"]
    return 0.5 * (y - y_d).T @ H @ (y - y_d)


def compute_gradient_using_fd(b, y_d, u, grid, order, bc_opts, delta=1e-6):
    """
    Returns the gradient , obtained using the brute - force approach with first - order finite differences.
    b - Current guess for the material parameter.
    grid - The computational grid.
    order - The order of the SBP scheme.
    bc_opts - Boundary condition options.
    delta - Step size in the brute - force FD approx .
    """

    def cost(b):
        return cost_function(b, y_d, u, grid, order, bc_opts)

    # Compute the base cost for the current guess
    gradient = np.zeros_like(b)
    for i in range(len(b)):
        b_perturbed = b.copy()
        b_perturbed[i] += delta  # Perturb the i-th component
        gradient[i] = cost(b_perturbed) - cost(b)
    gradient /= delta

    return gradient


def plot_gradient_difference(mx=21, my=21):
    # Setup problem
    grid, bc_opts, initial_parameters, true_parameters, order = setup.problem_setup(
        mx=mx, my=my
    )
    # Data and initial guess
    b0 = initial_parameters["b"]
    y_d = true_parameters["y_data"]
    u = true_parameters["u"]

    # Compute the gradients using both methods
    gradient_adjoint = compute_gradient_using_adjoint(b0, y_d, u, grid, order, bc_opts)
    gradient_fd = compute_gradient_using_fd(b0, y_d, u, grid, order, bc_opts)

    # Plotting the difference between the two gradients
    X, Y = grid["X"], grid["Y"]
    grad_adj = np.reshape(gradient_adjoint, X.shape)
    grad_fd = np.reshape(gradient_fd, X.shape)
    grad_diff = np.reshape(gradient_adjoint - gradient_fd, X.shape)
    abs_diff = np.abs(grad_diff)

    # Plotting [[gradient adjoint, gradient fd], [abs difference, difference]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    im1 = axes[0, 0].pcolormesh(X, Y, grad_adj, shading="auto")
    axes[0, 0].set_title("Gradient (Adjoint)")
    axes[0, 0].set_xlabel("$x_1$")
    axes[0, 0].set_ylabel("$x_2$")
    fig.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].pcolormesh(X, Y, grad_fd, shading="auto")
    axes[0, 1].set_title("Gradient (Finite Difference)")
    axes[0, 1].set_xlabel("$x_1$")
    axes[0, 1].set_ylabel("$x_2$")
    fig.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].pcolormesh(X, Y, abs_diff, shading="auto")
    axes[1, 0].set_title("Absolute Difference")
    axes[1, 0].set_xlabel("$x_1$")
    axes[1, 0].set_ylabel("$x_2$")
    fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(X, Y, grad_diff, shading="auto")
    axes[1, 1].set_title("Difference (Adjoint - FD)")
    axes[1, 1].set_xlabel("$x_1$")
    axes[1, 1].set_ylabel("$x_2$")
    fig.colorbar(im4, ax=axes[1, 1])

    fig.suptitle("Exercise 21: Difference between gradient methods")
    plt.tight_layout()


def BFGS(cost_function, y_d, b0, u, grid, order, bc_opts):
    # BFGS solver options
    opts = {"disp": True, "maxiter": 500}
    # BFGS solver options

    def F(b):
        return cost_function(b, y_d, u, grid, order, bc_opts)

    def gradient_func(b):
        return compute_gradient_using_adjoint(b, y_d, u, grid, order, bc_opts)

    # Solve using BFGS

    out = minimize(
        F,
        b0,
        method="BFGS",
        jac=gradient_func,
        options=opts,
    )
    return out


def f(b_l, b0=0.01):
    b = b0 + b_l**2
    return b


def b_l_from_b(b, b0=0.01):
    b_l = np.sqrt(b - b0)
    return b_l


def BFGS_bl(cost_function, b_initial, y_d, u, grid, order, bc_opts, b0=0.01):
    # BFGS solver options
    opts = {"disp": True, "maxiter": 500}

    def F(b_l):
        b = f(b_l, b0)
        return cost_function(b, y_d, u, grid, order, bc_opts)

    def gradient_bl(b_l):
        b = f(b_l, b0)
        grad_b = compute_gradient_using_adjoint(b, y_d, u, grid, order, bc_opts)
        return 2 * b_l * grad_b

    # Solve using BFGS
    b_l_initial = b_l_from_b(b_initial, b0)
    out = minimize(F, b_l_initial, method="BFGS", jac=gradient_bl, options=opts)
    return out


def Tikhonov_regularization(b, H, epsilon=1e-1):
    """Applies Tikhonov regularization to the material parameter b."""
    cost = 0.5 * b.T @ H @ b
    return epsilon * cost


def Tikhonov_gradient(b, H, epsilon=1e-1):
    """Computes the gradient of the Tikhonov regularization term."""
    return epsilon * b.T @ H


def H1_regularization(b, H, Dx_m, Dy_m, epsilon=1e-1):
    """Applies H1 regularization to the material parameter b."""
    cost = 0.5 * ((Dx_m @ b).T @ H @ (Dx_m @ b) + (Dy_m @ b).T @ H @ (Dy_m @ b))
    return epsilon * cost


def H1_gradient(b, H, Dx_m, Dy_m, epsilon=1e-1):
    """Computes the gradient of the H1 regularization term."""
    return epsilon * b.T @ (Dx_m.T @ H @ Dx_m + Dy_m.T @ H @ Dy_m)


def TV_regularization(b, H, Dx_m, Dy_m, epsilon=1e-1, gamma=1e-6):
    """Applies Total Variation (TV) regularization to the material parameter b."""
    one_vector = np.ones_like(b)
    cost = one_vector.T @ H @ ((Dx_m @ b) ** 2 + (Dy_m @ b) ** 2 + gamma**2) ** 0.5
    return epsilon * cost


def TV_gradient(b, H, Dx_m, Dy_m, epsilon=1e-1, gamma=1e-6):
    """Computes the gradient of the Total Variation (TV) regularization term."""
    w = (Dx_m @ b) ** 2 + (Dy_m @ b) ** 2 + gamma**2
    grad_w = np.diag(2 * Dx_m @ b) @ Dx_m + np.diag(2 * Dy_m @ b) @ Dy_m
    one = np.ones_like(b)
    return epsilon * one.T @ H @ np.diag(0.5 * w ** (-0.5)) @ grad_w


def cost_function_with_regularization(
    b, y_d, u, grid, order, bc_opts, reg_type="Tikhonov", epsilon=1e-3
):
    current_discr = pd.assemble_matrices(grid, order, b, bc_opts)
    y = current_discr["poisson_solver"](u)
    H = current_discr["H"]
    Dx_m = current_discr["Dx_m"]
    Dy_m = current_discr["Dy_m"]

    # Compute the data misfit term
    data_misfit = 0.5 * (y - y_d).T @ H @ (y - y_d)

    # Compute the regularization term based on the specified type
    if reg_type == "Tikhonov":
        reg_term = Tikhonov_regularization(b, H, epsilon)
    elif reg_type == "H1":
        reg_term = H1_regularization(b, H, Dx_m, Dy_m, epsilon)
    elif reg_type == "TV":
        reg_term = TV_regularization(b, H, Dx_m, Dy_m, epsilon)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")

    return data_misfit + reg_term


def gradient_with_regularization(
    b, y_d, u, grid, order, bc_opts, reg_type="Tikhonov", epsilon=1e-3
):
    current_discr = pd.assemble_matrices(grid, order, b, bc_opts)
    H = current_discr["H"]
    Dx_m = current_discr["Dx_m"]
    Dy_m = current_discr["Dy_m"]

    def data_misfit_gradient(b):
        y = current_discr["poisson_solver"](u)
        y_dagger = current_discr["poisson_solver"](y_d - y)
        return -H @ ((Dx_m @ y_dagger) * (Dx_m @ y)) - H @ (
            (Dy_m @ y_dagger) * (Dy_m @ y)
        )

    # Compute the gradient of the regularization term based on the specified type
    if reg_type == "Tikhonov":
        grad_reg_term = Tikhonov_gradient(b, H, epsilon)
    elif reg_type == "H1":
        grad_reg_term = H1_gradient(b, H, Dx_m, Dy_m, epsilon)
    elif reg_type == "TV":
        grad_reg_term = TV_gradient(b, H, Dx_m, Dy_m, epsilon)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")

    return data_misfit_gradient(b) + grad_reg_term


def FD_gradient(b, func, delta=1e-6):
    """Computes the gradient of a function using finite differences."""
    gradient = np.zeros_like(b)
    for i in range(len(b)):
        b_perturbed = b.copy()
        b_perturbed[i] += delta  # Perturb the i-th component
        gradient[i] = func(b_perturbed) - func(b)
    gradient /= delta
    return gradient


def exercise25():
    # Setup problem
    grid, bc_opts, initial_parameters, true_parameters, order = setup.problem_setup()
    b_true = true_parameters["b"]

    # Assemble matrices and vectors
    b_init = np.random.rand(*b_true.shape) * 10  # Random initial guess for b
    sbp_discr = pd.assemble_matrices(grid, order, b_init, bc_opts)
    H = sbp_discr["H"]
    Dx_m = sbp_discr["Dx_m"]
    Dy_m = sbp_discr["Dy_m"]

    grad_Tikhonov = Tikhonov_gradient(b_init, H)
    grad_Tikhonov_FD = FD_gradient(b_init, lambda b: Tikhonov_regularization(b_init, H))
    grad_diff_Tikhonov = np.abs(grad_Tikhonov - grad_Tikhonov_FD)

    grad_H1 = H1_gradient(b_init, H, Dx_m, Dy_m)
    grad_H1_FD = FD_gradient(b_init, lambda b: H1_regularization(b, H, Dx_m, Dy_m))
    grad_diff_H1 = np.abs(grad_H1 - grad_H1_FD)

    grad_TV = TV_gradient(b_init, H, Dx_m, Dy_m)
    grad_TV_FD = FD_gradient(b_init, lambda b: TV_regularization(b, H, Dx_m, Dy_m))
    grad_diff_TV = np.abs(grad_TV - grad_TV_FD)

    plots = {
        "Tikhonov": (grad_Tikhonov, grad_Tikhonov_FD, grad_diff_Tikhonov),
        "H1": (grad_H1, grad_H1_FD, grad_diff_H1),
        "TV": (grad_TV, grad_TV_FD, grad_diff_TV),
    }

    # Plotting the differences
    X, Y = grid["X"], grid["Y"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i, (reg_type, (grad, grad_fd, grad_diff)) in enumerate(plots.items()):
        grad_reshaped = np.reshape(grad, X.shape)
        grad_fd_reshaped = np.reshape(grad_fd, X.shape)
        grad_diff_reshaped = np.reshape(grad_diff, X.shape)

        im1 = axes[i, 0].pcolormesh(X, Y, grad_reshaped, shading="auto")
        axes[i, 0].set_title(f"{reg_type} Gradient (Exact)")
        fig.colorbar(im1, ax=axes[i, 0])

        im2 = axes[i, 1].pcolormesh(X, Y, grad_fd_reshaped, shading="auto")
        axes[i, 1].set_title(f"{reg_type} Gradient (Finite Difference)")
        fig.colorbar(im2, ax=axes[i, 1])

        im3 = axes[i, 2].pcolormesh(X, Y, grad_diff_reshaped, shading="auto")
        axes[i, 2].set_title(f"{reg_type} Absolute Difference")
        fig.colorbar(im3, ax=axes[i, 2])

    fig.suptitle("Exercise 25: Gradient Comparison for Regularization Methods")
    plt.tight_layout()
    plt.savefig("./Seminars/S3/Figures/exercise25.png")


def BFGS_regularized(
    b_init, y_d, u, grid, order, bc_opts, reg_type="Tikhonov", epsilon=1e-1
):
    # BFGS solver options
    opts = {"disp": True, "maxiter": 500, "return_all": True}

    def F_bl(b_l):
        b = f(b_l)
        return cost_function_with_regularization(
            b, y_d, u, grid, order, bc_opts, reg_type=reg_type, epsilon=epsilon
        )

    def gradient_bl(b_l):
        b = f(b_l)
        grad_b = gradient_with_regularization(
            b, y_d, u, grid, order, bc_opts, reg_type=reg_type, epsilon=epsilon
        )
        return 2 * b_l * grad_b

    # Solve using BFGS
    b_l_init = b_l_from_b(b_init)
    out = minimize(F_bl, b_l_init, method="BFGS", jac=gradient_bl, options=opts)
    out.cost_history = np.array([F_bl(b_l) for b_l in out.allvecs])
    out.x = f(out.x)  # Transform back to b

    return out


def exercise26(epsilon=1e-1):
    # Setup problem
    grid, bc_opts, _, true_parameters, order = setup.problem_setup()
    b_true = true_parameters["b"]
    y_d = true_parameters["y_data"]
    u = true_parameters["u"]
    rng = np.random.default_rng(0)
    b_init = 1.0 + 0.2 * rng.standard_normal(b_true.shape)
    b_init = np.maximum(b_init, 0.01)

    # Solve with Tikhonov regularization
    result_tikhonov = BFGS_regularized(
        b_init, y_d, u, grid, order, bc_opts, reg_type="Tikhonov", epsilon=epsilon
    )
    b_sol_tikhonov = result_tikhonov.x

    # Solve with H1 regularization
    result_h1 = BFGS_regularized(
        b_init, y_d, u, grid, order, bc_opts, reg_type="H1", epsilon=epsilon
    )
    b_sol_h1 = result_h1.x

    # Solve with TV regularization
    result_tv = BFGS_regularized(
        b_init, y_d, u, grid, order, bc_opts, reg_type="TV", epsilon=epsilon
    )
    b_sol_tv = result_tv.x

    # Plotting the results
    X, Y = grid["X"], grid["Y"]
    B_true = np.reshape(b_true, X.shape)
    B_sol_tikhonov = np.reshape(b_sol_tikhonov, X.shape)
    B_sol_h1 = np.reshape(b_sol_h1, X.shape)
    B_sol_tv = np.reshape(b_sol_tv, X.shape)

    fig, axes = plt.subplots(4, 1, figsize=(6, 8))
    fig.suptitle(
        f"Exercise 26: BFGS Optimization with Regularization (ε={epsilon:.2e})"
    )

    im1 = axes[0].pcolormesh(X, Y, B_true, shading="auto")
    axes[0].set_title("True material b")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(X, Y, B_sol_tikhonov, shading="auto")
    axes[1].set_title("Final material b after Tikhonov Regularization")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].pcolormesh(X, Y, B_sol_h1, shading="auto")
    axes[2].set_title("Final material b after H1 Regularization")
    fig.colorbar(im3, ax=axes[2])

    im4 = axes[3].pcolormesh(X, Y, B_sol_tv, shading="auto")
    axes[3].set_title("Final material b after TV Regularization")
    fig.colorbar(im4, ax=axes[3])

    plt.tight_layout()
    plt.savefig(f"./Seminars/S3/Figures/exercise26_epsilon_{epsilon:.2e}.png")


def exercise27(epsilon_values=np.logspace(-5, 1, 10)):
    # Setup problem
    grid, bc_opts, _, true_parameters, order = setup.problem_setup()
    b_true = true_parameters["b"]
    y_d = true_parameters["y_data"]
    u = true_parameters["u"]
    rng = np.random.default_rng(0)
    b_init = 1.0 + 0.2 * rng.standard_normal(b_true.shape)
    b_init = np.maximum(b_init, 0.01)

    raw_cost_histories = {
        "Tikhonov": [],
        "H1": [],
        "TV": [],
    }
    normalized_cost_histories = {
        "Tikhonov": [],
        "H1": [],
        "TV": [],
    }

    for epsilon in epsilon_values:
        for reg_type in raw_cost_histories:
            result = BFGS_regularized(
                b_init,
                y_d,
                u,
                grid,
                order,
                bc_opts,
                reg_type=reg_type,
                epsilon=epsilon,
            )
            cost_history = result.cost_history
            raw_cost_histories[reg_type].append(cost_history)
            normalized_cost_histories[reg_type].append(cost_history / cost_history[0])

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), squeeze=False)
    regularizers = ("Tikhonov", "H1", "TV")

    for row, reg_type in enumerate(regularizers):
        for i, epsilon in enumerate(epsilon_values):
            label = f"ε={epsilon:.1e}"
            axes[row, 0].semilogy(raw_cost_histories[reg_type][i], label=label)
            axes[row, 1].semilogy(normalized_cost_histories[reg_type][i], label=label)

        axes[row, 0].set_title(f"{reg_type}: raw cost")
        axes[row, 1].set_title(f"{reg_type}: normalized cost")
        axes[row, 0].set_ylabel("Cost")
        axes[row, 1].set_ylabel("Cost / initial cost")
        axes[row, 0].set_xlabel("Iteration")
        axes[row, 1].set_xlabel("Iteration")
        axes[row, 0].legend()
        axes[row, 1].legend()

    fig.suptitle("Exercise 27: Cost histories")
    plt.tight_layout()
    plt.savefig("./Seminars/S3/Figures/exercise27_cost_history.png")


if __name__ == "__main__":
    ###################### Seminar 3 #########################
    exercise25()

    exercise26(epsilon=1e0)  # Too large for all
    # exercise26(epsilon=1e-1) # Too large for all
    # exercise26(epsilon=1e-2) # Too large for H1, smooths out way too much
    # exercise26(epsilon=1e-3) # Good for Tikhonov and H1, too large for TV
    # exercise26(epsilon=1e-4) # Good for all methods
    # exercise26(epsilon=1e-6) # Too small for all, leads to oscillations

    exercise27(epsilon_values=np.array([1e-6, 1e-4, 1e-2, 1e0]))  
    
    
    plt.show()
