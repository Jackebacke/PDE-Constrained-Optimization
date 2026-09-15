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


def exercise_22(mx=21, my=21, plot=True):
    # Setup problem
    grid, bc_opts, initial_parameters, true_parameters, order = setup.problem_setup(
        mx=mx, my=my
    )
    # Data and initial guess
    y_d = true_parameters["y_data"]
    b_true = true_parameters["b"]
    u = true_parameters["u"]
    b0 = initial_parameters["b"]

    result = BFGS(cost_function, y_d, b0, u, grid, order, bc_opts)
    b_sol = result.x
    final_discr = pd.assemble_matrices(grid, order, b_sol, bc_opts)
    y_sol = final_discr["poisson_solver"](b_sol)

    # Plotting the results
    if plot:
        X, Y = grid["X"], grid["Y"]
        B_sol = np.reshape(b_sol, X.shape)
        Y_sol = np.reshape(y_sol, X.shape)
        B_true = np.reshape(b_true, X.shape)
        Y_d = np.reshape(true_parameters["y_data"], X.shape)

        fig, axes = plt.subplots(4, 1, figsize=(8, 12))
        fig.suptitle("Exercise 22: BFGS optimization")

        im1 = axes[0].pcolormesh(X, Y, Y_d, shading="auto")
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")
        axes[0].set_title("Target state y_d")
        fig.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].pcolormesh(X, Y, B_true, shading="auto")
        axes[1].set_xlabel("$x_1$")
        axes[1].set_ylabel("$x_2$")
        axes[1].set_title("True material b")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(X, Y, B_sol, shading="auto")
        axes[2].set_xlabel("$x_1$")
        axes[2].set_ylabel("$x_2$")
        axes[2].set_title("Final material b after BFGS")
        fig.colorbar(im3, ax=axes[2])

        im3 = axes[3].pcolormesh(X, Y, Y_sol, shading="auto")
        axes[3].set_xlabel("$x_1$")
        axes[3].set_ylabel("$x_2$")
        axes[3].set_title("Final state y after BFGS")
        fig.colorbar(im3, ax=axes[3])
        plt.tight_layout()
        
        plt.savefig("./Seminars/S1_S2/Figures/exercise22.png")
    return result


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


def exercise_23(mx=21, my=21, plot=True):
    # Setup problem
    grid, bc_opts, initial_parameters, true_parameters, order = setup.problem_setup(
        mx=mx, my=my
    )
    # Data and initial guess
    y_d = true_parameters["y_data"]
    b_init = initial_parameters["b"]
    b_true = true_parameters["b"]
    u = true_parameters["u"]

    result = BFGS_bl(cost_function, b_init, y_d, u, grid, order, bc_opts)
    bl_sol = result.x
    b_sol = f(bl_sol)
    final_discr = pd.assemble_matrices(grid, order, b_sol, bc_opts)
    y_sol = final_discr["poisson_solver"](true_parameters["u"])

    print("Minimum b:", np.min(b_sol))
    print("Maximum b:", np.max(b_sol))
    print("Constraint b > 0 satisfied:", np.all(b_sol > 0))

    if plot:
        X, Y = grid["X"], grid["Y"]
        B_sol = np.reshape(b_sol, X.shape)
        B_true = np.reshape(b_true, X.shape)
        Y_sol = np.reshape(y_sol, X.shape)
        Y_d = np.reshape(true_parameters["y_data"], X.shape)

        fig, axes = plt.subplots(4, 1, figsize=(8, 12))
        fig.suptitle("Exercise 23: BFGS optimization with transformation")

        im1 = axes[0].pcolormesh(X, Y, Y_d, shading="auto")
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")
        axes[0].set_title("Target state y_d")
        fig.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].pcolormesh(X, Y, B_true, shading="auto")
        axes[1].set_xlabel("$x_1$")
        axes[1].set_ylabel("$x_2$")
        axes[1].set_title("True material b")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(X, Y, B_sol, shading="auto")
        axes[2].set_xlabel("$x_1$")
        axes[2].set_ylabel("$x_2$")
        axes[2].set_title("Final material b after BFGS with transformation")
        fig.colorbar(im3, ax=axes[2])

        im3 = axes[3].pcolormesh(X, Y, Y_sol, shading="auto")
        axes[3].set_xlabel("$x_1$")
        axes[3].set_ylabel("$x_2$")
        axes[3].set_title("Final state y after BFGS with transformation")
        fig.colorbar(im3, ax=axes[3])
        plt.tight_layout()
        
        plt.savefig("./Seminars/S1_S2/Figures/exercise23.png")
    return result


if __name__ == "__main__":
    # # -------------------------- Medium inversion ---------------------------
    # # Exercise 19:
    plot_setup()

    # # Exercise 20:
    plot_gradient_adjoint(mx=21, my=21)

    # # Exercise 21:
    plot_gradient_difference(mx=21, my=21)

    # Exercise 22:
    exercise_22(mx=21, my=21, plot=True)
    
    # Exercise 23:
    exercise_23(mx=21, my=21, plot=True)
    plt.show()
