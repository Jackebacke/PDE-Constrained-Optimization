import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spsplg
import setup_source_inversion as setup


def plot_setup():
    """Demo"""

    # Setup problem
    sbp_discr, grid, z, u0, alpha = setup.problem_setup()
    # Solve Poisson equation for the initial guess u0
    y0 = sbp_discr["poisson_solver"](u0)

    # Get grid coordinates and reshape values for plotting
    X, Y = grid["X"], grid["Y"]
    z = np.reshape(z, X.shape)
    u0 = np.reshape(u0, X.shape)
    y0 = np.reshape(y0, X.shape)

    # Plotting the target state z, initial guess u0, and the state y0 corresponding to the initial guess u0
    fig, axes = plt.subplots(3, 1)
    im1 = axes[0].pcolormesh(X, Y, z, shading="auto")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Target state z")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(X, Y, u0, shading="auto")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Initial guess u0")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].pcolormesh(X, Y, y0, shading="auto")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("State corresponding to initial guess")
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    # plt.show()


def compute_gradient_using_adjoint(u, z, alpha, sbp_discr):
    """Returns the gradient, obtained by solving the adjoint equation."""
    # Solve the forward Poisson equation for y
    y = sbp_discr["poisson_solver"](u)
    # Then solve adjoint equation for y_dagger
    y_dagger = sbp_discr["poisson_solver"](z - y)

    H = sbp_discr["H"]
    # Compute the gradient using the adjoint solution
    return (alpha * u - y_dagger).T @ H


def plot_gradient_adjoint(mx=41, my=41, order=5):
    """Plots the gradient of the objective function with respect to u."""
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx, my, order)

    gradient = compute_gradient_using_adjoint(u0, z, alpha, sbp_discr)
    X, Y = grid["X"], grid["Y"]
    grad = np.reshape(gradient, X.shape)

    plt.figure()
    im = plt.pcolormesh(X, Y, grad, shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        "Gradient of the objective function with respect to u, using adjoint method"
    )
    plt.colorbar(im)
    # plt.show()


def cost_funtion(u, z, alpha, sbp_discr):
    H = sbp_discr["H"]
    y = sbp_discr["poisson_solver"](u)
    return 0.5 * (y - z).T @ H @ (y - z) + 0.5 * alpha * u.T @ H @ u


def compute_gradient_using_fd(u, cost_function, sbp_discr, delta=1e-6, LU=True):
    """
    Returns the gradient , obtained using the brute - force approach with first - order finite differences.
    u - Current guess for the source
    cost_function(u) - Function that maps state and source to cost
    sbp_discr - Dict with SBP - related operators
    delta - Step size in the brute - force FD approx .
    LU - If True, use LU decomposition for solving the linear system.
    """
    if LU:
        # Precompute the LU decomposition of the Poisson operator for efficiency
        D = sbp_discr["Laplace_with_BC"]
        lu = spsplg.splu(D)
        sbp_discr["poisson_solver"] = lambda u: lu.solve(u)

    gradient = np.zeros_like(u)
    for i in range(len(u)):
        e_i = np.eye(len(u))[i]
        gradient[i] = cost_function(u + delta * e_i) - cost_function(u)
    gradient /= delta
    return gradient


def plot_gradient_fd(mx=11, my=11, order=5):
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx, my, order)
    gradient = compute_gradient_using_fd(
        u0, lambda u: cost_funtion(u, z, alpha, sbp_discr), sbp_discr
    )

    # Plotting
    X, Y = grid["X"], grid["Y"]
    grad = np.reshape(gradient, X.shape)
    plt.figure()
    im = plt.pcolormesh(X, Y, grad, shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        "Gradient of the objective function with respect to u, using finite differences"
    )
    plt.colorbar(im)
    plt.tight_layout()


def plot_gradient_difference(mx=11, my=11, order=5):
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx, my, order)
    gradient_adjoint = compute_gradient_using_adjoint(u0, z, alpha, sbp_discr)
    gradient_fd = compute_gradient_using_fd(
        u0, lambda u: cost_funtion(u, z, alpha, sbp_discr), sbp_discr
    )

    # Plotting the difference between the two gradients
    X, Y = grid["X"], grid["Y"]
    grad_diff = np.reshape(gradient_adjoint - gradient_fd, X.shape)
    plt.figure()
    im = plt.pcolormesh(X, Y, grad_diff, shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Difference between gradient methods (adjoint - finite differences)")
    plt.colorbar(im)
    plt.tight_layout()


def gradient_descent(
    u0, z, alpha, sbp_discr, step_size=1e3, tolerance=10e-4, num_iterations=10e4
):
    """Performs gradient descent to minimize the cost function."""

    def _norm(v, H):
        """Compute the SBP norm of a vector."""
        return np.sqrt(v.T @ H @ v)

    u = u0.copy()
    H = sbp_discr["H"]
    relative_residuals = [np.inf]
    i = 0
    while relative_residuals[-1] > tolerance and i < num_iterations:
        gradient = compute_gradient_using_adjoint(u, z, alpha, sbp_discr)
        u_next = u - step_size * gradient

        # Compute the relative residual
        u_next_norm = _norm(u_next, H)
        residual_norm = _norm(u_next - u, H)
        # print(f"Iteration {i}: relative residual = {relative_residuals[-1]:.2e}, norm of u_next = {u_next_norm:.2e}, norm of residual = {residual_norm:.2e}")
        relative_residuals.append(residual_norm / u_next_norm)

        # Next iteration
        u = u_next
        i += 1
    if relative_residuals[-1] <= tolerance:
        print(
            f"Gradient descent converged in {int(i)} iterations with relative residual {relative_residuals[-1]:.2e}"
        )
        converged = True
    else:
        print(
            f"Gradient descent did not converge in {int(num_iterations)} iterations, final relative residual: {relative_residuals[-1]:.2e}"
        )
        converged = False
    # Final solution
    y = sbp_discr["poisson_solver"](u)

    return y, u, relative_residuals, i, converged


def exercise_10():
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx=21, my=21, order=5)

    start = time.time()
    gradient_fd = compute_gradient_using_fd(
        u0, lambda u: cost_funtion(u, z, alpha, sbp_discr), sbp_discr, LU=False
    )
    end = time.time()
    print(f"Time taken without LU decomposition: {end - start:.6f} seconds")

    start_LU = time.time()
    gradient_fd = compute_gradient_using_fd(
        u0, lambda u: cost_funtion(u, z, alpha, sbp_discr), sbp_discr, LU=True
    )
    end_LU = time.time()
    print(
        f"Time taken by finite difference method with LU decomposition: {end_LU - start_LU:.6f} seconds"
    )


def calculate_and_plot_GD(
    mx=21, my=21, order=5, step_size=1e3, tolerance=1e-4, num_iterations=1e4, plot=True
):
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx, my, order)
    y_final, u_final, relative_residuals, iterations, converged = gradient_descent(
        u0, z, alpha, sbp_discr, step_size, tolerance, num_iterations
    )

    # Plotting the final solution and the relative residuals
    if plot:
        X, Y = grid["X"], grid["Y"]
        U_final = np.reshape(u_final, X.shape)
        Y_final = np.reshape(y_final, X.shape)
        Z = np.reshape(z, X.shape)

        fig, axes = plt.subplots(4, 1)

        im1 = axes[0].pcolormesh(X, Y, Z, shading="auto")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Target state z")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(X, Y, U_final, shading="auto")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Final source u after gradient descent")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(X, Y, Y_final, shading="auto")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("Final state y after gradient descent")
        fig.colorbar(im3, ax=axes[2])

        im4 = axes[3].semilogy(relative_residuals)
        axes[3].set_xlabel("Iteration")
        axes[3].set_ylabel("Relative Residual")
        axes[3].set_title("Convergence of Gradient Descent")

        plt.tight_layout()

    return relative_residuals, iterations, converged


def exercise_12():
    # Exercise 12:
    tol = 1e-4
    step_sizes = np.logspace(1, 5, 100)  # Step sizes from 10^1 to 10^5
    relative_residuals_list = []
    iterations_list = []
    converged_list = []
    print("---------------- Exercise 12: ----------------")
    print(f"Testing step sizes: {step_sizes}")
    for beta in step_sizes:
        print(f"Running gradient descent with step size {beta}")
        relative_residuals, iterations, converged = calculate_and_plot_GD(
            step_size=beta, plot=False, tolerance=tol
        )
        relative_residuals_list.append(relative_residuals)
        iterations_list.append(iterations)
        converged_list.append(converged)

    # Plot iterations vs step size and non-convergence cases
    plt.figure()
    first = True
    for i, beta in enumerate(step_sizes):
        if converged_list[i]:
            plt.scatter(
                beta,
                iterations_list[i],
                color="blue",
                label="Converged" if i == 0 else "",
            )
        else:
            plt.scatter(
                beta,
                iterations_list[i],
                color="red",
                label="Not Converged" if first else "",
            )
            first = False
    plt.xscale("log")
    plt.xlabel("Step Size (beta)")
    plt.ylabel("Number of Iterations")
    plt.title(f"Gradient Descent Convergence with Tolerance {tol}")
    plt.legend()
    plt.grid()
    plt.savefig("gradient_descent_convergence.png", dpi=300)


if __name__ == "__main__":
    plot_setup()
    # plot_gradient(compute_gradient_using_adjoint, mx=41, my=41)
    plot_gradient_adjoint()
    plot_gradient_fd(mx=21, my=21)
    plot_gradient_difference(
        mx=21, my=21
    )  # The difference is very small, indicating that the adjoint method is correctly implemented. I think that the difference is mainly due to the finite difference approximation, which is a first-order approximation. Smaller delta would maybe decrease the difference.

    # Exercise 9: The Adjoint method solves the PDE twice, once for y and then for y_dagger. The finite difference method solves the PDE once for each component of u = mx*my times.

    # Exercise 10:
    exercise_10()

    # Exercise 11:
    calculate_and_plot_GD()

    # Exercise 12:
    exercise_12()

    plt.show()
