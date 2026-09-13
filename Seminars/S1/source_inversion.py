import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spsplg
import setup_source_inversion as setup
from scipy.optimize import minimize


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


def cost_funtion(y, u, z, alpha, sbp_discr):
    H = sbp_discr["H"]
    return 0.5 * (y - z).T @ H @ (y - z) + 0.5 * alpha * u.T @ H @ u


def compute_gradient_using_fd(u, cost_function, sbp_discr, delta=1e-6, LU=True):
    """
    Returns the gradient , obtained using the brute - force approach with first - order finite differences.
    u - Current guess for the source
    cost_function(y, u) - Function that maps state and source to cost
    sbp_discr - Dict with SBP - related operators
    delta - Step size in the brute - force FD approx .
    LU - If True, use LU decomposition for solving the linear system.
    """
    if LU:
        # Precompute the LU decomposition of the Poisson operator for efficiency
        D = sbp_discr["Laplace_with_BC"]
        lu = spsplg.splu(D)

        def solve_state(source):
            return lu.solve(source)
    else:
        solve_state = sbp_discr["poisson_solver"]

    y = solve_state(u)
    base_cost = cost_function(y, u)
    gradient = np.zeros_like(u)
    for i in range(len(u)):
        perturbed_u = u.copy()
        perturbed_u[i] += delta
        perturbed_y = solve_state(perturbed_u)
        gradient[i] = cost_function(perturbed_y, perturbed_u) - base_cost
    gradient /= delta
    return gradient


def plot_gradient_fd(mx=11, my=11, order=5):
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx, my, order)
    gradient = compute_gradient_using_fd(
        u0,
        lambda y, u: cost_funtion(y, u, z, alpha, sbp_discr),
        sbp_discr,
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
        u0,
        lambda y, u: cost_funtion(y, u, z, alpha, sbp_discr),
        sbp_discr,
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


def line_search(
    costfunction,
    state_solver,
    u,
    gradient,
    initial_step_size=1.0,
    armijo_param=0.5,
    step_reduction=0.5,
    max_iterations=20,
):
    """
    Perform a backtracking line search to find an appropriate step size.
    Input:
    - costfunction(y, u): The cost function to minimize.
    - state_solver(u): Computes the state corresponding to a source.
    - u: Current point in the optimization.
    - gradient: The gradient at the current point.
    - initial_step_size: Initial guess for the step size.
    - armijo_param: Parameter for the Armijo condition (0 < armijo_param < 1).
    - step_reduction: Factor by which to reduce the step size if the Armijo condition is not satisfied (0 < step_reduction < 1).
    - max_iterations: Maximum number of iterations to try for the line search.
    Output:
    - step_size: The step size that satisfies the Armijo condition.
    """
    p = -gradient  # Descent direction
    m = np.dot(gradient, p)  # local slope
    step_size = initial_step_size
    current_cost = costfunction(state_solver(u), u)
    for _ in range(max_iterations):
        trial_u = u + step_size * p
        new_cost = costfunction(state_solver(trial_u), trial_u)
        if new_cost <= current_cost + step_size * armijo_param * m:  # Armijo condition
            return step_size
        step_size *= step_reduction  # Reduce the step size
    return step_size  # Return the last tried step size if no suitable one was found


def gradient_descent(
    u0,
    z,
    alpha,
    sbp_discr,
    perform_line_search=True,
    step_size=1e3,
    tolerance=10e-4,
    num_iterations=10e4,
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
        if perform_line_search:
            step_size = line_search(
                lambda y, u: cost_funtion(y, u, z, alpha, sbp_discr),
                sbp_discr["poisson_solver"],
                u,
                gradient,
                initial_step_size=step_size,
            )
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
        u0,
        lambda y, u: cost_funtion(y, u, z, alpha, sbp_discr),
        sbp_discr,
        LU=False,
    )
    end = time.time()
    print(f"Time taken without LU decomposition: {end - start:.6f} seconds")

    start_LU = time.time()
    gradient_fd = compute_gradient_using_fd(
        u0,
        lambda y, u: cost_funtion(y, u, z, alpha, sbp_discr),
        sbp_discr,
        LU=True,
    )
    end_LU = time.time()
    print(
        f"Time taken by finite difference method with LU decomposition: {end_LU - start_LU:.6f} seconds"
    )


def calculate_and_plot_GD(
    mx=21,
    my=21,
    order=5,
    step_size=1e3,
    tolerance=1e-4,
    num_iterations=1e4,
    plot=True,
    do_line_search=True,
):
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx, my, order)
    y_final, u_final, relative_residuals, iterations, converged = gradient_descent(
        u0,
        z,
        alpha,
        sbp_discr,
        step_size=step_size,
        tolerance=tolerance,
        num_iterations=num_iterations,
        perform_line_search=do_line_search,
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


def exercise_13(initial_step_size=5000):
    start = time.time()
    calculate_and_plot_GD(step_size=initial_step_size, plot=True, do_line_search=False)
    print(f"Time taken with fixed step size: {time.time() - start:.2f} seconds")
    start = time.time()
    calculate_and_plot_GD(step_size=initial_step_size, plot=True, do_line_search=True)
    print(f"Time taken with line search: {time.time() - start:.2f} seconds")


def BFGS(cost_function, u0, cost_args):
    # BFGS solver options
    opts = {"disp": True, "maxiter": 500}
    # BFGS solver options
    sbp_discr = cost_args[-1]

    def cost_func(u):
        y = sbp_discr["poisson_solver"](u)
        return cost_function(y, u, *cost_args)

    def gradient_func(u):
        return compute_gradient_using_adjoint(u, *cost_args)

    # Solve using BFGS
    out = minimize(
        cost_func,
        u0,
        method="BFGS",
        jac=gradient_func,
        options=opts,
    )
    return out


def exercise_14(plot=True):
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx=21, my=21, order=5)
    cost_args = (z, alpha, sbp_discr)

    result = BFGS(cost_funtion, u0, cost_args)
    u_sol = result.x
    y_sol = sbp_discr["poisson_solver"](u_sol)

    # Plotting the results
    if plot:
        X, Y = grid["X"], grid["Y"]
        U_sol = np.reshape(u_sol, X.shape)
        Y_sol = np.reshape(y_sol, X.shape)
        Z = np.reshape(z, X.shape)

        fig, axes = plt.subplots(3, 1)

        im1 = axes[0].pcolormesh(X, Y, Z, shading="auto")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Target state z")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].pcolormesh(X, Y, U_sol, shading="auto")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Final source u after BFGS")
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].pcolormesh(X, Y, Y_sol, shading="auto")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("Final state y after BFGS")
        fig.colorbar(im3, ax=axes[2])

        plt.tight_layout()
    return result


def exact_solution(z, alpha, sbp_discr):
    """Compute the exact solution for the source inversion problem."""
    H = sbp_discr["H"]
    D = sbp_discr["Laplace_with_BC"]
    D_inv = spsplg.inv(D)

    # Solve for u using the formula u = (D^(-T) * H * D^(-1) + alpha * H)^(-1) * (D^(-T) * H * z)
    RHS = D_inv.T @ H @ D_inv + alpha * H
    LHS = D_inv.T @ H @ z
    u_exact = spsplg.spsolve(RHS, LHS)
    return u_exact


def exercise_15():
    sbp_discr, grid, z, u0, alpha = setup.problem_setup(mx=31, my=31, order=5)
    u_exact = exact_solution(z, alpha, sbp_discr)
    y_exact = sbp_discr["poisson_solver"](u_exact)

    BFGS_result = BFGS(cost_funtion, u0, (z, alpha, sbp_discr))
    u_sol = BFGS_result.x
    y_sol = sbp_discr["poisson_solver"](u_sol)

    # Plotting the results
    X, Y = grid["X"], grid["Y"]
    U_exact = np.reshape(u_exact, X.shape)
    Y_exact = np.reshape(y_exact, X.shape)
    Z = np.reshape(z, X.shape)
    U_sol = np.reshape(u_sol, X.shape)
    Y_sol = np.reshape(y_sol, X.shape)
    U_diff = np.reshape(abs(U_sol - U_exact), X.shape)
    Y_diff = np.reshape(abs(Y_sol - Y_exact), X.shape)

    fig, axes = plt.subplots(4, 1)

    im2 = axes[0].pcolormesh(X, Y, U_exact, shading="auto")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Exact source u")
    fig.colorbar(im2, ax=axes[0])

    im3 = axes[1].pcolormesh(X, Y, Y_exact, shading="auto")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Exact state y")
    fig.colorbar(im3, ax=axes[1])

    im4 = axes[2].pcolormesh(X, Y, U_diff, shading="auto")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("|U_BFGS - U_exact|")
    fig.colorbar(im3, ax=axes[2])

    im4 = axes[3].pcolormesh(X, Y, Y_diff, shading="auto")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    axes[3].set_title("|Y_BFGS - Y_exact|")
    fig.colorbar(im4, ax=axes[3])

    plt.tight_layout()


def exercise_16(alpha=1e-4, plot=True):
    sbp_discr, grid, z, u0, _ = setup.problem_setup(mx=11, my=11, order=5)
    u_exact = exact_solution(z, alpha, sbp_discr)
    y_exact = sbp_discr["poisson_solver"](u_exact)

    # Solve using BFGS
    BFGS_result = BFGS(cost_funtion, u0, (z, alpha, sbp_discr))
    iterations_BFGS = BFGS_result.nit
    converged_BFGS = BFGS_result.success

    # Solve using gradient descent
    _, _, _, iterations_GD, converged_GD = gradient_descent(
        u0,
        z,
        alpha,
        sbp_discr,
        perform_line_search=True,
        step_size=1e3,
        tolerance=1e-4,
        num_iterations=1e4,
    )

    # L2 error calculations
    H = sbp_discr["H"]
    l2_error = np.sqrt((y_exact - z).T @ H @ (y_exact - z))

    if plot:
        # Plot
        X, Y = grid["X"], grid["Y"]
        U_exact = np.reshape(u_exact, X.shape)
        Y_exact = np.reshape(y_exact, X.shape)
        Z = np.reshape(z, X.shape)
        Diff = np.reshape(abs(Y_exact - Z), X.shape)

        fig, axes = plt.subplots(4, 1)
        im0 = axes[0].pcolormesh(X, Y, Z, shading="auto")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Target state z")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(X, Y, U_exact, shading="auto")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Exact source u")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].pcolormesh(X, Y, Y_exact, shading="auto")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("Exact state y")
        fig.colorbar(im2, ax=axes[2])

        im3 = axes[3].pcolormesh(X, Y, Diff, shading="auto")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")
        axes[3].set_title("|Y_exact - Z| ")
        fig.colorbar(im3, ax=axes[3])

        plt.suptitle(f"Alpha = {alpha}, L2 error = {l2_error:.2e}")
        plt.tight_layout()

    return l2_error, iterations_BFGS, converged_BFGS, iterations_GD, converged_GD


def exercise_16_sweep(alphas = [10**(i) for i in range(-6, 6)]):
    L2_Errors = []
    BFGS_results = []
    GD_results = []
    for a in alphas:
        print(f"Running exercise_16 with alpha = {a}")
        l2_error, iter_BFGS, converged_BFGS, iter_GD, converged_GD = exercise_16(
            alpha=a, plot=False
        )
        L2_Errors.append(l2_error)
        BFGS_results.append((iter_BFGS, converged_BFGS))
        GD_results.append((iter_GD, converged_GD))

    # Plots
    fig, axes = plt.subplots(2, 1)
    axes[0].semilogx(alphas, L2_Errors, marker="o")
    axes[0].set_xlabel("Regularization parameter alpha")
    axes[0].set_ylabel("L2 error")
    axes[0].set_title("L2 error vs Regularization parameter alpha")
    axes[0].grid()

    bfgs_iterations = [res[0] for res in BFGS_results]
    gd_iterations = [res[0] for res in GD_results]
    bfgs_colors = ["tab:blue" if res[1] else "tab:red" for res in BFGS_results]
    gd_colors = ["tab:orange" if res[1] else "tab:red" for res in GD_results]

    axes[1].semilogx(alphas, bfgs_iterations, label="BFGS", color="tab:blue")
    axes[1].semilogx(alphas, gd_iterations, label="Gradient Descent", color="tab:orange")
    axes[1].scatter(alphas, bfgs_iterations, c=bfgs_colors, marker="o")
    axes[1].scatter(alphas, gd_iterations, c=gd_colors, marker="o")
    axes[1].set_xlabel("Regularization parameter alpha")
    axes[1].set_ylabel("Number of iterations")
    axes[1].set_title("Number of iterations vs Regularization parameter alpha")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    figures_dir = Path(__file__).resolve().parent / "Figures"
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / "alpha_sweep_results.png", dpi=300)


if __name__ == "__main__":
    # Exercise 5:
    # plot_setup()

    # Exercise 6:
    # plot_gradient_adjoint()

    # Exercise 7:
    # plot_gradient_fd(mx=21, my=21)
    # plot_gradient_difference(mx=21, my=21)
    # The difference is very small, indicating that the adjoint method is correctly implemented. I think that the difference is mainly due to the finite difference approximation, which is a first-order approximation. Smaller delta would maybe decrease the difference.

    # Exercise 9: The Adjoint method solves the PDE twice, once for y and then for y_dagger. The finite difference method solves the PDE once for each component of u = mx*my + 1 times.

    # Exercise 10:
    # exercise_10()

    # Exercise 11:
    # calculate_and_plot_GD()

    # Exercise 12:
    # exercise_12()

    ########################## Seminar 2 ###########################

    # Exercise 13:
    # exercise_13()

    # Sometimes the line search takes longer, but it can also lead to faster convergence in terms of iterations. For some big initial step sizes, line search can lead to convergence where fixed step size does not converge. For smaller initial step sizes, line search can lead to slower convergence in terms of iterations, but it can also lead to faster convergence in terms of time.
    # The line search adapts the step size based on the local landscape of the cost function, which can be beneficial in cases where a fixed step size might be too large or too small.
    # Initial step size: Just a guess, difficult to know in advance.
    # Efficiency: Every backtracking step requires evaluating the cost function = solving PDE. But evaluating gradient involves solving PDE twice.

    # Exercise 14:
    # exercise_14()
    # BFGS is way better than gradient descent for this problem.

    # Exercise 15:
    # exercise_15()
    # Similar results

    # Exercise 16:
    alphas = np.logspace(-10, 10, 40)  # Regularization parameters from 10^-6 to 10^6
    # exercise_16_sweep(alphas)
    # exercise_16(alpha=1e-10, plot=True)
    # exercise_16(alpha=1e-5, plot=True)
    # exercise_16(alpha=1e7, plot=True)

    plt.show()
