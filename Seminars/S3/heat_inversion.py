from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import poisson_discretization_DpDm as pd
import scipy.sparse as sp
import setup_heat as setup
from sympy import diff, exp, lambdify, pi, sin, symbols


def solver(
    b,
    setup_config,
):
    # Load problem setup
    grid = setup_config["grid"]
    order = setup_config["order"]
    bc_opts = setup_config["bc_opts"]
    y0 = setup_config["y0"]
    u = setup_config["source_fun"]
    Tend = setup_config["Tend"]
    dt = setup_config["dt"]

    # Matrices
    ops = pd.assemble_matrices(grid, order, b, bc_opts)
    D = ops["Laplace_with_BC"]
    RHS = sp.eye(D.shape[0], format="csc") - dt * D  # RHS matrix for implicit Euler
    LU = sp.linalg.splu(RHS)  # LU factorization for efficient solves

    # Time-stepping loop
    Y = [y0]
    T = [0.0]
    for n in range(int(Tend / dt)):
        t_next = T[-1] + dt
        y_prev = Y[-1]
        LHS = y_prev + dt * u(t_next)  # LHS vector for implicit Euler

        y_next = LU.solve(LHS)
        Y.append(y_next)
        T.append(t_next)

    return np.array(Y), np.array(T)


def plot_movie(Y, T, setup_config, plot_config):
    """Animate and optionally save a sequence of heat-equation solutions."""
    if not plot_config.get("plot_movie"):
        return

    grid = setup_config["grid"]
    Tend = setup_config["Tend"]
    X1, X2 = grid["X"], grid["Y"]
    Y_reshaped = [y.reshape(X1.shape) for y in Y]
    plot_interval = max(1, int(plot_config.get("plot_interval", 1)))
    frame_indices = list(range(0, len(T), plot_interval))
    if frame_indices[-1] != len(T) - 1:
        frame_indices.append(len(T) - 1)
    min_y = min(y.min() for y in Y_reshaped)
    max_y = max(y.max() for y in Y_reshaped)

    fig, ax = plt.subplots()
    first_frame = frame_indices[0]
    im = ax.pcolormesh(
        X1,
        X2,
        Y_reshaped[first_frame],
        shading="auto",
        vmin=min_y,
        vmax=max_y,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    title = ax.set_title(f"Time: {T[first_frame]:.2f}/{Tend:.2f}")
    suptitle = plot_config.get("suptitle", "Heat Equation Solution Over Time $y(x,t)$")
    fig.suptitle(suptitle)
    plt.colorbar(im, ax=ax)

    def update(frame):
        time_index = frame_indices[frame]
        im.set_array(Y_reshaped[time_index].ravel())
        title.set_text(f"Time: {T[time_index]:.2f}/{Tend:.2f}")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    if plot_config.get("save_movie"):
        figures_dir = Path(__file__).resolve().parent / "Figures"
        figures_dir.mkdir(exist_ok=True)
        movie_filename = figures_dir / plot_config.get(
            "movie_filename", "heat_inversion.mp4"
        )
        movie_writer = plot_config.get("movie_writer", "ffmpeg")
        ani.save(movie_filename, writer=movie_writer)


def exercise1():
    # Solve the heat equation
    setup_config = setup.synthetic_inversion()
    b = setup_config["b_true"]
    Y, T = solver(b, setup_config)

    # Plot the solution as a movie
    plot_config = {
        "plot_movie": True,
        "save_movie": True,
        "movie_filename": "exercise1_heat_inversion.mp4",
        "movie_writer": "ffmpeg",
        "plot_interval": 1,
        "suptitle": "Exercise 1: Heat Inversion",
    }
    plot_movie(Y, T, setup_config, plot_config)


def exercise2():
    # MMS
    t, x, y = symbols("t x y")
    sol = sin(pi * x) * sin(2 * pi * y) * exp(-t)
    dYdt = diff(sol, t)
    Y_xx = diff(sol, x, 2)
    Y_yy = diff(sol, y, 2)
    source = dYdt - (Y_xx + Y_yy)
    source_func = lambdify((x, y, t), source, "numpy")
    sol_func = lambdify((x, y, t), sol, "numpy")

    # Solve
    plot_config = {
        "plot_movie": True,
        "save_movie": True,
        "movie_filename": "exercise2_heat_inversion_correctness.mp4",
        "plot_interval": 5,
        "suptitle": "Exercise 2: Method of Manufactured Solutions (MMS)",
    }
    setup_config = setup.synthetic_inversion()
    Xv = setup_config["grid"]["Xv"]
    Yv = setup_config["grid"]["Yv"]
    b = np.ones_like(setup_config["b_initial"])
    setup_config["y0"] = sol_func(Xv, Yv, 0.0)
    setup_config["source_fun"] = lambda time: source_func(Xv, Yv, time)
    setup_config["bc_opts"] = {"type": "dirichlet"}

    Y, T = solver(b, setup_config)
    plot_movie(Y, T, setup_config, plot_config)

    # Compare numerical solution with exact solution
    X1, X2 = setup_config["grid"]["X"], setup_config["grid"]["Y"]
    Y_exact = [sol_func(X1, X2, t) for t in T]
    Y_numerical = [Y[i].reshape(X1.shape) for i in range(len(T))]
    # Calculate the error between the numerical and exact solutions
    errors = [np.linalg.norm(Y_numerical[i] - Y_exact[i]) for i in range(len(T))]

    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    ax[0].plot(T, errors)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Error (L2 norm)")
    ax[0].set_title("Error between numerical and exact solutions over time")
    ax[0].grid()

    # Plot the exact solution at the final time
    im1 = ax[1].pcolormesh(X1, X2, Y_exact[-1], shading="auto")
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")
    ax[1].set_title("Exact solution at final time")
    plt.colorbar(im1, ax=ax[1])

    # Plot the numerical solution at the final time
    im2 = ax[2].pcolormesh(X1, X2, Y_numerical[-1], shading="auto")
    ax[2].set_xlabel("$x_1$")
    ax[2].set_ylabel("$x_2$")
    ax[2].set_title("Numerical solution at final time")
    plt.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    plt.savefig(
        "./Seminars/S3/Figures/exercise2_heat_inversion_correctness.png", dpi=600
    )


def adjoint_solver(b, Y, Y_d, setup_config):
    # Load problem setup
    grid = setup_config["grid"]
    order = setup_config["order"]
    bc_opts = setup_config["bc_opts"]
    Tend = setup_config["Tend"]
    dt = setup_config["dt"]

    # Matrices
    ops = pd.assemble_matrices(grid, order, b, bc_opts)
    D = ops["Laplace_with_BC"]
    RHS = sp.eye(D.shape[0], format="csc") - dt * D  # RHS matrix for implicit Euler
    LU = sp.linalg.splu(RHS)  # LU factorization for efficient solves

    # Time-stepping loop
    Y_dagger = [np.zeros_like(Y[-1])]  # Initialize with zeros for the adjoint variable
    T_dagger = [0.0]
    for n in range(int(Tend / dt)):
        t_next = T_dagger[-1] + dt

        inverse_time_index = (
            int(Tend / dt) - n - 1
        )  # Index for the corresponding time step in the forward solution
        LHS = Y_dagger[-1] + dt * (
            Y[inverse_time_index] - Y_d[inverse_time_index]
        )  # LHS vector for implicit Euler

        y_next = LU.solve(LHS)
        Y_dagger.append(y_next)
        T_dagger.append(t_next)

    # Reverse the lists to match the forward time direction
    Y_dagger.reverse()
    T_dagger.reverse()

    return np.array(Y_dagger), np.array(T_dagger)


def exercise3():
    plot_config = {
        "plot_movie": True,
        "save_movie": True,
        "movie_filename": "exercise3_heat_inversion_adjoint.mp4",
        "suptitle": "Exercise 3: Adjoint solution for Heat Inversion",
        "plot_interval": 3,
    }
    setup_config = setup.synthetic_inversion()
    b = setup_config["b_true"]
    Y_d = solver(b, setup_config)[0]  # Solve with true b to get synthetic data
    b_initial = setup_config["b_initial"]
    Y, T = solver(b_initial, setup_config)
    # plot_movie(Y, T, setup_config, plot_config)

    Y_dagger, T_dagger = adjoint_solver(b_initial, Y, Y_d, setup_config)
    plot_movie(Y_dagger, T_dagger, setup_config, plot_config)

    print(f"Shape of Y_dagger: {Y_dagger.shape}, Shape of T_dagger: {T_dagger.shape}")
    print(f"Shape of Y: {Y.shape}, Shape of T: {T.shape}")


def integration_weights(N_timesteps, method="trapezoidal"):
    N = N_timesteps
    if method == "trapezoidal":
        w_k = np.ones(N)
        w_k[0] = 0.5
        w_k[-1] = 0.5
    elif method == "midpoint":
        w_k = np.ones(N)
        w_k[0] = 0.0
        w_k[-1] = 0.0
    else:
        raise ValueError(
            f"Unknown integration method: {method}. Use 'trapezoidal' or 'midpoint'."
        )
    return w_k


def adjoint_gradient(b, Y, Y_d, setup_config, intergration_method="trapezoidal"):
    # Solve the adjoint problem
    Y, T = solver(b, setup_config)
    Y_dagger, T_dagger = adjoint_solver(b, Y, Y_d, setup_config)

    ops = pd.assemble_matrices(
        setup_config["grid"], setup_config["order"], b, setup_config["bc_opts"]
    )
    H = ops["H"]
    Dx_m = ops["Dx_m"]
    Dy_m = ops["Dy_m"]
    dt = setup_config["dt"]
    w_k = integration_weights(len(T), method=intergration_method)

    sum_term = np.zeros_like(b)
    for k in range(len(T)):
        y_k = Y[k]
        y_dagger_k = Y_dagger[k]
        # Reverse time index for adjoint solution
        sum_term -= (
            w_k[k]
            * dt
            * ((Dx_m @ y_dagger_k) * (Dx_m @ y_k) + (Dy_m @ y_dagger_k) * (Dy_m @ y_k))
        )

    return H @ sum_term  # Return the gradient with respect to b


def cost_function(Y, Y_d, H, setup_config, intergration_method="trapezoidal"):
    w_k = integration_weights(len(Y), method=intergration_method)
    dt = setup_config["dt"]
    cost = 0.0
    for k in range(len(w_k)):
        residual = Y[k] - Y_d[k]  # TODO: Check if problemset is correct?
        cost += w_k[k] * (residual.T @ H @ residual)
    return 0.5 * dt * cost  # Return the cost function value


def FD_gradient(b, Y_d, setup_config, delta=1e-6, intergration_method="trapezoidal"):
    # Compute the gradient using finite differences
    ops = pd.assemble_matrices(
        setup_config["grid"], setup_config["order"], b, setup_config["bc_opts"]
    )
    H = ops["H"]
    cost = lambda Y: cost_function(
        Y, Y_d, H, setup_config, intergration_method=intergration_method
    )

    grad = np.zeros_like(b)
    Y = solver(b, setup_config)[0]  # Solve with current b to get Y
    for i in range(len(b)):
        print(f"Computing FD gradient for b[{i}] / {len(b)}...", end="\r")
        b_pertubed = b.copy()
        b_pertubed[i] += delta
        Y_pertubed, _ = solver(b_pertubed, setup_config)
        grad[i] = (cost(Y_pertubed) - cost(Y)) / delta
    return grad


def exercise4():
    setup_config = setup.synthetic_inversion()
    b = setup_config["b_true"]
    Y_d = solver(b, setup_config)[0]  # Solve with true b to get synthetic data
    b_initial = setup_config["b_initial"]
    Y, T = solver(b_initial, setup_config)
    # Compute the gradient of the objective function w.r.t. b using the adjoint method
    gradient_adjoint = adjoint_gradient(b_initial, Y, Y_d, setup_config)
    # Compute the gradient of the objective function w.r.t. b using finite differences
    gradient_fd = FD_gradient(b_initial, Y_d, setup_config)

    ops = pd.assemble_matrices(
        setup_config["grid"], setup_config["order"], b_initial, setup_config["bc_opts"]
    )
    H = ops["H"]
    diff = gradient_adjoint - gradient_fd
    error = np.sqrt(diff @ H @ diff)
    print(
        f"L2-Error between adjoin and FD gradients (trapezoidal integration): {error:.6e}"
    )

    # Plot the gradient
    X1, X2 = setup_config["grid"]["X"], setup_config["grid"]["Y"]
    G_adjoint = gradient_adjoint.reshape(X1.shape)
    G_fd = gradient_fd.reshape(X1.shape)

    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    im = ax[0].pcolormesh(X1, X2, G_adjoint, shading="auto")
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    ax[0].set_title("Gradient of the objective function w.r.t. $b$ (Adjoint)")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].pcolormesh(X1, X2, G_fd, shading="auto")
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")
    ax[1].set_title(
        "Gradient of the objective function w.r.t. $b$ (Finite Differences)"
    )
    plt.colorbar(im, ax=ax[1])

    im = ax[2].pcolormesh(X1, X2, np.abs(G_adjoint - G_fd), shading="auto")
    ax[2].set_xlabel("$x_1$")
    ax[2].set_ylabel("$x_2$")
    ax[2].set_title(f"Absolute Difference between gradients (L2-error: {error:.6e})")
    plt.colorbar(im, ax=ax[2])
    fig.suptitle("Exercise 4: Gradient Comparison (trapezoidal integration)")
    plt.tight_layout()
    plt.savefig("./Seminars/S3/Figures/exercise4_gradient_comparison.png", dpi=600)


def exercise5():
    setup_config = setup.synthetic_inversion()
    b = setup_config["b_true"]
    Y_d = solver(b, setup_config)[0]  # Solve with true b to get synthetic data
    b_initial = setup_config["b_initial"]
    Y, T = solver(b_initial, setup_config)
    # Compute the gradient of the objective function w.r.t. b using the adjoint method
    w_k_method = "midpoint"
    gradient_adjoint = adjoint_gradient(
        b_initial, Y, Y_d, setup_config, intergration_method=w_k_method
    )
    # Compute the gradient of the objective function w.r.t. b using finite differences
    gradient_fd = FD_gradient(
        b_initial, Y_d, setup_config, intergration_method=w_k_method
    )

    ops = pd.assemble_matrices(
        setup_config["grid"], setup_config["order"], b_initial, setup_config["bc_opts"]
    )
    H = ops["H"]
    diff = gradient_adjoint - gradient_fd
    error = np.sqrt(diff @ H @ diff)
    print(
        f"L2-Error between adjoin and FD gradients (midpoint integration): {error:.6e}"
    )

    # Plot the gradient
    X1, X2 = setup_config["grid"]["X"], setup_config["grid"]["Y"]
    G_adjoint = gradient_adjoint.reshape(X1.shape)
    G_fd = gradient_fd.reshape(X1.shape)

    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    im = ax[0].pcolormesh(X1, X2, G_adjoint, shading="auto")
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    ax[0].set_title("Gradient of the objective function w.r.t. $b$ (Adjoint)")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].pcolormesh(X1, X2, G_fd, shading="auto")
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")
    ax[1].set_title(
        "Gradient of the objective function w.r.t. $b$ (Finite Differences)"
    )
    plt.colorbar(im, ax=ax[1])

    im = ax[2].pcolormesh(X1, X2, np.abs(G_adjoint - G_fd), shading="auto")
    ax[2].set_xlabel("$x_1$")
    ax[2].set_ylabel("$x_2$")
    ax[2].set_title(f"Absolute Difference between gradients, (L2-error: {error:.6e})")
    plt.colorbar(im, ax=ax[2])

    fig.suptitle("Exercise 5: Gradient Comparison (midpoint integration)")
    plt.tight_layout()
    plt.savefig("./Seminars/S3/Figures/exercise5_gradient_comparison.png", dpi=600)


if __name__ == "__main__":
    # exercise1()

    # exercise2()

    # exercise3()

    # exercise4()

    exercise5()

    plt.show()
