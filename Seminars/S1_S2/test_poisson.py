import numpy as np
import poisson_discretization_DpDm as pd
from sympy import diff, lambdify, pi, sin, symbols
import matplotlib.pyplot as plt


def example_u(x, y):
    """Example source function"""
    return pd.shifted_gaussian(x, y, 0, 0.5, 0.2) + np.exp(1 * x**2)


def example_b(x, y):
    # b=1 for the standard Poisson equation
    return 1.0 + 0 * x

    # Variable-coefficient Poisson equation
    # return 1.0 + 10*shifted_gaussian(x, y, 0.2, 0.5, 0.1)


def exercise2():
    """Demo"""

    # Domain boundaries
    lim_x = [-1, 1]
    lim_y = [0, 1]

    # Create grid
    mx = 61
    my = 31
    grid = pd.create_grid(mx, my, lim_x, lim_y)

    # Build SBP-SAT discretization,
    # solve Poisson equation, and plot.
    y, sbp_discr = pd.solve_and_plot(grid, example_u, pd.example_b, PLOT=True)

def _MMS_test():
    x, y = symbols("x y")

    # Define the manufactured solution
    sol = sin(pi * x) * sin(4 * pi * y)

    # Differentiate to get the source term
    u_xx = diff(sol, x, 2)
    u_yy = diff(sol, y, 2)
    u = u_xx + u_yy

    u_func = lambdify((x, y), u, "numpy")
    sol_func = lambdify((x, y), sol, "numpy")
    return u_func, sol_func

def exercise3():
    """Demo"""

    # Domain boundaries
    lim_x = [-1, 1]
    lim_y = [0, 1]

    # Create grid
    mx = 61
    my = 31
    grid = pd.create_grid(mx, my, lim_x, lim_y)

    # Build SBP-SAT discretization,
    # solve Poisson equation, and plot.
    u_func, sol_func = _MMS_test()
    plt.pcolormesh(grid["X"], grid["Y"], sol_func(grid["X"], grid["Y"]), shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exact solution")
    plt.colorbar(label="Exact solution")
    plt.tight_layout()
    y, sbp_discr = pd.solve_and_plot(grid, u_func, example_b, PLOT=True)

def solve_mms_dirichlet(mx, my, PLOT=True):
    # Define the manufactured solution
    x, y = symbols("x y")
    sol = sin(pi * x) * sin(4 * pi * y)

    # Differentiate to get the source term
    u_xx = diff(sol, x, 2)
    u_yy = diff(sol, y, 2)
    u = u_xx + u_yy

    u_func = lambdify((x, y), u, "numpy")
    sol_func = lambdify((x, y), sol, "numpy")

    # Solve the Poisson equation with Dirichlet boundary conditions
    lim_x = [-1, 1]
    lim_y = [0, 1]
    grid = pd.create_grid(mx, my, lim_x, lim_y)
    y, sbp_discr = pd.solve_and_plot(grid, u_func, example_b, PLOT=False)
    
    # Calculate the error between the numerical and exact solutions
    y = np.reshape(y, grid["X"].shape) # Match shape of grid
    actual_sol = sol_func(grid["X"], grid["Y"])
    error = y - actual_sol  
    e = error.flatten()
    H = sbp_discr["H"]
    l2_error = np.sqrt(e.T @ H @ e)
    print(f"L2 error: {l2_error}")
    
    if PLOT:    
        # Plotting the numerical solution, exact solution, and error
        fig, axes = plt.subplots(3, 1)
        im1 = axes[0].pcolormesh(grid["X"], grid["Y"], actual_sol, shading="auto")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Exact solution")
        fig.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].pcolormesh(grid["X"], grid["Y"], y, shading="auto")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Numerical solution")
        fig.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].pcolormesh(grid["X"], grid["Y"], error, shading="auto")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("Error")
        fig.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        # plt.show()
    
    
    
    
    
def exercise4():
    mx = 61
    my = 31

    for size in range(1,4):
        mx = mx * size
        my = my * size
        print(f"Grid size: {mx} x {my}")
        solve_mms_dirichlet(mx, my, PLOT=False)

    solve_mms_dirichlet(mx, my, PLOT=True)
    
    

if __name__ == "__main__":
    # exercise2()
    # exercise3()
    exercise4()
    plt.show()
