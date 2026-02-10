import streamlit as st
import torch
import time
import matplotlib.pyplot as plt
import math

# -------------------- APP TITLE --------------------
st.title("2-D Temperature Analysis Using Finite Difference Method")

# -------------------- GRID SIZE --------------------
m = st.number_input(
    "Enter Number of Grids",
    min_value=3,
    step=1,
    format="%d"
)

# -------------------- INDEX MAPPING --------------------
def coordinate_to_index(m, i, j):
    return i * (m - 1) + j

# -------------------- SOLVER --------------------

def red_black_sor_2d_vectorized(Tu, Td, Tl, Tr, n, tol=1e-6, max_iter=5000):
    """
    Fully vectorized Red-Black SOR for 2D steady heat equation.

    Tu, Td, Tl, Tr : boundary temperatures
    n              : number of interior points per direction
    tol            : convergence tolerance
    max_iter       : maximum iterations

    Returns:
    T              : (n x n) temperature field
    elapsed_time   : computation time (seconds)
    iterations     : number of iterations used
    """

    start_time = time.perf_counter()

    # Interior temperature field
    T = torch.zeros((n, n), dtype=torch.float32)

    # Optimal relaxation factor
    omega = 2.0 / (1.0 + math.sin(math.pi / n))

    # Create checkerboard masks
    I, J = torch.meshgrid(
        torch.arange(n), torch.arange(n), indexing="ij"
    )
    red = (I + J) % 2 == 0
    black = ~red

    for k in range(max_iter):
        T_old = T.clone()

        # ---------- RED UPDATE ----------
        north = torch.zeros_like(T)
        south = torch.zeros_like(T)
        west  = torch.zeros_like(T)
        east  = torch.zeros_like(T)

        north[1:, :] = T[:-1, :]
        north[0, :] = Td

        south[:-1, :] = T[1:, :]
        south[-1, :] = Tu

        west[:, 1:] = T[:, :-1]
        west[:, 0] = Tl

        east[:, :-1] = T[:, 1:]
        east[:, -1] = Tr

        T_new = 0.25 * (north + south + west + east)
        T[red] = (1 - omega) * T[red] + omega * T_new[red]

        # ---------- BLACK UPDATE ----------
        north[1:, :] = T[:-1, :]
        south[:-1, :] = T[1:, :]
        west[:, 1:] = T[:, :-1]
        east[:, :-1] = T[:, 1:]

        T_new = 0.25 * (north + south + west + east)
        T[black] = (1 - omega) * T[black] + omega * T_new[black]

        # Convergence check
        if torch.max(torch.abs(T - T_old)) < tol:
            break

    elapsed_time = time.perf_counter() - start_time

    return T, elapsed_time, k + 1

def red_black_sor_2d(Tu, Td, Tl, Tr, n, tol=1e-3, max_iter=5000):

    start_time = time.perf_counter()   # ⏱ START TIMER

    T = torch.zeros((n, n), dtype=torch.float32)
    omega = 2.0 / (1.0 + math.sin(math.pi / n))

    for k in range(max_iter):
        T_old = T.clone()

        # RED nodes
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    north = Tu if i == 0 else T[i-1, j]
                    south = Td if i == n-1 else T[i+1, j]
                    west  = Tl if j == 0 else T[i, j-1]
                    east  = Tr if j == n-1 else T[i, j+1]

                    T_new = 0.25 * (north + south + west + east)
                    T[i, j] = (1 - omega) * T[i, j] + omega * T_new

        # BLACK nodes
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 1:
                    north = Tu if i == 0 else T[i-1, j]
                    south = Td if i == n-1 else T[i+1, j]
                    west  = Tl if j == 0 else T[i, j-1]
                    east  = Tr if j == n-1 else T[i, j+1]

                    T_new = 0.25 * (north + south + west + east)
                    T[i, j] = (1 - omega) * T[i, j] + omega * T_new

        if torch.max(torch.abs(T - T_old)) < tol:
            break

    end_time = time.perf_counter()     # ⏱ END TIMER
    elapsed_time = end_time - start_time

    return T, elapsed_time



def gauss_seidel_2d( Tu, Td, Tl, Tr, n, tol=1e-2, max_iter=5000):
    progress_holder = st.empty()
    progress = progress_holder.progress(0)
    """
    Solves A x = b for 2D Laplace equation using Gauss-Seidel
    without forming matrix A.

    Inputs:
    b        : torch tensor (n x n) RHS
    Tu, Td   : top and bottom boundary values
    Tl, Tr   : left and right boundary values
    n        : number of interior nodes per direction
    tol      : convergence tolerance
    max_iter : maximum iterations

    Output:
    T : torch tensor (n x n) temperature field
    """

    T = torch.zeros((n, n), dtype=torch.float32)

    for k in range(max_iter):
        T_old = T.clone()

        for i in range(n):
            for j in range(n):
                north = Tu if i == 0     else T[i-1, j]
                south = Td if i == n-1   else T[i+1, j]
                west  = Tl if j == 0     else T[i, j-1]
                east  = Tr if j == n-1   else T[i, j+1]

                T[i, j] = 0.25 * (north + south + west + east)
        percent = int((k) / (max_iter - 1) * 90)
        progress.progress(min(percent, 100))

        if torch.norm(T - T_old) < tol:
            break

    return T

def create_temp_2d(m, Tu, Td, Tl, Tr):
    progress_holder = st.empty()
    progress = progress_holder.progress(0)

    time.sleep(0.1)
    progress.progress(5)

    N = (m - 1) ** 2
    coef = torch.zeros((N, N))
    const = torch.zeros((N, 1))

    for i in range(m - 1):
        for j in range(m - 1):
            row = coordinate_to_index(m, i, j)
            coef[row, row] = -4

            if i > 0:
                coef[row, coordinate_to_index(m, i - 1, j)] += 1
            else:
                const[row] -= Tu

            if i < m - 2:
                coef[row, coordinate_to_index(m, i + 1, j)] += 1
            else:
                const[row] -= Td

            if j > 0:
                coef[row, coordinate_to_index(m, i, j - 1)] += 1
            else:
                const[row] -= Tl

            if j < m - 2:
                coef[row, coordinate_to_index(m, i, j + 1)] += 1
            else:
                const[row] -= Tr

        percent = int((i + 1) / (m - 1) * 90)
        progress.progress(percent)

    # Solve (NO inverse)
    result = torch.linalg.solve(coef, const)

    Temp = torch.zeros((m - 1, m - 1))
    for i in range(m - 1):
        for j in range(m - 1):
            Temp[i, j] = result[coordinate_to_index(m, i, j)]

    progress.progress(100)
    time.sleep(0.3)
    progress_holder.empty()

    return Temp

# -------------------- PLOT --------------------
def plot_temperature(Temp, Tu, Td, Tl, Tr, m):
    Tfull = torch.zeros((m + 1, m + 1))
    Tfull[1:-1, 1:-1] = Temp

    Tfull[0, :]  = Tu  # bottom
    Tfull[-1, :] = Td   # top
    Tfull[:, 0]  = Tl   # left
    Tfull[:, -1] = Tr   # right


    fig, ax = plt.subplots()
    im = ax.imshow(Tfull, cmap="inferno", origin="lower")
    fig.colorbar(im, ax=ax, label="Temperature (K)")
    ax.set_title("Temperature Distribution Heat Map")
    #ax.set_xlabel("x")
    #ax.set_ylabel("y")

    return fig

# -------------------- QUERY --------------------
def temperature_at_point(Temp, x, y):
    return Temp[x, y].item()

# -------------------- BOUNDARY CONDITIONS --------------------
st.subheader("Boundary Conditions")

col1,col2 = st.columns(2)
with col1:
    Td = st.number_input("Top Temperature (K)", min_value=0, step=10, format="%d")
    Tu = st.number_input("Bottom Temperature (K)", min_value=0, step=10, format="%d")
with col2:
    Tl = st.number_input("Left Temperature (K)", min_value=0, step=10, format="%d")
    Tr = st.number_input("Right Temperature (K)", min_value=0, step=10, format="%d")

# -------------------- CENTER BUTTON --------------------
_, mid, _ = st.columns([1, 3, 1])
with mid:
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        #calculate_1 = st.button("Calculate Temperature", type="primary")
        calculate_2 = st.button("Calculate Temperature", type="primary",key="efficient way")

# -------------------- RUN SOLVER (ONLY ON BUTTON) --------------------
#if calculate_1:
#   with st.spinner("Calculating temperature..."):
#      st.session_state["computed"] = True
#     st.session_state["params"] = (m, Tu, Td, Tl, Tr)
if calculate_2:
    with st.spinner("Calculating temperature..."):
        Temp, solve_time, iters = red_black_sor_2d_vectorized(
            Tu=Td,
            Td=Tu,
            Tl=Tl,
            Tr=Tr,
            n=m-1,
            tol=0.01,
            max_iter=5000
        )

        st.session_state["Temp"] = Temp
        st.session_state["time"] = solve_time
        st.session_state["iters"] = iters
        st.session_state["computed"] = True
        st.session_state["params"] = (m, Tu, Td, Tl, Tr)



# -------------------- DISPLAY RESULTS --------------------
if st.session_state.get("computed", False):
    Temp = st.session_state["Temp"]

    st.success("Calculation complete!")

    if "time" in st.session_state:
        #st.metric("Solver Time (s)", f"{st.session_state['time']:.4f}")
        #st.metric("Iterations", st.session_state["iters"])

        fig = plot_temperature(Temp, Tu, Td, Tl, Tr, m)
        st.pyplot(fig)
        plt.close(fig)


    st.subheader("Query Temperature at a Point")

    c1, c2 = st.columns(2)
    with c1:
        x = st.number_input("X Coordinate", 0, m - 2, step=1, key="x_query")
    with c2:
        y = st.number_input("Y Coordinate", 0, m - 2, step=1, key="y_query")

    # Submit button
    query_btn = st.button("Get Temperature at Point", key="query_temp_btn")

    # Show result ONLY after clicking button
    if query_btn:
        temp_val = temperature_at_point(Temp, x, y)
        st.success(f"Temperature at ({x}, {y}) = {temp_val:.2f} K")

