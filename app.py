import streamlit as st
import torch
import time
import math
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("2-D Temperature Analysis Using Finite Difference Method")

# -------------------- GRID SIZE --------------------
m = st.number_input("Enter Number of Grids",
                    min_value=3,
                    step=1,
                    format="%d")

# -------------------- SOLVER --------------------
def red_black_sor_2d_vectorized(Tu, Td, Tl, Tr, n, tol=1e-6, max_iter=5000):

    start_time = time.perf_counter()
    T = torch.zeros((n, n), dtype=torch.float32)
    omega = 2.0 / (1.0 + math.sin(math.pi / n))

    I, J = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    red = (I + J) % 2 == 0
    black = ~red

    for k in range(max_iter):
        T_old = T.clone()

        north = torch.zeros_like(T)
        south = torch.zeros_like(T)
        west  = torch.zeros_like(T)
        east  = torch.zeros_like(T)

        north[1:, :] = T[:-1, :]
        north[0, :] = Tu
        south[:-1, :] = T[1:, :]
        south[-1, :] = Td
        west[:, 1:] = T[:, :-1]
        west[:, 0] = Tl
        east[:, :-1] = T[:, 1:]
        east[:, -1] = Tr

        T_new = 0.25 * (north + south + west + east)
        T[red] = (1 - omega) * T[red] + omega * T_new[red]

        north[1:, :] = T[:-1, :]
        south[:-1, :] = T[1:, :]
        west[:, 1:] = T[:, :-1]
        east[:, :-1] = T[:, 1:]

        T_new = 0.25 * (north + south + west + east)
        T[black] = (1 - omega) * T[black] + omega * T_new[black]

        if torch.max(torch.abs(T - T_old)) < tol:
            break

    elapsed_time = time.perf_counter() - start_time
    return T, elapsed_time, k + 1


# -------------------- LOG DATA --------------------
def log_run_data(grid_size, iterations, time_taken):
    file_name = "solver_runs.csv"

    new_data = pd.DataFrame([{
        "grid_size": grid_size,
        "iterations": iterations,
        "time_taken": time_taken
    }])

    if os.path.exists(file_name):
        old_data = pd.read_csv(file_name)
        updated = pd.concat([old_data, new_data], ignore_index=True)
    else:
        updated = new_data

    updated.to_csv(file_name, index=False)


# -------------------- PERFORMANCE ANALYSIS --------------------
def plot_performance_analysis():
    file_name = "solver_runs.csv"

    if not os.path.exists(file_name):
        return

    df = pd.read_csv(file_name)
    df = df.sort_values("grid_size")
    if len(df) < 3:
        st.warning("Run solver with at least 3 different grid sizes.")
        return

    # -------- Iterations vs n (Expected Linear) --------
    X_iter = df[["grid_size"]]
    y_iter = df["iterations"]

    model_iter = LinearRegression()
    model_iter.fit(X_iter, y_iter)

    df["iter_pred"] = model_iter.predict(X_iter)
    r2_iter = r2_score(y_iter, df["iter_pred"])

    fig_iter = go.Figure()
    fig_iter.add_trace(go.Scatter(
        x=df["grid_size"],
        y=y_iter,
        mode="markers",
        name="Actual"
    ))
    fig_iter.add_trace(go.Scatter(
        x=df["grid_size"],
        y=df["iter_pred"],
        mode="lines",
        name="Fit (O(n))"
    ))

    fig_iter.update_layout(
        title=f"Iterations vs Grid Size  |  R² = {r2_iter:.4f}",
        xaxis_title="Grid Size (n)",
        yaxis_title="Iterations",
        height=550
    )

    st.plotly_chart(fig_iter, use_container_width=True)

    # -------- Time vs n³ (Expected Cubic Scaling) --------
    df["n_cubed"] = df["grid_size"] ** 3
    X_time = df[["n_cubed"]]
    y_time = df["time_taken"]

    model_time = LinearRegression()
    model_time.fit(X_time, y_time)

    df["time_pred"] = model_time.predict(X_time)
    r2_time = r2_score(y_time, df["time_pred"])

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=df["grid_size"],
        y=y_time,
        mode="markers",
        name="Actual"
    ))
    fig_time.add_trace(go.Scatter(
        x=df["grid_size"],
        y=df["time_pred"],
        mode="lines",
        name="Fit (O(n³))"
    ))

    fig_time.update_layout(
        title=f"Time vs Grid Size (Cubic Scaling)  |  R² = {r2_time:.4f}",
        xaxis_title="Grid Size (n)",
        yaxis_title="Time (seconds)",
        height=550
    )

    st.plotly_chart(fig_time, use_container_width=True)


# -------------------- HEATMAP --------------------
def plot_temperature(Temp, Tu, Td, Tl, Tr):

    m_internal = Temp.shape[0] + 1
    Tfull = torch.zeros((m_internal + 1, m_internal + 1))
    Tfull[1:-1, 1:-1] = Temp

    Tfull[0, :] = Tu
    Tfull[-1, :] = Td
    Tfull[:, 0] = Tl
    Tfull[:, -1] = Tr

    T_np = Tfull.numpy()

    fig = px.imshow(
        T_np,
        color_continuous_scale="inferno",
        origin="lower",
        aspect="equal"
    )

    fig.update_traces(
        hovertemplate="X: %{x}<br>Y: %{y}<br>Temperature: %{z:.2f} K<extra></extra>"
    )

    fig.update_layout(height=750)

    return fig


# -------------------- BOUNDARY INPUT --------------------
st.subheader("Boundary Conditions")

col1, col2 = st.columns(2)

with col1:
    Td = st.number_input("Top Temperature (K)", min_value=0, step=10)
    Tu = st.number_input("Bottom Temperature (K)", min_value=0, step=10)

with col2:
    Tl = st.number_input("Left Temperature (K)", min_value=0, step=10)
    Tr = st.number_input("Right Temperature (K)", min_value=0, step=10)

# -------------------- CALCULATE --------------------
if st.button("Calculate Temperature", type="primary"):

    Temp, solve_time, iters = red_black_sor_2d_vectorized(
        Tu=Td,
        Td=Tu,
        Tl=Tl,
        Tr=Tr,
        n=m-1,
        tol=0.01
    )

    st.session_state["Temp"] = Temp
    st.session_state["time"] = solve_time
    st.session_state["iters"] = iters
    st.session_state["Tu"] = Tu
    st.session_state["Td"] = Td
    st.session_state["Tl"] = Tl
    st.session_state["Tr"] = Tr

    log_run_data(m, iters, solve_time)

# -------------------- DISPLAY --------------------
if "Temp" in st.session_state:

    st.success("Calculation complete!")

    st.metric("Solver Time (s)", f"{st.session_state['time']:.4f}")
    st.metric("Iterations", st.session_state["iters"])

    fig = plot_temperature(
        st.session_state["Temp"],
        st.session_state["Tu"],
        st.session_state["Td"],
        st.session_state["Tl"],
        st.session_state["Tr"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Solver Performance Analysis")
    plot_performance_analysis()