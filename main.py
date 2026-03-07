from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import math
import time
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

app = FastAPI(title="2D Thermal Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CalculateRequest(BaseModel):
    m: int
    Tu: float # Top
    Td: float # Bottom
    Tl: float # Left
    Tr: float # Right

def red_black_sor_solver(Tu, Td, Tl, Tr, n, tol=1e-3, max_iter=20000):
    """
    SOR Solver with increased iteration limit for high-resolution grids.
    m > 200 requires more passes to satisfy the tolerance threshold.
    """
    start_time = time.perf_counter()
    # Initialize T with an average of boundary conditions for faster convergence
    avg_temp = (Tu + Td + Tl + Tr) / 4.0
    T = torch.full((n, n), avg_temp, dtype=torch.float32)
    
    # Theoretical optimal omega for square grid
    omega = 2.0 / (1.0 + math.sin(math.pi / (n + 1)))
    
    I, J = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    red_mask = (I + J) % 2 == 0
    black_mask = ~red_mask
    
    for iter_count in range(max_iter):
        T_old = T.clone()
        for mask in [red_mask, black_mask]:
            # Use 5-point stencil with boundary padding
            north = torch.zeros_like(T); north[1:, :] = T[:-1, :]; north[0, :] = Tu
            south = torch.zeros_like(T); south[:-1, :] = T[1:, :]; south[-1, :] = Td
            west = torch.zeros_like(T); west[:, 1:] = T[:, :-1]; west[:, 0] = Tl
            east = torch.zeros_like(T); east[:, :-1] = T[:, 1:]; east[:, -1] = Tr
            
            T_new = 0.25 * (north + south + west + east)
            T[mask] = (1 - omega) * T[mask] + omega * T_new[mask]
        
        # Convergence check: stop if the maximum change is less than tolerance
        if torch.max(torch.abs(T - T_old)) < tol:
            break
            
    return T, time.perf_counter() - start_time, iter_count + 1

def get_analytical_matrix(m, Tu, Td, Tl, Tr):
    """Calculates the Fourier Series solution for comparison."""
    x_vals = np.linspace(0, m, m + 1)
    y_vals = np.linspace(0, m, m + 1)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid = np.zeros((m + 1, m + 1))
    terms = 100 
    k = float(m)
    for n in range(1, 2 * terms, 2):
        n_pi_k = (n * np.pi) / k
        sinh_npi = np.sinh(n * np.pi)
        if np.isinf(sinh_npi): break
        coeff = 4.0 / (n * np.pi * sinh_npi)
        # Td at Y=0 (Bottom), Tu at Y=k (Top)
        btm_top = (Tu * np.sinh(n_pi_k * Y) + Td * np.sinh(n_pi_k * (k - Y))) * np.sin(n_pi_k * X)
        lft_rgt = (Tl * np.sinh(n_pi_k * (k - X)) + Tr * np.sinh(n_pi_k * X)) * np.sin(n_pi_k * Y)
        grid += coeff * (btm_top + lft_rgt)
    
    grid = np.flipud(grid) 
    grid[0, :] = Tu; grid[-1, :] = Td
    grid[:, 0] = Tl; grid[:, -1] = Tr
    return grid

@app.post("/api/calculate")
def calculate(req: CalculateRequest):
    # Solve inner nodes (m-1 size)
    temp_core, solve_time, iters = red_black_sor_solver(req.Tu, req.Td, req.Tl, req.Tr, req.m-1)
    
    # Construct full matrix with boundaries
    fdm_full = np.zeros((req.m + 1, req.m + 1))
    fdm_full[1:-1, 1:-1] = temp_core.numpy()
    fdm_full[0, :] = req.Tu; fdm_full[-1, :] = req.Td
    fdm_full[:, 0] = req.Tl; fdm_full[:, -1] = req.Tr
    
    analytic_full = get_analytical_matrix(req.m, req.Tu, req.Td, req.Tl, req.Tr)
    
    # Calculate Accuracy
    l2_diff = np.sqrt(np.mean((fdm_full - analytic_full)**2))
    max_val = max(np.max(np.abs(analytic_full)), 1.0)
    accuracy = max(0, 100 * (1 - (l2_diff / max_val)))

    # Log to CSV for regression analysis
    file_name = "solver_runs.csv"
    new_entry = pd.DataFrame([{"grid_size": req.m, "iterations": iters, "time_taken": solve_time}])
    if os.path.exists(file_name):
        pd.concat([pd.read_csv(file_name), new_entry], ignore_index=True).to_csv(file_name, index=False)
    else:
        new_entry.to_csv(file_name, index=False)

    return {
        "fdm": fdm_full.tolist(), "analytic": analytic_full.tolist(), 
        "time": solve_time, "iters": iters, "accuracy": float(accuracy)
    }

@app.get("/api/regression")
def get_regression():
    file_name = "solver_runs.csv"
    if not os.path.exists(file_name): return {"error": "No data"}
    df = pd.read_csv(file_name)
    
    # Remove outliers for a cleaner curve
    if len(df) > 5:
        Q1, Q3 = df['time_taken'].quantile(0.25), df['time_taken'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['time_taken'] < (Q1 - 1.5 * IQR)) | (df['time_taken'] > (Q3 + 1.5 * IQR)))]
    
    df = df.sort_values("grid_size").drop_duplicates("grid_size")
    if len(df) < 3: return {"error": "Need more data points"}
    
    X = df[["grid_size"]].values
    y_t = df["time_taken"].values
    X_poly = PolynomialFeatures(degree=3).fit_transform(X)
    reg_t = LinearRegression().fit(X_poly, y_t)
    
    y_i = df["iterations"].values
    reg_i = LinearRegression().fit(X, y_i)
    
    return {
        "grid_size": df["grid_size"].tolist(),
        "time_taken": y_t.tolist(), "line_time": reg_t.predict(X_poly).tolist(), "r2_time": float(r2_score(y_t, reg_t.predict(X_poly))),
        "iterations": y_i.tolist(), "line_iters": reg_i.predict(X).tolist(), "r2_iters": float(r2_score(y_i, reg_i.predict(X)))
    }