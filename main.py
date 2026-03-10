from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import math
import time
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

app = FastAPI(title="High-Performance Thermal Analysis Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

progress_store = {"current": 0}

class CalculateRequest(BaseModel):
    m: int
    Tu: float 
    Td: float 
    Tl: float 
    Tr: float 

def red_black_sor_solver(Tu, Td, Tl, Tr, n, tol=1e-3, max_iter=10000):
    global progress_store
    progress_store["current"] = 0
    start_time = time.perf_counter()
    avg_temp = (Tu + Td + Tl + Tr) / 4.0
    T = torch.full((n, n), avg_temp, dtype=torch.float32)
    omega = 2.0 / (1.0 + math.sin(math.pi / (n + 1)))
    I, J = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    red_mask = (I + J) % 2 == 0
    black_mask = ~red_mask
    initial_diff = 0
    for iter_count in range(max_iter):
        T_old = T.clone()
        for mask in [red_mask, black_mask]:
            north = torch.zeros_like(T); north[1:, :] = T[:-1, :]; north[0, :] = Tu
            south = torch.zeros_like(T); south[:-1, :] = T[1:, :]; south[-1, :] = Td
            west = torch.zeros_like(T); west[:, 1:] = T[:, :-1]; west[:, 0] = Tl
            east = torch.zeros_like(T); east[:, :-1] = T[:, 1:]; east[:, -1] = Tr
            T_new = 0.25 * (north + south + west + east)
            T[mask] = (1 - omega) * T[mask] + omega * T_new[mask]
        current_diff = torch.max(torch.abs(T - T_old)).item()
        if iter_count == 0: initial_diff = max(current_diff, 1e-9)
        if current_diff > tol:
            conv_progress = int(100 * (1 - math.log(current_diff/tol) / math.log(initial_diff/tol)))
            progress_store["current"] = max(0, min(99, conv_progress))
        else: break
    progress_store["current"] = 100
    return T, time.perf_counter() - start_time, iter_count + 1

@app.get("/api/progress")
def get_progress():
    return {"progress": progress_store["current"]}

@app.post("/api/calculate")
def calculate(req: CalculateRequest):
    temp_core, solve_time, iters = red_black_sor_solver(req.Tu, req.Td, req.Tl, req.Tr, req.m-1)
    fdm_full = np.zeros((req.m + 1, req.m + 1))
    fdm_full[1:-1, 1:-1] = temp_core.numpy()
    fdm_full[0, :] = req.Tu; fdm_full[-1, :] = req.Td
    fdm_full[:, 0] = req.Tl; fdm_full[:, -1] = req.Tr
    
    x_range = np.linspace(0, 1, req.m + 1)
    y_range = np.linspace(0, 1, req.m + 1)
    X, Y = np.meshgrid(x_range, y_range)
    analytic = np.zeros((req.m + 1, req.m + 1))
    for n_term in range(1, 151, 2):
        npi = n_term * np.pi
        sinh_npi = np.sinh(npi)
        if np.isinf(sinh_npi): break
        coeff = 4.0 / (npi * sinh_npi)
        analytic += coeff * (
            req.Tu * np.sinh(npi * Y) * np.sin(npi * X) +
            req.Td * np.sinh(npi * (1 - Y)) * np.sin(npi * X) +
            req.Tl * np.sinh(npi * (1 - X)) * np.sin(npi * Y) +
            req.Tr * np.sinh(npi * X) * np.sin(npi * Y)
        )
    analytic = np.flipud(analytic)
    analytic[0,:]=req.Tu; analytic[-1,:]=req.Td; analytic[:,0]=req.Tl; analytic[:,-1]=req.Tr
    
    internal_fdm = fdm_full[1:-1, 1:-1]
    internal_ana = analytic[1:-1, 1:-1]
    if internal_fdm.size > 0:
        rmse = np.sqrt(np.mean((internal_fdm - internal_ana)**2))
        t_range = max(abs(req.Tu - req.Td), abs(req.Tl - req.Tr), 1.0)
        validation_index = max(0, 100 * (1 - (rmse / t_range)))
    else: validation_index = 100.0

    # DATA LOGGER: Storing boundaries, grid, and performance
    file_name = "solver_runs.csv"
    new_entry = pd.DataFrame([{
        "grid_size": req.m, 
        "T_top": req.Td, "T_bottom": req.Tu, "T_left": req.Tl, "T_right": req.Tr,
        "iterations": iters, 
        "time_taken": solve_time
    }])
    if os.path.exists(file_name):
        pd.concat([pd.read_csv(file_name), new_entry], ignore_index=True).to_csv(file_name, index=False)
    else:
        new_entry.to_csv(file_name, index=False)
    
    return {"fdm": fdm_full.tolist(), "analytic": analytic.tolist(), "time": solve_time, "iters": iters, "validation_score": float(validation_index)}

@app.get("/api/regression")
def get_regression():
    if not os.path.exists("solver_runs.csv"): return {"error": "No data"}
    df = pd.read_csv("solver_runs.csv").sort_values("grid_size").drop_duplicates("grid_size")
    if len(df) < 3: return {"error": "Insufficient dataset"}
    X = df[["grid_size"]].values
    reg_i = LinearRegression().fit(X, df["iterations"].values)
    poly = PolynomialFeatures(degree=3)
    X_p = poly.fit_transform(X)
    reg_t = LinearRegression().fit(X_p, df["time_taken"].values)
    x_curve = np.linspace(X.min(), X.max()*1.2, 50).reshape(-1, 1)
    return {
        "x_range": x_curve.flatten().tolist(),
        "y_iter_curve": reg_i.predict(x_curve).tolist(),
        "y_time_curve": reg_t.predict(poly.transform(x_curve)).tolist(),
        "coeffs": {
            "iter_m": float(reg_i.coef_[0]), "iter_c": float(reg_i.intercept_),
            "time_coeffs": reg_t.coef_.tolist(), "time_c": float(reg_t.intercept_)
        }
    }