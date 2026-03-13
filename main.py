import os
import torch
import math
import time
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="2D Steady State thermal Analysis Dashboard ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

progress_store = {"current": 0}

class CalculateRequest(BaseModel):
    m: int
    Tu: float; Td: float; Tl: float; Tr: float

def sor_solver(Tu, Td, Tl, Tr, n, tol=1e-3, max_iter=10000):
    global progress_store
    progress_store["current"] = 0
    last_progress = 0

    start_time = time.perf_counter()

    T = torch.full((n, n), (Tu+Td+Tl+Tr)/4.0, dtype=torch.float32)

    omega = 2.0 / (1.0 + math.sin(math.pi / (n + 1)))

    I, J = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    red_mask = (I + J) % 2 == 0
    black_mask = ~red_mask

    initial_diff = 0

    for it in range(max_iter):
        T_old = T.clone()

        for mask in [red_mask, black_mask]:

            north = torch.zeros_like(T); north[1:,:] = T[:-1,:]; north[0,:] = Tu
            south = torch.zeros_like(T); south[:-1,:] = T[1:,:]; south[-1,:] = Td
            west  = torch.zeros_like(T); west[:,1:] = T[:,:-1]; west[:,0] = Tl
            east  = torch.zeros_like(T); east[:,:-1] = T[:,1:]; east[:,-1] = Tr

            T_new = 0.25 * (north + south + west + east)

            T[mask] = (1 - omega) * T[mask] + omega * T_new[mask]

        diff = torch.max(torch.abs(T - T_old)).item()

        if it == 0:
            initial_diff = max(diff, 1e-9)

        if diff > tol:
            progress = int(
                max(0, min(99,
                100 * (1 - math.log(diff/tol) / math.log(initial_diff/tol))
                ))
            )

            # only increase progress
            if progress > last_progress:
                last_progress = progress
                progress_store["current"] = progress

        else:
            break

    progress_store["current"] = 100

    return T, time.perf_counter() - start_time, it + 1
@app.get("/api/health")
def health(): return {"status": "online"}

@app.get("/api/progress")
def get_progress(): return {"progress": progress_store["current"]}

@app.post("/api/calculate")
def calculate(req: CalculateRequest):
    temp_core, solve_time, iters = sor_solver(req.Tu, req.Td, req.Tl, req.Tr, req.m-1)
    fdm = np.zeros((req.m+1, req.m+1))
    fdm[1:-1, 1:-1] = temp_core.numpy()
    fdm[0,:]=req.Tu; fdm[-1,:]=req.Td; fdm[:,0]=req.Tl; fdm[:,-1]=req.Tr
    
    # Correct orientation: Array indexing is y=0 at top. 
    # Flip to visual indexing (y=0 at base) for comparison with Fourier and plotting.
    fdm = np.flipud(fdm)

    x, y = np.meshgrid(np.linspace(0,1,req.m+1), np.linspace(0,1,req.m+1))
    ana = np.zeros_like(fdm)
    for n in range(1, 151, 2):
        npi = n*np.pi; s_npi = np.sinh(npi)
        if np.isinf(s_npi): break
        c = 4.0/(npi*s_npi)
        ana += c*(req.Tu*np.sinh(npi*y)*np.sin(npi*x) + req.Td*np.sinh(npi*(1-y))*np.sin(npi*x) + req.Tl*np.sinh(npi*(1-x))*np.sin(npi*y) + req.Tr*np.sinh(npi*x)*np.sin(npi*y))

    ana[0,:]=req.Td; ana[-1,:]=req.Tu; ana[:,0]=req.Tl; ana[:,-1]=req.Tr
    
    internal_fdm = fdm[1:-1, 1:-1]
    internal_ana = ana[1:-1, 1:-1]
    rmse = np.sqrt(np.mean((internal_fdm - internal_ana)**2))
    v_score = max(0, 100 * (1 - (rmse / max(abs(req.Tu-req.Td),1.0))))
    
    try:
        supabase.table("thermal_logs").insert({"grid_size":req.m, "t_top":req.Td, "t_bottom":req.Tu, "t_left":req.Tl, "t_right":req.Tr, "iterations":iters, "time_taken":solve_time, "validation_score":float(v_score)}).execute()
    except Exception as e: print(f"DB Error: {e}")
    
    return {"fdm": fdm.tolist(), "analytic": ana.tolist(), "time": solve_time, "iters": iters}

@app.get("/api/regression")
def get_raw_data():
    try:
        res = supabase.table("thermal_logs").select("grid_size", "iterations", "time_taken").execute()
        if not res.data:
            return {"error": "no_data"}
        
        # Sort data by grid size for better visual flow in the scatter plot
        df = pd.DataFrame(res.data).sort_values("grid_size")
        
        return {
            "m_values": df["grid_size"].tolist(),
            "iter_values": df["iterations"].tolist(),
            "time_values": df["time_taken"].tolist()
        }
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        return {"error": "db_error"}