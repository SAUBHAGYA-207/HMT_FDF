import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

function App() {
  const [params, setParams] = useState({ m: 50, Tu: 100, Td: 0, Tl: 0, Tr: 0 });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [regData, setRegData] = useState(null);
  const [prediction, setPrediction] = useState({ iters: '--', time: '--' });

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  useEffect(() => { fetchRegression(); }, []);

  const fetchRegression = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/regression?t=${Date.now()}`);
      if (!res.data.error) setRegData(res.data);
    } catch (e) { console.error(e); }
  };

  const runSimulation = async () => {
    if (regData?.coeffs) {
      const { iter_m, iter_c, time_coeffs, time_c } = regData.coeffs;
      const m = params.m;
      setPrediction({
        iters: Math.round(iter_m * m + iter_c),
        time: (time_c + time_coeffs[1]*m + time_coeffs[2]*m**2 + time_coeffs[3]*m**3).toFixed(4)
      });
    }

    setLoading(true); setProgress(0); setResults(null);
    const poll = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/progress`);
        setProgress(res.data.progress);
      } catch (e) { console.error(e); }
    }, 250);

    try {
      const res = await axios.post(`${API_BASE_URL}/api/calculate`, params);
      setResults(res.data);
      clearInterval(poll); setProgress(100);
      await fetchRegression();
    } catch (e) { clearInterval(poll); }
    setLoading(false);
  };

  const heatmapLayout = (title) => ({
    width: 480, height: 460,
    title: { text: title, font: { color: '#0f172a', size: 16 } },
    xaxis: { title: 'Spatial Dimension (x)', dtick: Math.max(1, Math.round(params.m * 0.1)), automargin: true },
    yaxis: { title: 'Spatial Dimension (y)', dtick: Math.max(1, Math.round(params.m * 0.1)), autorange: true, automargin: true },
    margin: { l: 80, r: 40, b: 80, t: 80 }
  });

  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>2D Thermal Diffusion Analysis System</h1>
        <p>Numerical Solutions for Steady-State Heat Conduction | IIT Patna</p>
      </div>

      <div className="main-layout">
        <div className="card">
          <h3>Configuration Panel</h3>
          <div className="input-group"><label>Nodal Density (m)</label><input type="number" value={params.m} onChange={(e)=>setParams({...params, m: Number(e.target.value)})} /></div>
          <div className="boundary-grid">
            <div className="input-group"><label>Boundary Top (K)</label><input type="number" value={params.Td} onChange={(e)=>setParams({...params, Td: Number(e.target.value)})} /></div>
            <div className="input-group"><label>Boundary Bottom (K)</label><input type="number" value={params.Tu} onChange={(e)=>setParams({...params, Tu: Number(e.target.value)})} /></div>
            <div className="input-group"><label>Boundary Left (K)</label><input type="number" value={params.Tl} onChange={(e)=>setParams({...params, Tl: Number(e.target.value)})} /></div>
            <div className="input-group"><label>Boundary Right (K)</label><input type="number" value={params.Tr} onChange={(e)=>setParams({...params, Tr: Number(e.target.value)})} /></div>
          </div>
          <button className="btn-exec" onClick={runSimulation} disabled={loading}>{loading ? `Processing (${progress}%)` : 'Execute Simulation'}</button>
          {loading && <div className="progress-area"><div className="progress-bg"><div className="progress-fill" style={{ width: `${progress}%` }}></div></div></div>}
        </div>

        <div className="card">
          <h3>Predictive Performance Metrics</h3>
          <div className="comparison-grid">
            <div className="stat-item"><span>Est. Iterations</span><strong>{prediction.iters}</strong>{results && <small>Actual: {results.iters}</small>}</div>
            <div className="stat-item"><span>Est. Runtime</span><strong>{prediction.time}s</strong>{results && <small>Actual: {results.time.toFixed(4)}s</small>}</div>
          </div>
          {results && <div className="accuracy-box fade-in"><span>Accuracy against Analytical</span><h2>{results.validation_score.toFixed(3)}%</h2></div>}
        </div>
      </div>

      {results && (
        <div className="fade-in">
          <div className="card plot-grid">
            <Plot data={[{ z: results.fdm, type: 'heatmap', colorscale: 'Jet', zsmooth: 'fast', hovertemplate: 'x: %{x}<br>y: %{y}<br>Temp: %{z:.2f} K<extra></extra>' }]} layout={heatmapLayout('Numerical Solution (SOR)')} />
            <Plot data={[{ z: results.analytic, type: 'heatmap', colorscale: 'Jet', zsmooth: 'fast', hovertemplate: 'x: %{x}<br>y: %{y}<br>Temp: %{z:.2f} K<extra></extra>' }]} layout={heatmapLayout('Analytical Solution (Fourier)')} />
          </div>
          {regData && (
            <div className="card">
              <h3>Computational Complexity Profile</h3>
              <div className="reg-plots">
                <Plot data={[{ x: regData.x_range, y: regData.y_iter_curve, mode: 'lines', name: 'Estimated O(n)' }, { x: [params.m], y: [results.iters], mode: 'markers', name: 'Measured', marker: { color: '#ef4444', size: 10 } }]} layout={{ width: 480, height: 380, title: 'Iterative Convergence Trend', xaxis: {title: 'Mesh Resolution (m)'}, yaxis: {title: 'Iterations'} }} />
                <Plot data={[{ x: regData.x_range, y: regData.y_time_curve, mode: 'lines', name: 'Estimated O(n<sup>3</sup>) '}, { x: [params.m], y: [results.time], mode: 'markers', name: 'Measured', marker: { color: '#ef4444', size: 10 } }]} layout={{ width: 480, height: 380, title: 'Temporal Complexity Trend', xaxis: {title: 'Mesh Resolution (m)'}, yaxis: {title: 'Computation Time (s)'} }} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;