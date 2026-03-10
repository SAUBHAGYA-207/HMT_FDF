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
  
  const [query, setQuery] = useState({ x: 0, y: 0 });
  const [queryResult, setQueryResult] = useState(null);

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
    setLoading(true); setProgress(0); setResults(null); setQueryResult(null);
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

  const handleQuery = () => {
    if (!results) return;
    const { x, y } = query;
    const m = params.m;
    if (x < 0 || x > m || y < 0 || y > m) {
      alert(`Coordinates must be between 0 and ${m}`);
      return;
    }
    const fdmVal = results.fdm[m - Math.round(y)][Math.round(x)];
    const anaVal = results.analytic[m - Math.round(y)][Math.round(x)];
    const acc = 100 * (1 - Math.abs(fdmVal - anaVal) / Math.max(anaVal, 1));
    setQueryResult({ fdm: fdmVal.toFixed(2), analytic: anaVal.toFixed(2), accuracy: acc.toFixed(3) });
  };

  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>2D Thermal Diffusion Analysis System</h1>
        <p>Numerical Solutions for Steady-State Heat Conduction | IIT Patna</p>
      </div>

      <div className="top-section-grid">
        <div className="card">
          <h3>Configuration Panel</h3>
          <div className="input-group"><label>Mesh Resolution (m)</label><input type="number" value={params.m} onChange={(e)=>setParams({...params, m: Number(e.target.value)})} /></div>
          <div className="boundary-grid">
            <div className="input-group"><label>Boundary Top (K)</label><input type="number" value={params.Td} onChange={(e)=>setParams({...params, Td: Number(e.target.value)})} /></div>
            <div className="input-group"><label>Boundary Bottom (K)</label><input type="number" value={params.Tu} onChange={(e)=>setParams({...params, Tu: Number(e.target.value)})} /></div>
            <div className="input-group"><label>Boundary Left (K)</label><input type="number" value={params.Tl} onChange={(e)=>setParams({...params, Tl: Number(e.target.value)})} /></div>
            <div className="input-group"><label>Boundary Right (K)</label><input type="number" value={params.Tr} onChange={(e)=>setParams({...params, Tr: Number(e.target.value)})} /></div>
          </div>
          <button className="btn-exec" onClick={runSimulation} disabled={loading}>{loading ? `Processing (${progress}%)` : 'Execute Simulation'}</button>
          {loading && <div className="progress-bg"><div className="progress-fill" style={{ width: `${progress}%` }}></div></div>}
        </div>

        <div className="metrics-layout">
          <div className="card metric-card"><span>Est. Iterations</span><strong>{prediction.iters}</strong></div>
          <div className="card metric-card"><span>Actual Iterations</span><strong>{results ? results.iters : '--'}</strong></div>
          <div className="card metric-card"><span>Est. Runtime</span><strong>{prediction.time}s</strong></div>
          <div className="card metric-card"><span>Actual Runtime</span><strong>{results ? `${results.time.toFixed(4)}s` : '--'}</strong></div>
        </div>
      </div>

      {results && (
        <div className="fade-in">
          <div className="card accuracy-banner">
            <span>Solution Convergence Index</span>
            <h2>{results.validation_score.toFixed(3)}%</h2>
          </div>

          <div className="plot-grid">
            <Plot data={[{ z: results.fdm, type: 'heatmap', colorscale: 'Jet' }]} layout={{ title: 'Numerical Solution (SOR)', width: 450, height: 450 }} />
            <Plot data={[{ z: results.analytic, type: 'heatmap', colorscale: 'Jet' }]} layout={{ title: 'Analytical Solution', width: 450, height: 450 }} />
          </div>

          <div className="card query-container">
            <h3>Coordinate Precision Analysis</h3>
            <div className="query-inputs">
              <div className="input-group"><label>x-coordinate</label><input type="number" value={query.x} onChange={(e)=>setQuery({...query, x: Number(e.target.value)})} /></div>
              <div className="input-group"><label>y-coordinate</label><input type="number" value={query.y} onChange={(e)=>setQuery({...query, y: Number(e.target.value)})} /></div>
              <button className="btn-query" onClick={handleQuery}>Query Temperature</button>
            </div>
            {queryResult && (
              <div className="query-results fade-in">
                <div className="query-stat"><span>FDM Temp</span><strong>{queryResult.fdm} K</strong></div>
                <div className="query-stat"><span>Analytic Temp</span><strong>{queryResult.analytic} K</strong></div>
                <div className="query-stat"><span>Local Accuracy</span><strong>{queryResult.accuracy}%</strong></div>
              </div>
            )}
          </div>

          {regData && (
            <div className="card complexity-card">
              <h3>System Complexity Profiles</h3>
              <div className="plot-grid">
                <Plot data={[{ x: regData.x_range, y: regData.y_iter_curve, mode: 'lines', name: 'Est. O(n)' }, { x: [params.m], y: [results.iters], mode: 'markers', marker: { color: 'red' } }]} layout={{ width: 450, height: 350, title: 'Iteration Trend', xaxis: {title: 'm'}, yaxis: {title: 'Iters'} }} />
                <Plot data={[{ x: regData.x_range, y: regData.y_time_curve, mode: 'lines', name: 'Est. O(n³)' }, { x: [params.m], y: [results.time], mode: 'markers', marker: { color: 'red' } }]} layout={{ width: 450, height: 350, title: 'Runtime Trend', xaxis: {title: 'm'}, yaxis: {title: 'Time (s)'} }} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;