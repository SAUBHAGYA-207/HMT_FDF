import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

function App() {
  const [params, setParams] = useState({ m: 50, Tu: 0, Td: 100, Tl: 0, Tr: 0 });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [regData, setRegData] = useState(null);

  const handleInputChange = (e) => setParams({ ...params, [e.target.name]: Number(e.target.value) });

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const runSimulation = async () => {
  setLoading(true);
  try {
    // Uses the live URL on the web, or localhost if the variable isn't set
    const res = await axios.post(`${API_BASE_URL}/api/calculate`, params);
    setResults(res.data);
    const reg = await axios.get(`${API_BASE_URL}/api/regression`);
    if (!reg.data.error) setRegData(reg.data);
  } catch (e) { console.error(e); }
  setLoading(false);
};

  const heatmapLayout = (title) => ({
    width: 460, height: 440,
    title: { text: title, font: { color: '#064e3b', weight: 'bold' } },
    xaxis: { title: { text: 'Number of Grids (m)' }, zeroline: true, showline: true, mirror: 'ticks' },
    yaxis: { title: { text: 'Iterations' }, autorange: true, zeroline: true, showline: true, mirror: 'ticks' },
    margin: { l: 60, r: 40, b: 60, t: 80 },
    paper_bgcolor: 'rgba(0,0,0,0)',
  });

  return (
    <div className="dashboard-container" style={{ backgroundColor: '#f8fafc', minHeight: '100vh', padding: '20px' }}>
      <div className="header" style={{ borderBottom: '4px solid #10b981', marginBottom: '30px' }}>
        <h1 style={{ color: '#064e3b' }}>2D Steady State Thermal Analysis Dashboard</h1>
        <p style={{ color: '#065f46' }}>Finite Difference Method (SOR) vs. Analytical Fourier Series</p>
      </div>

      <div className="top-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div className="card" style={{ backgroundColor: 'white', borderRadius: '12px', padding: '20px', border: '1px solid #d1fae5' }}>
          <h3 style={{ color: '#047857' }}>Simulation Control</h3>
          <div className="input-group">
            <label>Grid Size (m)</label>
            <input type="number" name="m" value={params.m} onChange={handleInputChange} style={{ borderColor: '#10b981' }} />
          </div>
          <div className="input-grid" style={{ marginTop: '15px' }}>
            <div className="input-group"><label>Bottom Temperature </label><input type="number" name="Td" value={params.Td} onChange={handleInputChange} /></div>
            <div className="input-group"><label>Top Temperature</label><input type="number" name="Tu" value={params.Tu} onChange={handleInputChange} /></div>
            <div className="input-group"><label>Left Temperature</label><input type="number" name="Tl" value={params.Tl} onChange={handleInputChange} /></div>
            <div className="input-group"><label>Right Temperature</label><input type="number" name="Tr" value={params.Tr} onChange={handleInputChange} /></div>
          </div>
          <button className="btn btn-primary" onClick={runSimulation} disabled={loading} style={{ backgroundColor: '#10b981', border: 'none' }}>
            {loading ? 'Solving...' : 'Execute Analysis'}
          </button>
        </div>

        {results && (
          <div className="card" style={{ backgroundColor: 'white', borderRadius: '12px', padding: '20px', border: '1px solid #d1fae5' }}>
            <h3 style={{ color: '#047857' }}>Validation Summary</h3>
            <div className="metrics-container" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div className="metric-box">
                <div className="metric-label" style={{ color: '#065f46' }}>Solver Accuracy</div>
                <div className="metric-value" style={{ color: '#059669', fontSize: '1.8rem', fontWeight: 'bold' }}>{results.accuracy.toFixed(3)}%</div>
              </div>
              <div style={{ display: 'flex', gap: '20px' }}>
                <div className="metric-box" style={{ flex: 1 }}><div className="metric-label">Iterations</div><div className="metric-value">{results.iters}</div></div>
                <div className="metric-box" style={{ flex: 1 }}><div className="metric-label">Time Taken</div><div className="metric-value">{results.time.toFixed(4)}s</div></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {results && (
        <div className="card" style={{ marginTop: '30px', backgroundColor: 'white', padding: '20px', borderRadius: '12px' }}>
          <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Plot data={[{ z: results.fdm, type: 'heatmap', colorscale: 'Jet', zsmooth: 'fast' }]} layout={heatmapLayout('Numerical Result (FDM)')} />
            <Plot data={[{ z: results.analytic, type: 'heatmap', colorscale: 'Jet', zsmooth: 'fast' }]} layout={heatmapLayout('Analytical Result (Fourier)')} />
          </div>
        </div>
      )}

      {regData && (
        <div className="card" style={{ marginTop: '30px', backgroundColor: 'white', padding: '20px', borderRadius: '12px' }}>
          <h3 style={{ color: '#047857' }}>Complexity Trend Verification</h3>
          <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Plot 
              data={[{ x: regData.grid_size, y: regData.iterations, mode: 'markers', name: 'Raw Data', marker: { color: '#065f46' } }, { x: regData.grid_size, y: regData.line_iters, mode: 'lines', name: `Linear Fit (R²=${regData.r2_iters.toFixed(3)})`, line: { color: '#10b981' } }]}
              layout={{ width: 460, height: 380, title: 'Iteration Complexity', xaxis: { title: 'Number of Grids (m)' }, yaxis: { title: 'Iterations' }, legend: { orientation: 'h', y: -0.2 } }}
            />
            <Plot 
              data={[{ x: regData.grid_size, y: regData.time_taken, mode: 'markers', name: 'Raw Data', marker: { color: '#064e3b' } }, { x: regData.grid_size, y: regData.line_time, mode: 'lines', name: `Cubic Fit (R²=${regData.r2_time.toFixed(3)})`, line: { color: '#059669' } }]}
              layout={{ width: 460, height: 380, title: 'Time Complexity', xaxis: { title: 'Number of Grids (m)' }, yaxis: { title: 'Time Taken (s)' }, legend: { orientation: 'h', y: -0.2 } }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;