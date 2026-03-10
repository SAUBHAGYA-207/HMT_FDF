import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

function App() {
  const [params, setParams] = useState({ m: 50, Tu: 100, Td: 0, Tl: 0, Tr: 0 });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('checking');
  const [regData, setRegData] = useState(null);
  const [query, setQuery] = useState({ x: 0, y: 0 });
  const [queryResult, setQueryResult] = useState(null);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  useEffect(() => {
    const init = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/health`);
        setStatus(res.data.status === 'online' ? 'online' : 'offline');
        const r = await axios.get(`${API_BASE_URL}/api/regression`);
        if(!r.data.error) setRegData(r.data);
      } catch { setStatus('offline'); }
    };
    init();
  }, [API_BASE_URL]);

  const runSimulation = async () => {
    setLoading(true); setProgress(0); setResults(null);
    const poll = setInterval(async () => {
      const res = await axios.get(`${API_BASE_URL}/api/progress`);
      setProgress(res.data.progress);
    }, 300);
    try {
      const res = await axios.post(`${API_BASE_URL}/api/calculate`, params);
      setResults(res.data);
      clearInterval(poll); setProgress(100);
    } catch { clearInterval(poll); }
    setLoading(false);
  };

  const handleQuery = () => {
    if (!results) return;
    const { x, y } = query; const m = params.m;
    const fdmVal = results.fdm[m - Math.round(y)][Math.round(x)];
    const anaVal = results.analytic[m - Math.round(y)][Math.round(x)];
    setQueryResult({ fdm: fdmVal.toFixed(2), ana: anaVal.toFixed(2) });
  };

  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>Thermal Analysis Engine</h1>
        <p>Numerical Solutions | IIT Patna</p>
        <div className={`status-pill ${status}`}>
          {status === 'online' ? '● System Connected' : '○ Connection Error'}
        </div>
      </div>

      <div className="top-section-grid">
        <div className="card">
          <h3>Simulation Setup</h3>
          <div className="input-group"><label>Grid Resolution (m)</label><input type="number" value={params.m} onChange={(e)=>setParams({...params, m: Number(e.target.value)})} /></div>
          <div className="boundary-grid">
            {['Td','Tu','Tl','Tr'].map(k => (
              <div key={k} className="input-group"><label>{k} (K)</label><input type="number" value={params[k]} onChange={(e)=>setParams({...params, [k]: Number(e.target.value)})} /></div>
            ))}
          </div>
          <button className="btn-exec" onClick={runSimulation} disabled={loading}>{loading ? `Solving (${progress}%)` : 'Solve Heat Equation'}</button>
        </div>
        
        {results && (
          <div className="card query-container">
            <h3>Coordinate Query</h3>
            <div className="query-inputs">
              <input type="number" placeholder="x" onChange={(e)=>setQuery({...query, x:Number(e.target.value)})}/>
              <input type="number" placeholder="y" onChange={(e)=>setQuery({...query, y:Number(e.target.value)})}/>
              <button onClick={handleQuery}>Get Temp</button>
            </div>
            {queryResult && <div className="query-results"><strong>Numerical: {queryResult.fdm}K</strong> | <strong>Analytical: {queryResult.ana}K</strong></div>}
          </div>
        )}
      </div>

      {results && (
        <div className="plot-grid">
          <Plot data={[{ z: results.fdm, type: 'heatmap', colorscale: 'Jet' }]} layout={{ title: 'SOR Numerical', width: 450, height: 400 }} />
          <Plot data={[{ z: results.analytic, type: 'heatmap', colorscale: 'Jet' }]} layout={{ title: 'Fourier Analytical', width: 450, height: 400 }} />
        </div>
      )}
    </div>
  );
}
export default App;