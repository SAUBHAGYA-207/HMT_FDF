import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

function App() {
  const [params, setParams] = useState({ m: 50, Tu: 25, Td: 25, Tl: 25, Tr: 25 });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('checking'); // Dynamic status
  const [regData, setRegData] = useState(null);
  const [prediction, setPrediction] = useState({ iters: '--', time: '--' });
  const [query, setQuery] = useState({ x: 0, y: 0 });
  const [queryResult, setQueryResult] = useState(null);

  const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await axios.get(`${API}/api/health`);
        if (res.data.status === 'online') {
          setStatus('online');
          fetchReg();
        } else {
          setStatus('offline');
        }
      } catch (e) {
        setStatus('offline');
      }
    };
    checkBackend();
  }, [API]);

  const fetchReg = async () => {
    try { const r = await axios.get(`${API}/api/regression`); if(!r.data.error) setRegData(r.data); } catch(e){}
  };

  const solve = async () => {
    if (regData?.coeffs) {
      const { iter_m, iter_c, time_coeffs, time_c } = regData.coeffs;
      const m = params.m;
      setPrediction({ iters: Math.round(iter_m*m + iter_c), time: (time_c + time_coeffs[1]*m + time_coeffs[2]*m**2 + time_coeffs[3]*m**3).toFixed(4) });
    }
    setLoading(true); setProgress(0); setResults(null);
    const p = setInterval(async () => { 
        try { const r = await axios.get(`${API}/api/progress`); setProgress(r.data.progress); } catch(e){}
    }, 250);
    try {
      const res = await axios.post(`${API}/api/calculate`, params);
      setResults(res.data); clearInterval(p); setProgress(100); fetchReg();
    } catch (e) { clearInterval(p); }
    setLoading(false);
  };

  const probe = () => {
    if(!results) return;
    const {x, y} = query; const m = params.m;
    if(x < 0 || x > m || y < 0 || y > m) return alert("Out of bounds!");
    const f = results.fdm[y][x]; const a = results.analytic[y][x];
    const acc = 100 * (1 - Math.abs(f-a)/Math.max(a,1));
    setQueryResult({ f: f.toFixed(3), a: a.toFixed(3), acc: acc.toFixed(3) });
  };

  const squareLayout = (title) => ({
    title: title, width: 500, height: 500,
    xaxis: { constrain: 'domain', title: 'X-axis' },
    yaxis: { scaleanchor: 'x', scaleratio: 1, title: 'Y-axis' },
    margin: { l: 60, r: 40, t: 50, b: 60 },
  });

  return (
    <div className="dashboard">
      <div className="header">
        <h1>2D Steady State thermal Analysis Dashboard</h1>
        <div className={`status-badge ${status}`}>
          {status === 'online' ? '● System Online' : status === 'offline' ? '○ System Offline' : '◌ Checking...'}
        </div>
      </div>

      <div className="top-row">
        <div className="card config-panel">
          <h3>Parameters</h3>
          <div className="input-group">
            <label>Mesh (m)</label>
            <input type="number" value={params.m} onChange={e=>setParams({...params, m:Number(e.target.value)})} />
          </div>
          <div className="boundary-inputs">
            {['Td','Tu','Tl','Tr'].map(k => (
              <div key={k} className="input-group">
                <label>{k} (K)</label>
                <input type="number" placeholder={k} onChange={e=>setParams({...params, [k]:Number(e.target.value)})} />
              </div>
            ))}
          </div>
          <button className="execute-btn" onClick={solve} disabled={loading || status === 'offline'}>
            {loading ? `Solving ${progress}%` : 'Execute Analysis'}
          </button>
          {loading && <div className="loader-bar"><div className="fill" style={{width: `${progress}%`}}></div></div>}
        </div>

        <div className="metrics-container">
          <div className="card metric-box"><span>Est. Iterations</span><strong>{prediction.iters}</strong></div>
          <div className="card metric-box"><span>Actual Iterations</span><strong>{results?.iters || '--'}</strong></div>
          <div className="card metric-box"><span>Est. Time</span><strong>{prediction.time}</strong></div>
          <div className="card metric-box"><span>Actual Time</span><strong>{results ? results.time.toFixed(4)+'s' : '--'}</strong></div>
        </div>
      </div>

      {results && (
        <div className="results-area fade-in">
          <div className="card plot-row">
            <div className="heatmap-container">
                <h4>Numerical Solution (FDM)</h4>
                <Plot data={[{z:results.fdm, type:'heatmap', colorscale:'Jet'}]} layout={squareLayout('')} />
            </div>
            <div className="heatmap-container">
                <h4>Analytical Solution (Fourier)</h4>
                <Plot data={[{z:results.analytic, type:'heatmap', colorscale:'Jet'}]} layout={squareLayout('')} />
            </div>
          </div>

          <div className="bottom-row">
            <div className="card probe-card">
              <h3>Coordinate Precision Probe</h3>
              <div className="probe-inputs">
                <div className="input-group">
                  <label style={{fontSize: '0.7rem', fontWeight: 'bold', color: '#64748b', display: 'block', marginBottom: '4px'}}>X-COORDINATE</label>
                  <input type="number" placeholder="0" onChange={e=>setQuery({...query, x:Number(e.target.value)})}/>
                </div>
                <div className="input-group">
                  <label style={{fontSize: '0.7rem', fontWeight: 'bold', color: '#64748b', display: 'block', marginBottom: '4px'}}>Y-COORDINATE</label>
                  <input type="number" placeholder="0" onChange={e=>setQuery({...query, y:Number(e.target.value)})}/>
                </div>
                <button onClick={probe}>Run Query</button>
              </div>
              
              {queryResult && (
                <div className="probe-results fade-in">
                  <p>
                    <span>Numerical (FDM):</span> 
                    <strong>{queryResult.f} K</strong>
                  </p>
                  <p>
                    <span>Analytical (Fourier):</span> 
                    <strong>{queryResult.a} K</strong>
                  </p>
                  <p className="acc-text">
                    <span>Point Accuracy:</span> 
                    <strong>{queryResult.acc}%</strong>
                  </p>
                </div>
              )}
            </div>

<div className="card regression-card">
  <h3>Historical Performance Comparison</h3>
  <div className="small-plots" style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
    
    {/* Plot 1: Iterations Scatter */}
    <Plot 
      data={[
        {
          x: regData?.m_values, 
          y: regData?.iter_values, 
          mode: 'markers', 
          name: 'Past Runs', 
          marker: {color: '#10b981', size: 8, opacity: 0.6}
        },
        {
          x: [params.m], 
          y: [results.iters], 
          mode: 'markers', 
          name: 'Current', 
          marker: {color: '#ef4444', size: 12, symbol: 'diamond'}
        }
      ]} 
      layout={{
        width: 420, height: 350,
        title: '<b>Iteration Distribution</b>',
        xaxis: { title: { text: 'Grid Size (m)', standoff: 20 }, gridcolor: '#f1f5f9' },
        yaxis: { title: { text: 'Iterations (N)', standoff: 20 }, gridcolor: '#f1f5f9' },
        margin: { t: 60, l: 80, r: 30, b: 80 },
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
      }} 
    />

    {/* Plot 2: Runtime Scatter */}
    <Plot 
      data={[
        {
          x: regData?.m_values, 
          y: regData?.time_values, 
          mode: 'markers', 
          name: 'Past Runs', 
          marker: {color: '#6366f1', size: 8, opacity: 0.6}
        },
        {
          x: [params.m], 
          y: [results.time], 
          mode: 'markers', 
          name: 'Current', 
          marker: {color: '#ef4444', size: 12, symbol: 'diamond'}
        }
      ]} 
      layout={{
        width: 420, height: 350,
        title: '<b>Runtime Distribution</b>',
        xaxis: { title: { text: 'Grid Size (m)', standoff: 20 }, gridcolor: '#f1f5f9' },
        yaxis: { title: { text: 'Time (Seconds)', standoff: 20 }, gridcolor: '#f1f5f9' },
        margin: { t: 60, l: 80, r: 30, b: 80 },
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
      }} 
    />
  </div>
</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;