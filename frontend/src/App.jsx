import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

function App() {
  const [params, setParams] = useState({ m: 50, Tu: 100, Td: 0, Tl: 0, Tr: 0 });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [regData, setRegData] = useState(null);

  const handleInputChange = (e) => setParams({ ...params, [e.target.name]: Number(e.target.value) });

  const API_BASE_URL ='http://localhost:8000';

  useEffect(() => {
    fetchRegression();
  }, []);

  const fetchRegression = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/regression?t=${new Date().getTime()}`);
      if (!response.data.error) {
        setRegData({ ...response.data });
      }
    } catch (e) {
      console.error("Failed to fetch regression data:", e);
    }
  };

  const runSimulation = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/api/calculate`, params);
      setResults(res.data);
      
      setTimeout(async () => {
        await fetchRegression();
        setLoading(false);
      }, 500);

    } catch (e) {
      console.error("Simulation Error:", e);
      setLoading(false);
    }
  };

  const heatmapLayout = (title) => {
    const dynamicStep = Math.max(1, Math.round(params.m * 0.10));
    return {
      width: 480, height: 460,
      title: { text: title, font: { color: '#064e3b', weight: 'bold' } },
      xaxis: { 
        title: { text: 'Distance along Width (x)' }, 
        dtick: dynamicStep,
        zeroline: true, showline: true, mirror: 'ticks', automargin: true 
      },
      yaxis: { 
        title: { text: 'Distance along Height (y)' }, 
        dtick: dynamicStep,
        autorange: true, zeroline: true, showline: true, mirror: 'ticks', automargin: true 
      },
      margin: { l: 80, r: 40, b: 80, t: 80 }, 
      paper_bgcolor: 'rgba(0,0,0,0)',
    };
  };

  return (
    <div className="dashboard-container" style={{ backgroundColor: '#f8fafc', minHeight: '100vh', padding: '20px' }}>
      <div className="header" style={{ borderBottom: '4px solid #10b981', marginBottom: '30px' }}>
        <h1 style={{ color: '#064e3b' }}>2D Steady State Thermal Analysis Dashboard</h1>
        <p style={{ color: '#065f46' }}>Numerical SOR vs. Analytical Fourier Series</p>
      </div>

      <div className="top-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div className="card" style={{ backgroundColor: 'white', borderRadius: '12px', padding: '20px', border: '1px solid #d1fae5' }}>
          <h3 style={{ color: '#047857' }}>Simulation Control</h3>
          <div className="input-group">
            <label>Grid Size (m)</label>
            <input type="number" name="m" value={params.m} onChange={handleInputChange} style={{ borderColor: '#10b981' }} />
          </div>
          <div className="input-grid" style={{ marginTop: '15px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            {/* INVERTED LABELS BELOW */}
            <div className="input-group"><label>Bottom Temp (Tu)</label><input type="number" name="Tu" value={params.Tu} onChange={handleInputChange} /></div>
            <div className="input-group"><label>Top Temp (Td)</label><input type="number" name="Td" value={params.Td} onChange={handleInputChange} /></div>
            <div className="input-group"><label>Left Temp (Tl)</label><input type="number" name="Tl" value={params.Tl} onChange={handleInputChange} /></div>
            <div className="input-group"><label>Right Temp (Tr)</label><input type="number" name="Tr" value={params.Tr} onChange={handleInputChange} /></div>
          </div>
          <button className="btn btn-primary" onClick={runSimulation} disabled={loading} style={{ backgroundColor: '#10b981', border: 'none', width: '100%', marginTop: '15px', color: 'white', padding: '10px', borderRadius: '8px', cursor: 'pointer' }}>
            {loading ? 'Solving Physics...' : 'Execute Analysis'}
          </button>
        </div>

        {results && (
          <div className="card" style={{ backgroundColor: 'white', borderRadius: '12px', padding: '20px', border: '1px solid #d1fae5' }}>
            <h3 style={{ color: '#047857' }}>Validation Summary</h3>
            <div className="metrics-container" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div className="metric-box" style={{ background: '#f0fdf4', padding: '15px', borderRadius: '8px' }}>
                <div className="metric-label" style={{ color: '#065f46' }}>Solver Accuracy</div>
                <div className="metric-value" style={{ color: '#059669', fontSize: '1.8rem', fontWeight: 'bold' }}>{results.accuracy.toFixed(3)}%</div>
              </div>
              <div style={{ display: 'flex', gap: '20px' }}>
                <div className="metric-box" style={{ flex: 1, background: '#f8fafc', padding: '10px', borderRadius: '8px' }}>
                  <div className="metric-label">Iterations</div>
                  <div className="metric-value" style={{ fontWeight: 'bold' }}>{results.iters}</div>
                </div>
                <div className="metric-box" style={{ flex: 1, background: '#f8fafc', padding: '10px', borderRadius: '8px' }}>
                  <div className="metric-label">Time Taken</div>
                  <div className="metric-value" style={{ fontWeight: 'bold' }}>{results.time.toFixed(4)}s</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {results && (
        <div className="card" style={{ marginTop: '30px', backgroundColor: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #d1fae5' }}>
          <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Plot 
              data={[{ 
                z: results.fdm, 
                type: 'heatmap', 
                colorscale: 'Jet', 
                zsmooth: 'fast',
                hovertemplate: 'x: %{x}<br>y: %{y}<br>Temp: %{z:.2f} K<extra></extra>' 
              }]} 
              layout={heatmapLayout('Numerical Result (FDM)')} 
            />
            <Plot 
              data={[{ 
                z: results.analytic, 
                type: 'heatmap', 
                colorscale: 'Jet', 
                zsmooth: 'fast',
                hovertemplate: 'x: %{x}<br>y: %{y}<br>Temp: %{z:.2f} K<extra></extra>' 
              }]} 
              layout={heatmapLayout('Analytical Result (Fourier)')} 
            />
          </div>
        </div>
      )}

      {regData && (
        <div className="card" style={{ marginTop: '30px', backgroundColor: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #d1fae5' }}>
          <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Plot 
              data={[
                { x: regData.grid_size, y: regData.iterations, mode: 'markers', name: 'Raw Data', marker: { color: '#065f46', size: 10 } }, 
                { x: regData.grid_size, y: regData.line_iters, mode: 'lines', name: `Linear Fit (R²=${regData.r2_iters.toFixed(3)})`, line: { color: '#10b981', width: 3 } }
              ]}
              layout={{ 
                width: 480, height: 420, 
                title: 'Iteration Complexity', 
                xaxis: { title: { text: 'Number of Grids (m)' }, dtick: Math.round(Math.max(...regData.grid_size) * 0.1), automargin: true }, 
                yaxis: { title: { text: 'Total Iterations' }, automargin: true }, 
                legend: { orientation: 'h', y: -0.3 },
                margin: { l: 80, r: 40, b: 100, t: 50 } 
              }}
            />
            <Plot 
              data={[
                { x: regData.grid_size, y: regData.time_taken, mode: 'markers', name: 'Raw Data', marker: { color: '#064e3b', size: 10 } }, 
                { x: regData.grid_size, y: regData.line_time, mode: 'lines', name: `Cubic Fit (R²=${regData.r2_time.toFixed(3)})`, line: { color: '#059669', width: 3 } }
              ]}
              layout={{ 
                width: 480, height: 420, 
                title: 'Time Complexity', 
                xaxis: { title: { text: 'Number of Grids (m)' }, dtick: Math.round(Math.max(...regData.grid_size) * 0.1), automargin: true }, 
                yaxis: { title: { text: 'Time Taken (s)' }, automargin: true }, 
                legend: { orientation: 'h', y: -0.3 },
                margin: { l: 80, r: 40, b: 100, t: 50 } 
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;