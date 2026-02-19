import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Music, History as HistoryIcon, Sparkles, Download, RotateCcw, Plus, Trash2, Mic2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// --- CONFIG ---
const SPRING_API = "http://127.0.0.1:8081/api/tracks";
const STATIC_BASE = "http://127.0.0.1:8000/static/output";

const App = () => {
  const [view, setView] = useState('generate');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (view === 'history') fetchHistory();
  }, [view]);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${SPRING_API}/history`);
      setHistory(res.data);
    } catch (err) {
      console.error("Failed to fetch history", err);
    }
  };

  return (
    <div className="container">
      {/* Particles bg (simplified) */}
      <div className="particles">
        {[...Array(6)].map((_, i) => <div key={i} className="particle" />)}
      </div>

      <nav className="nav-bar">
        <button
          className={`nav-link ${view === 'generate' ? 'active' : ''}`}
          onClick={() => setView('generate')}
        >
          <Music size={18} style={{ marginRight: 8 }} /> Generate
        </button>
        <button
          className={`nav-link ${view === 'history' ? 'active' : ''}`}
          onClick={() => setView('history')}
        >
          <HistoryIcon size={18} style={{ marginRight: 8 }} /> History
        </button>
      </nav>

      <AnimatePresence mode="wait">
        {view === 'generate' && <Generator key="gen" setLoading={setLoading} loading={loading} />}
        {view === 'history' && <History key="hist" history={history} fetchHistory={fetchHistory} />}
      </AnimatePresence>

      <footer className="footer" style={{ marginTop: 60 }}>
        <p>🎵 CoverComposer React UI · Built with Spring Boot, FastAPI & Markov Chains</p>
      </footer>

      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">Composing with AI...</div>
        </div>
      )}
    </div>
  );
};

const Generator = ({ setLoading, loading }) => {
  const [result, setResult] = useState(null);
  const [formData, setFormData] = useState({
    mood: 'happy',
    genre: 'pop',
    tempo: 120,
    style: 'simple',
    duration: 'medium',
    useAI: false
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Talk to Spring Boot, which will call Python
      const res = await axios.post(`${SPRING_API}/generate`, {
        ...formData,
        use_ai: formData.useAI ? 'on' : 'off'
      });
      setResult(res.data);
    } catch (err) {
      alert("Composition failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (result) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="result-header">
          <div className="success-icon">✅</div>
          <h2>Track Composition Ready</h2>
        </div>

        <div className="track-details">
          <div className="detail-item">
            <div className="detail-icon">🎭</div>
            <div className="detail-info">
              <span className="detail-label">Mood</span>
              <span className="detail-value">{result.mood}</span>
            </div>
          </div>
          <div className="detail-item">
            <div className="detail-icon">🎸</div>
            <div className="detail-info">
              <span className="detail-label">Genre</span>
              <span className="detail-value">{result.genre}</span>
            </div>
          </div>
        </div>

        <div className="audio-player-wrapper">
          <h3>Preview Audio</h3>
          <audio controls autoPlay src={`${STATIC_BASE}/${result.filename}`} />
        </div>

        <div className="action-buttons">
          <a href={`${STATIC_BASE}/${result.filename}`} download className="btn btn-download">
            <Download size={18} /> Download WAV
          </a>
          <button onClick={() => setResult(null)} className="btn btn-new">
            <Plus size={18} /> New Track
          </button>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
      className="card"
    >
      <header className="header">
        <div className="logo-icon">🎵</div>
        <h1>Composer</h1>
        <p className="subtitle">AI-Powered <span>Melody</span> Generation</p>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="form-group">
            <label><Sparkles size={14} /> Mood</label>
            <select value={formData.mood} onChange={(e) => setFormData({ ...formData, mood: e.target.value })}>
              <option value="happy">Happy & Bright</option>
              <option value="sad">Sad & Melancholy</option>
              <option value="energetic">Energetic & Driving</option>
              <option value="calm">Calm & Peaceful</option>
            </select>
          </div>

          <div className="form-group">
            <label><Mic2 size={14} /> Genre</label>
            <select value={formData.genre} onChange={(e) => setFormData({ ...formData, genre: e.target.value })}>
              <option value="pop">Pop</option>
              <option value="rock">Rock</option>
              <option value="jazz">Jazz</option>
              <option value="electronic">Electronic</option>
            </select>
          </div>

          <div className="form-group">
            <label>Tempo (BPM)</label>
            <input type="number" value={formData.tempo} onChange={(e) => setFormData({ ...formData, tempo: e.target.value })} min="60" max="200" />
          </div>

          <div className="form-group">
            <label>Complexity</label>
            <select value={formData.style} onChange={(e) => setFormData({ ...formData, style: e.target.value })}>
              <option value="simple">Simple</option>
              <option value="complex">Complex</option>
            </select>
          </div>

          <div className="form-group" style={{ gridColumn: '1 / -1' }}>
            <label className="ai-toggle-container">
              <span>✨ Use Gemini AI (Google)</span>
              <div className="toggle-wrapper" style={{ marginLeft: 'auto' }}>
                <input
                  type="checkbox"
                  className="toggle-input"
                  checked={formData.useAI}
                  onChange={(e) => setFormData({ ...formData, useAI: e.target.checked })}
                />
                <div className="toggle-switch">
                  <div className="toggle-dot"></div>
                </div>
              </div>
            </label>
          </div>
        </div>

        <button type="submit" className="btn-generate">
          Generate Track
        </button>
      </form>
    </motion.div>
  );
};

const History = ({ history, fetchHistory }) => {
  const handleClear = async () => {
    if (!window.confirm("Delete all history?")) return;
    try {
      await axios.delete(`${SPRING_API}/clear`);
      fetchHistory();
    } catch (err) { alert("Failed to clear history"); }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}
      className="card"
    >
      <div className="section-title">
        <HistoryIcon size={24} /> History ({history.length})
      </div>

      <div className="history-list">
        {history.map((track) => (
          <div key={track.id} className="history-item">
            <div className="history-meta">
              <span className="history-mood">{track.mood}</span>
              <span className="history-sep">·</span>
              <span className="history-genre">{track.genre}</span>
            </div>
            <div className="history-player">
              <audio controls src={`${STATIC_BASE}/${track.filename}`} />
            </div>
            <div className="history-actions">
              <a href={`${STATIC_BASE}/${track.filename}`} download className="btn-mini btn-mini-download">
                <Download size={14} />
              </a>
            </div>
          </div>
        ))}
      </div>

      {history.length > 0 && (
        <div style={{ marginTop: 30, textAlign: 'center' }}>
          <button onClick={handleClear} className="btn-new" style={{ color: '#ff4757' }}>
            <Trash2 size={16} /> Clear All Records
          </button>
        </div>
      )}
    </motion.div>
  );
};

export default App;
