import { useState, useEffect, useRef } from 'react';
import { MOCK_DATA } from './mockData';

const PLACEHOLDERS = [
  "How does Synapse handle concurrent reads?",
  "what happens when server restarts",
  "What is the cosine similarity formula?",
  "How does the worker pool distribute computation?"
];

function App() {
  const [isLive, setIsLive] = useState(false);
  const [vectorCount, setVectorCount] = useState("10,240");
  
  const [query, setQuery] = useState("");
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  
  const [topK, setTopK] = useState(3);
  const [namespace, setNamespace] = useState("docs");
  
  const [results, setResults] = useState(null);
  const [displayedLatency, setDisplayedLatency] = useState("--ms");
  const [stats, setStats] = useState({ queries: 142, avgLatency: 3.8 });

  const animFrameRef = useRef(null);

  // Check live status
  useEffect(() => {
    const checkLive = async () => {
      try {
        const res = await fetch('/api/stats');
        if (res.ok) {
          const data = await res.json();
          setIsLive(true);
          setVectorCount(data.vectors.toLocaleString());
        }
      } catch (e) {
        console.log("Offline mode active.");
      }
    };
    checkLive();
  }, []);

  // Placeholder rotation
  useEffect(() => {
    const interval = setInterval(() => {
      if (!isFocused && query === "") {
        setPlaceholderIdx((prev) => (prev + 1) % PLACEHOLDERS.length);
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [isFocused, query]);

  const handleSearch = async () => {
    if (isSearching) return;
    
    const activeQuery = query.trim() || PLACEHOLDERS[placeholderIdx];
    const lowerQuery = activeQuery.toLowerCase();
    
    setIsSearching(true);
    setResults(null);
    setDisplayedLatency("0.0ms");

    // Skeletons
    const skeletons = Array(topK).fill({ isSkeleton: true });
    setResults(skeletons);

    let responseData = null;
    let targetLatency = 0;

    const startTime = performance.now();
    
    const updateLatency = (timestamp) => {
      const elapsed = timestamp - startTime;
      setDisplayedLatency(`${elapsed.toFixed(1)}ms`);
      animFrameRef.current = requestAnimationFrame(updateLatency);
    };
    animFrameRef.current = requestAnimationFrame(updateLatency);

    try {
      if (isLive) {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: activeQuery, namespace, top_k: topK })
        });
        if (!res.ok) throw new Error("API returned " + res.status);
        responseData = await res.json();
        targetLatency = responseData.latency;
      } else {
        throw new Error("Offline");
      }
    } catch (err) {
      // Mock Fallback
      responseData = MOCK_DATA["default"];
      for (const key in MOCK_DATA) {
        if (key !== "default" && lowerQuery.includes(key)) {
          responseData = MOCK_DATA[key];
          break;
        }
      }
      await new Promise(r => setTimeout(r, 80));
      targetLatency = responseData.latency + (Math.random() * 0.8 - 0.4);
    }

    cancelAnimationFrame(animFrameRef.current);
    setDisplayedLatency(`${targetLatency.toFixed(1)}ms`);
    
    setStats(prev => {
      const newQueries = prev.queries + 1;
      const newAvg = ((prev.avgLatency * prev.queries) + targetLatency) / newQueries;
      return { queries: newQueries, avgLatency: newAvg };
    });

    setResults(responseData.results.slice(0, topK));
    setIsSearching(false);
  };

  return (
    <>
      <div className="scanline"></div>
      {!isLive && (
        <div className="demo-banner">
          ⚠️ DEMO MODE — showing mock data. Connect a live Synapse instance to run real queries.
        </div>
      )}
      
      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="header-left">
            <span className="brand">SYNAPSE CACHE</span>
            <span className="version">v1.0.0</span>
          </div>
          <div className="header-right">
            <div className="status-pill">
              <span>STATUS: {isLive ? 'ONLINE' : 'MOCK'}</span>
              <div className={isLive ? "dot pulse-green" : "dot"} style={{ backgroundColor: isLive ? 'var(--success)' : 'var(--warning)' }}></div>
            </div>
            <div className="status-pill">
              <span>VECTORS: {vectorCount}</span>
            </div>
          </div>
        </header>

        {/* Main */}
        <main className="main">
          <div className={`query-container ${isFocused ? 'focused' : ''} ${isSearching ? 'loading' : ''}`}>
            <span className="prompt-char">&gt;</span>
            <input 
              type="text" 
              className="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder={PLACEHOLDERS[placeholderIdx]}
              spellCheck="false"
              autoComplete="off"
            />
          </div>

          {/* Controls */}
          <div className="controls">
            <button className="search-btn" onClick={handleSearch}>SEARCH</button>
            <div className="dropdown-group">
              <label>TOP-K:</label>
              <div className="select-wrapper">
                <select value={topK} onChange={e => setTopK(Number(e.target.value))}>
                  <option value={1}>1</option>
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                </select>
              </div>
            </div>
            <div className="dropdown-group">
              <label>NAMESPACE:</label>
              <div className="select-wrapper">
                <select value={namespace} onChange={e => setNamespace(e.target.value)}>
                  <option value="docs">docs</option>
                  <option value="code">code</option>
                </select>
              </div>
            </div>
          </div>

          {/* Divider */}
          <div className="divider">
            <div className="divider-line"></div>
            <span className="divider-text">── RESULTS ──</span>
            <div className="divider-line"></div>
            <span className="latency-text">LATENCY: {displayedLatency}</span>
          </div>

          {/* Results Area */}
          <div className="results">
            {!results && (
              <div className="empty-state">
                // run a query to see results
              </div>
            )}
            {results && results.length === 0 && (
              <div className="empty-state">No results found.</div>
            )}
            {results && results.map((res, index) => {
              if (res.isSkeleton) {
                return <div key={`skel-${index}`} className="skeleton-card" style={{ animationDelay: `${index * 60}ms` }}></div>;
              }

              let colorClass = 'color-low';
              let bgClass = 'bg-low';
              if (res.score >= 0.85) {
                colorClass = 'color-success';
                bgClass = 'bg-success';
              } else if (res.score >= 0.60) {
                colorClass = 'color-warning';
                bgClass = 'bg-warning';
              }

              const ns = res.metadata?.namespace || 'docs';
              const src = res.metadata?.source || 'unknown';
              const page = res.metadata?.page || 1;
              const metaString = `${res.id} · ${ns} · page ${page} · ${src}`;

              return (
                <div key={index} className="result-card" style={{ animationDelay: `${index * 60}ms` }}>
                  <div className="card-top">
                    <span className="card-rank">#{index + 1}</span>
                    <div className="similarity-bar-container">
                      <div 
                        className={`similarity-bar ${bgClass}`} 
                        style={{ width: isSearching ? '0%' : `${res.score * 100}%` }}
                      ></div>
                    </div>
                    <span className={`card-score ${colorClass}`}>{res.score.toFixed(4)}</span>
                  </div>
                  <div className="card-meta">{metaString}</div>
                  <div className="card-text">{res.text || '...'}</div>
                </div>
              );
            })}
          </div>
        </main>

        {/* Footer */}
        <footer className="footer">
          <div className="stat">QUERIES: {stats.queries}</div>
          <div className="stat">AVG LATENCY: {stats.avgLatency.toFixed(1)}ms</div>
          <div className="stat">VECTORS IN MEMORY: {vectorCount}</div>
        </footer>
      </div>
    </>
  );
}

export default App;
