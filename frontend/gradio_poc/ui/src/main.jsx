import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';

function App() {
  const [status, setStatus] = useState('checking...');
  useEffect(() => {
    fetch('/api/health')
      .then(response => { if (!response.ok) throw new Error('Health request failed'); return response.json(); })
      .then(data => setStatus(data.status))
      .catch(() => setStatus('unavailable'));
  }, []);
  return <main><h1>Relevia deployment test</h1><p>API status: <strong>{status}</strong></p></main>;
}

createRoot(document.getElementById('root')).render(<App />);
