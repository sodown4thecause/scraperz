import React, { useState, useEffect } from 'react';

function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/history/react-extension-user')
      .then(res => res.json())
      .then(data => {
        setHistory(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div>Loading history...</div>;
  }

  return (
    <div className="history-list">
      {history.length === 0 ? (
        <p>No history found.</p>
      ) : (
        history.map(job => (
          <div key={job.id} className="history-item">
            <p><strong>URL:</strong> {job.url}</p>
            <p><strong>Status:</strong> {job.status}</p>
            <p><strong>Date:</strong> {new Date(job.created_at).toLocaleString()}</p>
          </div>
        ))
      )}
    </div>
  );
}

export default History;
