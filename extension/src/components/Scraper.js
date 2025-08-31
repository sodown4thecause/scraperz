import React, { useState, useEffect } from 'react';

function Scraper() {
  const [url, setUrl] = useState('');
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);

  useEffect(() => {
    // Get the current tab's URL
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0] && tabs[0].url) {
        setUrl(tabs[0].url);
      }
    });

    // Load last used prompt from storage
    chrome.storage.local.get(['lastPrompt'], (res) => {
      if (res.lastPrompt) {
        setPrompt(res.lastPrompt);
      }
    });
  }, []);

  useEffect(() => {
    let pollInterval;
    if (jobId && (status === 'running' || status === 'pending')) {
      pollInterval = setInterval(() => {
        fetch(`http://localhost:8000/jobs/${jobId}/status`)
          .then(res => res.json())
          .then(data => {
            setStatus(data.status);
            if (data.status === 'completed' || data.status === 'failed') {
              setLoading(false);
              setResult(data.results.length > 0 ? data.results[0].data : { info: 'No results found.' });
              clearInterval(pollInterval);
            }
          })
          .catch(err => {
            setLoading(false);
            setStatus('Error polling status');
            console.error(err);
            clearInterval(pollInterval);
          });
      }, 2000);
    }
    return () => clearInterval(pollInterval);
  }, [jobId, status]);

  const handleAiAction = useCallback((action, params = {}) => {
    if (!result) return;

    setAiLoading(true);
    setAiResult(null);
    setAiAction(action);

    // Extract text content from the result object for processing
    const contentToProcess = typeof result === 'object' ? JSON.stringify(result) : result;

    fetch('http://localhost:8000/process_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        content: contentToProcess,
        action: action,
        params: params
      }),
    })
    .then(res => {
      if (!res.ok) {
        throw new Error('AI processing failed');
      }
      return res.json();
    })
    .then(data => {
      setAiResult(data);
      setAiLoading(false);
    })
    .catch(err => {
      console.error(err);
      setAiResult({ error: err.message });
      setAiLoading(false);
    });
  }, [result]);

  const handleScrape = () => {
    setLoading(true);
    setStatus('Initiating scrape...');
    setResult(null);
    chrome.storage.local.set({ lastPrompt: prompt });

    fetch('http://localhost:8000/scrape', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, prompt, user_id: 'react-extension-user' }),
    })
    .then(res => res.json())
    .then(data => {
      if (data.job_id) {
        setJobId(data.job_id);
        setStatus('running');
      } else {
        setLoading(false);
        setResult(data);
      }
    })
    .catch(err => {
      setLoading(false);
      setStatus('Error starting scrape');
      console.error(err);
    });
  };

  return (
    <div>
      <div className="input-group">
        <label>URL:</label>
        <input type="text" value={url} onChange={(e) => setUrl(e.target.value)} />
      </div>
      <div className="input-group">
        <label>Prompt:</label>
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
      </div>
      <button onClick={handleScrape} disabled={loading}>Scrape</button>
      {loading && <div id="loading">{status}</div>}
      {result && (
        <div id="results-container">
          <h2>Results:</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default Scraper;
   )}
          </div>
        </div>
      )}
    </div>
  );
}

export default Scraper;
