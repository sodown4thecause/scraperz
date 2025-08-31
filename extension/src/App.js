import React, { useState } from 'react';
import Scraper from './components/Scraper';
import History from './components/History';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('scraper');

  return (
    <div className="App">
      <div className="tabs">
        <button 
          className={`tab-button ${activeTab === 'scraper' ? 'active' : ''}`}
          onClick={() => setActiveTab('scraper')}
        >
          Scraper
        </button>
        <button 
          className={`tab-button ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          History
        </button>
      </div>
      <div className="tab-content">
        {activeTab === 'scraper' && <Scraper />}
        {activeTab === 'history' && <History />}
      </div>
    </div>
  );
}

export default App;
