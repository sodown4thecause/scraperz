import React from 'react';
import './ScrapingAnimation.css';

const ScrapingAnimation = () => {
  return (
    <div className="animation-container">
      <div className="globe">
        <div className="globe-overlay"></div>
        <div className="data-point data-point-1"></div>
        <div className="data-point data-point-2"></div>
        <div className="data-point data-point-3"></div>
        <div className="data-point data-point-4"></div>
      </div>
      <div className="processor">
        <div className="line line-1"></div>
        <div className="line line-2"></div>
        <div className="line line-3"></div>
        <div className="line line-4"></div>
        <div className="core"></div>
      </div>
      <div className="output">
        <div className="data-card"></div>
        <div className="data-card"></div>
        <div className="data-card"></div>
      </div>
    </div>
  );
};

export default ScrapingAnimation;
