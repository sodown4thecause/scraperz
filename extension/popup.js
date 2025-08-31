document.addEventListener('DOMContentLoaded', function() {
  const scrapeButton = document.getElementById('scrape-button');
  const urlInput = document.getElementById('url');
  const promptInput = document.getElementById('prompt');
  const loadingDiv = document.getElementById('loading');
  const resultsContainer = document.getElementById('results-container');
  const resultsPre = document.getElementById('results');

  let pollingInterval;

  // Load the last used URL and prompt from storage
  chrome.storage.local.get(['lastUrl', 'lastPrompt'], function(result) {
    if (result.lastUrl) {
      urlInput.value = result.lastUrl;
    }
    if (result.lastPrompt) {
      promptInput.value = result.lastPrompt;
    }
  });

  // Set the URL to the current tab's URL by default
  chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
    if (tabs[0] && tabs[0].url) {
      urlInput.value = tabs[0].url;
    }
  });

  function pollJobStatus(jobId) {
    pollingInterval = setInterval(() => {
      fetch(`http://localhost:8000/jobs/${jobId}/status`)
        .then(response => {
          if (!response.ok) {
            throw new Error('Failed to get job status.');
          }
          return response.json();
        })
        .then(data => {
          loadingDiv.textContent = `Status: ${data.status}...`;
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(pollingInterval);
            loadingDiv.classList.add('hidden');
            resultsContainer.classList.remove('hidden');
            resultsPre.textContent = JSON.stringify(data.results.length > 0 ? data.results[0].data : { info: 'No results found.' }, null, 2);
          }
        })
        .catch(error => {
          clearInterval(pollingInterval);
          loadingDiv.textContent = `Error polling status: ${error.message}`;
        });
    }, 2000); // Poll every 2 seconds
  }

  scrapeButton.addEventListener('click', function() {
    const url = urlInput.value;
    const prompt = promptInput.value;
    const userId = 'extension-user'; // A static user ID for now

    if (!url || !prompt) {
      alert('Please provide both a URL and a prompt.');
      return;
    }

    // Save the URL and prompt for next time
    chrome.storage.local.set({ lastUrl: url, lastPrompt: prompt });

    loadingDiv.textContent = 'Initiating scrape...';
    loadingDiv.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    if(pollingInterval) clearInterval(pollingInterval);

    fetch('http://localhost:8000/scrape', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: url,
        prompt: prompt,
        user_id: userId,
      }),
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(err => { throw new Error(err.detail || 'Network response was not ok'); });
      }
      return response.json();
    })
    .then(data => {
      if (data.job_id) {
        loadingDiv.textContent = 'Scraping job started... Polling for status.';
        pollJobStatus(data.job_id);
      } else {
        // Handle cases where job_id is not returned (e.g. cached result)
        loadingDiv.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        resultsPre.textContent = JSON.stringify(data, null, 2);
      }
    })
    .catch(error => {
      loadingDiv.classList.add('hidden');
      resultsContainer.classList.remove('hidden');
      resultsPre.textContent = 'Error: ' + error.message;
      console.error('There was a problem with the fetch operation:', error);
    });
  });
});