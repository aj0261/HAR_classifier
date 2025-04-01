document.addEventListener('DOMContentLoaded', () => {
  // --- Configuration ---
  // Use window.location.hostname and port if accessing from the network
  // Use 'localhost' or '127.0.0.1' if accessing locally on the server machine
  const serverAddress = `http://${window.location.hostname}:${window.location.port}`; // Assumes served by Flask

  // --- DOM Elements ---
  const connectionStatusEl = document.getElementById('connection-status');
  const predictorStatusEl = document.getElementById('predictor-status');
  const clientSidEl = document.getElementById('client-sid');
  const activityEl = document.getElementById('prediction-activity');
  const confidenceEl = document.getElementById('prediction-confidence');
  const modelEl = document.getElementById('prediction-model');
  const timestampEl = document.getElementById('prediction-timestamp');
  const durationEl = document.getElementById('prediction-duration');
  const errorEl = document.getElementById('prediction-error');
  const totalPredictionsEl = document.getElementById('total-predictions');
  const activityCountsEl = document.getElementById('activity-counts');

  // --- State ---
  let predictionCount = 0;
  const activityCounts = {}; // { 'ActivityName': count }

  // --- Socket.IO Connection ---
  console.log(`Attempting to connect to server at ${serverAddress}...`);
  const socket = io(serverAddress); // Connect to the Flask-SocketIO server

  // --- Socket Event Handlers ---
  socket.on('connect', () => {
      console.log('Connected to server!', socket.id);
      connectionStatusEl.textContent = 'Connected';
      connectionStatusEl.className = 'status-connected';
      clientSidEl.textContent = socket.id; // Show client's *browser* SID
  });

  socket.on('disconnect', (reason) => {
      console.log('Disconnected from server:', reason);
      connectionStatusEl.textContent = `Disconnected (${reason})`;
      connectionStatusEl.className = 'status-disconnected';
      predictorStatusEl.textContent = 'Unknown';
      predictorStatusEl.className = 'status-unknown';
      clientSidEl.textContent = 'N/A';
  });

  socket.on('connect_error', (error) => {
      console.error('Connection Error:', error);
      connectionStatusEl.textContent = 'Connection Failed';
      connectionStatusEl.className = 'status-error';
  });

  // Custom event from server acknowledging connection
  socket.on('connection_ack', (data) => {
      console.log('Connection acknowledged by server:', data);
      // Can use data.sid if needed (this would be the server's view of the SID)
  });

   // Custom event for predictor status
   socket.on('predictor_status', (data) => {
      console.log('Predictor status update:', data);
      if (data.ready) {
          predictorStatusEl.textContent = 'Ready';
          predictorStatusEl.className = 'status-connected';
      } else {
          predictorStatusEl.textContent = 'Not Ready';
          predictorStatusEl.className = 'status-disconnected';
      }
  });

  // Handle incoming predictions
  socket.on('prediction', (data) => {
      console.log('Prediction received:', data);
      clearError(); // Clear any previous error message
      activityMap = {
        "A": "Walking",
        "B": "jogging",
        "C": "using stairs",
        "D": "sitting",
        "E": "standing",
        "Walking": "Walking",
        "Stairs": "Using Stairs",
      }
      activityEl.textContent = activityMap[data.activity] || '--';
      modelEl.textContent = data.model_used || '--';
      timestampEl.textContent = data.timestamp_ms ? new Date(data.timestamp_ms).toLocaleString() : '--';
      durationEl.textContent = data.processing_time_ms ? data.processing_time_ms.toFixed(1) : '--';

      if (data.confidence !== null && data.confidence !== undefined) {
           confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;
      } else {
           confidenceEl.textContent = 'N/A'; // For SVM or errors
      }

      // Update insights
      predictionCount++;
      totalPredictionsEl.textContent = predictionCount;
      updateActivityCounts(data.activity);
  });

  // Handle prediction errors from the server
  socket.on('prediction_error', (data) => {
      console.error('Prediction Error from server:', data);
      displayError(data.error || 'Unknown prediction error');
      // Clear previous prediction details
      activityEl.textContent = 'Error';
      confidenceEl.textContent = '--';
      modelEl.textContent = data.model_used || 'None';
      timestampEl.textContent = data.timestamp_ms ? new Date(data.timestamp_ms).toLocaleString() : '--';
      durationEl.textContent = data.processing_time_ms ? data.processing_time_ms.toFixed(1) : '--';
  });

  // --- UI Update Functions ---
  function displayError(message) {
      errorEl.textContent = message;
      errorEl.style.display = 'block'; // Make sure it's visible
  }

  function clearError() {
      errorEl.textContent = 'None';
      // errorEl.style.display = 'none'; // Or hide it
  }

  function updateActivityCounts(activityName) {
      if (!activityName || activityName === 'Error') return; // Don't count errors

      activityCounts[activityName] = (activityCounts[activityName] || 0) + 1;

      // Update the list display
      activityCountsEl.innerHTML = ''; // Clear previous list
      const sortedActivities = Object.entries(activityCounts).sort(([,a],[,b]) => b-a); // Sort desc

      sortedActivities.forEach(([name, count]) => {
          const li = document.createElement('li');
          li.textContent = `${name}: ${count}`;
          activityCountsEl.appendChild(li);
      });

      // Optional: Update a chart here
      // updateChart();
  }

  // --- Optional: Chart.js Integration ---
  /*
  let activityChart = null;
  function setupChart() {
      const ctx = document.getElementById('activityChart').getContext('2d');
      activityChart = new Chart(ctx, {
          type: 'doughnut', // or 'bar'
          data: {
              labels: [], // Activity names
              datasets: [{
                  label: 'Activity Distribution',
                  data: [], // Counts
                  backgroundColor: [ // Add more colors as needed
                      'rgba(255, 99, 132, 0.7)',
                      'rgba(54, 162, 235, 0.7)',
                      'rgba(255, 206, 86, 0.7)',
                      'rgba(75, 192, 192, 0.7)',
                      'rgba(153, 102, 255, 0.7)',
                      'rgba(255, 159, 64, 0.7)'
                  ],
                  borderColor: [
                      'rgba(255, 99, 132, 1)',
                      'rgba(54, 162, 235, 1)',
                      'rgba(255, 206, 86, 1)',
                      'rgba(75, 192, 192, 1)',
                      'rgba(153, 102, 255, 1)',
                      'rgba(255, 159, 64, 1)'
                  ],
                  borderWidth: 1
              }]
          },
          options: {
              responsive: true,
              plugins: {
                  legend: {
                      position: 'top',
                  },
                  title: {
                      display: true,
                      text: 'Activity Distribution'
                  }
              }
          }
      });
  }

  function updateChart() {
      if (!activityChart) return;
      const labels = Object.keys(activityCounts);
      const data = Object.values(activityCounts);
      activityChart.data.labels = labels;
      activityChart.data.datasets[0].data = data;
      activityChart.update();
  }

  // setupChart(); // Call this once if using charts
  */

}); // End DOMContentLoaded