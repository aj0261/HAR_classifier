# main_server.py
import time
from collections import deque
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np

# Import configuration and predictor class
import config
from prediction.predictor import Predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_123!' # Change this!
socketio = SocketIO(app, cors_allowed_origins="*") # Allow all for development

# --- Global Variables ---
# Initialize predictor (loads models on startup)
predictor = Predictor(config)

# Client data buffers and state
client_buffers = {} # { sid: deque([(ts, [values]), ...]) }
samples_since_last_pred = {} # { sid: count }

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main web UI page."""
    return render_template('index.html')

# --- WebSocket Events ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections."""
    client_sid = request.sid
    print(f"\n📱 Client connected: {client_sid}")
    # Initialize buffer using deque with maxlen based on window + buffer
    buffer_size = config.WINDOW_SIZE_SAMPLES + config.STRIDE_SAMPLES * 3 # Keep bit more than a window
    client_buffers[client_sid] = deque(maxlen=buffer_size)
    samples_since_last_pred[client_sid] = 0
    emit('connection_ack', {'status': 'Connected', 'sid': client_sid}, room=client_sid)
    # Send initial status about predictor readiness
    emit('predictor_status', {'ready': predictor.is_ready}, room=client_sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    client_sid = request.sid
    print(f"\n🚫 Client disconnected: {client_sid}")
    # Clean up resources
    if client_sid in client_buffers: del client_buffers[client_sid]
    if client_sid in samples_since_last_pred: del samples_since_last_pred[client_sid]

@socketio.on('sensor_data')
def handle_sensor_data(data_point):
    """Handles incoming sensor data and triggers prediction."""
    client_sid = request.sid

    # --- Input Validation and Buffering ---
    if client_sid not in client_buffers:
        print(f"⚠️ Data from unknown SID {client_sid}. Attempting re-init.")
        handle_connect() # Try to setup buffer again
        # Might still miss this data point, or process if connect is fast enough

    if not isinstance(data_point, dict):
        print(f"❌ Invalid data format from {client_sid}. Expected dict, got {type(data_point)}")
        return

    timestamp_ms = data_point.get('timestamp', time.time() * 1000) # Use provided or generate

    try:
        # Extract sensor values in the order defined by INCOMING_SENSOR_KEYS
        sensor_values = [data_point[key] for key in config.INCOMING_SENSOR_KEYS]
    except KeyError as e:
        print(f"❌ Invalid data point from {client_sid}: Missing key {e}. Data: {data_point}")
        return
    except Exception as e:
         print(f"❌ Error processing incoming data keys from {client_sid}: {e}")
         return

    # Store tuple (timestamp, list_of_values)
    client_buffers[client_sid].append((timestamp_ms, sensor_values))
    samples_since_last_pred[client_sid] += 1

    # --- Prediction Trigger Logic ---
    current_buffer = client_buffers[client_sid]
    buffer_len = len(current_buffer)

    if buffer_len >= config.WINDOW_SIZE_SAMPLES and \
       samples_since_last_pred[client_sid] >= config.STRIDE_SAMPLES:

        if not predictor.is_ready:
            print(f"⏳ Predictor not ready, skipping prediction for {client_sid}")
            # Reset counter to avoid predicting immediately when ready if buffer is full
            samples_since_last_pred[client_sid] = 0
            emit('prediction_error', {'error': 'Predictor not ready'}, room=client_sid)
            return

        # Get window data (most recent samples)
        # deque slicing is efficient, convert to list for Predictor
        window_tuples = list(current_buffer)[-config.WINDOW_SIZE_SAMPLES:]
        window_timestamps = [item[0] for item in window_tuples]
        window_sensor_data = [item[1] for item in window_tuples] # List of lists

        # --- Call Predictor ---
        prediction_result = predictor.predict_activity(window_sensor_data, window_timestamps)

        # --- Emit Result ---
        if prediction_result.get('error'):
            print(f"SID {client_sid[-5:]}: Prediction Error -> {prediction_result['error']} "
                  f"| Dur: {prediction_result['processing_time_ms']:.1f} ms")
            emit('prediction_error', prediction_result, room=client_sid)
        else:
            # --- CORRECTED APPROACH ---
            # 1. Calculate confidence string separately
            conf_str = 'N/A' # Default
            if prediction_result['confidence'] is not None:
                conf_str = f"{prediction_result['confidence'] * 100:.1f}%"

            # 2. Use the calculated string in the main print f-string
            print(f"SID {client_sid[-5:]}: Pred -> {prediction_result['activity']} "
                  f"(Conf: {conf_str}) " # Use the pre-formatted string here
                  f"| Model: {prediction_result['model_used']} "
                  f"| Time: {prediction_result['timestamp_ms']:.0f} "
                  f"| Dur: {prediction_result['processing_time_ms']:.1f} ms")
            # --- END CORRECTION ---
            socketio.emit('prediction', prediction_result)

        # Reset counter *after* attempting prediction
        samples_since_last_pred[client_sid] = 0


if __name__ == '__main__':
    server_host = config.SERVER_HOST
    server_port = config.SERVER_PORT
    print(f"\n🚀 Starting HAR Real-time Server [{server_host}:{server_port}]")

    if predictor.is_ready:
        print("\n  Predictor ready.")
        print("   Connect your mobile app or open the web UI:")
        # --- ADDED/MODIFIED LINES ---
        print(f"   - Localhost: http://localhost:{server_port}")
        print(f"   - Loopback:  http://127.0.0.1:{server_port}")
        print(f"   - Network:   Look for 'Running on http://<YOUR-IP>:{server_port}' below (use this for other devices)")
        # --- END ADDED/MODIFIED LINES ---
    else:
        print("\n⚠️ WARNING: Predictor failed to initialize. Check errors above.")
        print("   Server is running, but predictions will fail.")

    print("\nWaiting for connections...") # Added clarification

    # Use SocketIO's development server
    # use_reloader=False is important for stability when models are loaded globally
    # Werkzeug (Flask's development server) will print the actual bound addresses below this
    socketio.run(app, host=server_host, port=server_port, debug=False, use_reloader=False)