import os
import pandas as pd
import numpy as np
import time
import joblib # For loading model/scaler
from collections import deque # For efficient buffering
from scipy.stats import iqr
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler # Need the class definition
from sklearn.svm import SVC # Need the class definition

from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
# Use secret key for session management if needed, though not strictly required for this basic example
app.config['SECRET_KEY'] = 'your_secret_key!' # Replace with a real secret key
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Configuration & Constants ---
MODEL_FILENAME = 'models\svm_har_model.joblib'
SCALER_FILENAME = 'models\scaler.joblib'

SAMPLING_RATE_HZ = 20
WINDOW_DURATION_SEC = 3.0 # Should match training
WINDOW_OVERLAP = 0.9 # Should match training

WINDOW_SAMPLES = int(SAMPLING_RATE_HZ * WINDOW_DURATION_SEC)
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP))

# Map model output (0, 1) back to labels
LABEL_MAP = {0: "Walking", 1: "Stairs"}

# --- Global Variables ---
svm_model = None
scaler = None
# Use deque for efficient fixed-size buffer, initialize per connection
# data_buffer = deque(maxlen=WINDOW_SAMPLES)
# Store the buffer associated with a specific client (session ID)
client_buffers = {}
# Keep track of samples received since last prediction for each client
samples_since_last_pred = {}


# --- Feature Extraction Function (Identical to training script) ---
def extract_features(window_df):
    """Extracts features from a window of sensor data."""
    features = []
    if window_df.empty or len(window_df) < 10:
         # Expected features: 6 axes * 7 stats + 2 resultant + 3 correlation + 3*2 FFT = 42 + 2 + 3 + 6 = 53
         return np.full(53, np.nan) # Adjust size if features change

    axes = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for axis in axes:
        data = window_df[axis]
        features.extend([
            data.mean(), data.std(), data.var(), data.min(), data.max(), data.median(), iqr(data)
        ])

    accel_mag = np.sqrt((window_df[['accel_x', 'accel_y', 'accel_z']]**2).sum(axis=1))
    features.extend([accel_mag.mean(), accel_mag.std()])

    # Correlation
    if window_df['accel_x'].std() > 0 and window_df['accel_y'].std() > 0: features.append(window_df['accel_x'].corr(window_df['accel_y']))
    else: features.append(0)
    if window_df['accel_x'].std() > 0 and window_df['accel_z'].std() > 0: features.append(window_df['accel_x'].corr(window_df['accel_z']))
    else: features.append(0)
    if window_df['accel_y'].std() > 0 and window_df['accel_z'].std() > 0: features.append(window_df['accel_y'].corr(window_df['accel_z']))
    else: features.append(0)

    # FFT
    N = len(window_df)
    yf_mag = np.abs(rfft(accel_mag.values))
    xf_freq = rfftfreq(N, 1 / SAMPLING_RATE_HZ)
    if len(xf_freq) > 1:
        dominant_freq_idx = np.argmax(yf_mag[1:]) + 1
        features.extend([xf_freq[dominant_freq_idx], np.sum(yf_mag**2) / N])
    else: features.extend([0, 0])

    yf_mag_y = np.abs(rfft(window_df['accel_y'].values))
    if len(xf_freq) > 1:
        dominant_freq_idx_y = np.argmax(yf_mag_y[1:]) + 1
        features.extend([xf_freq[dominant_freq_idx_y], np.sum(yf_mag_y**2) / N])
    else: features.extend([0, 0])

    yf_mag_z = np.abs(rfft(window_df['accel_z'].values))
    if len(xf_freq) > 1:
        dominant_freq_idx_z = np.argmax(yf_mag_z[1:]) + 1
        features.extend([xf_freq[dominant_freq_idx_z], np.sum(yf_mag_z**2) / N])
    else: features.extend([0, 0])

    return np.array(features) # Ensure returning numpy array

# --- Load Model and Scaler ---
def load_model_scaler():
    """Load the trained SVM model and scaler."""
    global svm_model, scaler
    try:
        svm_model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        print(f"  Model ({MODEL_FILENAME}) and Scaler ({SCALER_FILENAME}) loaded successfully.")
        # Print some model info if possible
        if hasattr(svm_model, 'kernel'):
             print(f"   Model Type: SVC (kernel={svm_model.kernel}, C={svm_model.C})")
        if hasattr(scaler, 'mean_'):
             print(f"   Scaler fitted with {len(scaler.mean_)} features.")

    except FileNotFoundError:
        print(f"❌ Error: Model ('{MODEL_FILENAME}') or Scaler ('{SCALER_FILENAME}') file not found.")
        print("   Please train the model and save both files in the same directory as this script.")
        exit() # Exit if model/scaler can't be loaded
    except Exception as e:
        print(f"❌ Error loading model or scaler: {e}")
        exit()

# --- Flask Routes ---
@app.route('/')
def index():
    return "HAR Real-time Predictor is running."

# --- WebSocket Events ---
from flask import request # Import request object to get session ID

@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    client_sid = request.sid # Get the unique session ID for this client
    print(f"\n📱 Mobile client connected! SID: {client_sid}")

    # Initialize buffer and counter for this specific client
    # Use a standard list and manage its size manually for easier window extraction
    client_buffers[client_sid] = []
    samples_since_last_pred[client_sid] = 0

    # No need to request activity/subject info for prediction mode
    # socketio.emit('request_info', room=client_sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_sid = request.sid
    print(f"\n🚫 Mobile client disconnected! SID: {client_sid}")
    # Clean up resources for this client
    if client_sid in client_buffers:
        del client_buffers[client_sid]
    if client_sid in samples_since_last_pred:
        del samples_since_last_pred[client_sid]

@socketio.on('sensor_data')
def handle_sensor_data(data_point):
    """Handle incoming sensor data and perform prediction"""
    client_sid = request.sid

    # Ensure buffer exists for this client
    if client_sid not in client_buffers:
        # ... (error handling as before) ...
        return

    if svm_model is None or scaler is None:
        # ... (waiting message as before) ...
        return

    try:
        # --- MODIFICATION START: Store timestamp ---
        # Extract sensor values AND the timestamp
        timestamp_ms = data_point['timestamp'] # Get the timestamp sent by the app
        sensor_values_only = [
             data_point['accel_x'], data_point['accel_y'], data_point['accel_z'],
             data_point['gyro_x'], data_point['gyro_y'], data_point['gyro_z']
        ]
        # Store as a tuple: (timestamp, [sensor_values])
        data_to_store = (timestamp_ms, sensor_values_only)
        # --- MODIFICATION END ---

        # Append to this client's buffer
        client_buffers[client_sid].append(data_to_store) # Append the tuple
        samples_since_last_pred[client_sid] += 1

        current_buffer = client_buffers[client_sid]
        buffer_len = len(current_buffer)

        # --- Prediction Logic ---
        if buffer_len >= WINDOW_SAMPLES and samples_since_last_pred[client_sid] >= STEP_SAMPLES:

            # Get the latest WINDOW_SAMPLES (list of tuples)
            window_tuples = current_buffer[-WINDOW_SAMPLES:]

            # --- START: Time Gap Logging ---
            # Extract timestamps and sensor data separately
            window_timestamps = [item[0] for item in window_tuples]
            window_sensor_data = [item[1] for item in window_tuples]

            if window_timestamps: # Check if the list is not empty
                start_timestamp_ms = window_timestamps[0]
                end_timestamp_ms = window_timestamps[-1]
                time_gap_ms = end_timestamp_ms - start_timestamp_ms
                print(f"DBG: Window time gap: {time_gap_ms} ms (Samples: {len(window_timestamps)}, Start: {start_timestamp_ms}, End: {end_timestamp_ms})")
            # --- END: Time Gap Logging ---


            # Convert sensor data only to DataFrame for feature extraction
            window_df = pd.DataFrame(window_sensor_data, columns=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])

            # 1. Extract Features
            features = extract_features(window_df) # Pass only sensor data DF

            # ... (rest of the prediction logic: NaN check, scaling, prediction, emitting) ...
            # Check if features are valid (not all NaN)
            if not np.isnan(features).all():
                # 2. Scale Features (reshape for scaler)
                features_reshaped = features.reshape(1, -1)
                try:
                    features_scaled = scaler.transform(features_reshaped)

                    # 3. Predict
                    prediction_code = svm_model.predict(features_scaled)[0]
                    prediction_label = LABEL_MAP.get(prediction_code, "Unknown")

                    # 4. Emit Prediction to client
                    print(f"SID {client_sid[-5:]}: Prediction -> {prediction_label} ({prediction_code})")
                    socketio.emit('prediction', {'activity': prediction_label}, room=client_sid)

                    # 5. Reset counter for this client
                    samples_since_last_pred[client_sid] = 0

                except Exception as e:
                    print(f"❌ Error during scaling/prediction for SID {client_sid}: {e}")
                    samples_since_last_pred[client_sid] = 0
            else:
                print(f"⚠️ Warning: Feature extraction resulted in NaNs for SID {client_sid}. Skipping prediction.")
                samples_since_last_pred[client_sid] = 0


        # Optional: Trim buffer periodically
        # Needs slight modification to handle tuples
        if buffer_len > WINDOW_SAMPLES * 1.5:
             keep_index = buffer_len - int(WINDOW_SAMPLES * 1.5)
             client_buffers[client_sid] = current_buffer[keep_index:]


    except KeyError as e:
        print(f"❌ Invalid data point received: Missing key {e}. Data: {data_point}")
    except Exception as e:
        print(f"❌ Unexpected error handling sensor data for SID {client_sid}: {e}")


if __name__ == '__main__':
    print("\n🚀 Starting HAR Real-time Predictor on 0.0.0.0:80")
    load_model_scaler() # Load model and scaler at startup
    if svm_model is None or scaler is None:
         print("❌ Exiting due to model/scaler loading failure.")
    else:
        print("  Connect your Android app. Waiting for sensor data...")
        # Use SocketIO's development server
        socketio.run(app, host='0.0.0.0', port=80, debug=False) # debug=True can cause issues with some setups