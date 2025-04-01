import os
import pandas as pd
import numpy as np
import time
import joblib # For loading scaler and label encoder
import tensorflow as tf # For loading TCN model
from collections import deque # For efficient buffering
# from scipy.stats import iqr # No longer needed
# from scipy.fft import rfft, rfftfreq # No longer needed
from sklearn.preprocessing import StandardScaler # Need the class definition
from sklearn.preprocessing import LabelEncoder # Need the class definition
# from sklearn.svm import SVC # No longer needed

from flask import Flask
from flask_socketio import SocketIO, emit
from flask import request # Import request object to get session ID

app = Flask(__name__)
# Use secret key for session management if needed
app.config['SECRET_KEY'] = 'your_secret_key_tcn!' # Replace with a real secret key
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Configuration & Constants ---
# --- Paths relative to this script ---
# <<<--- IMPORTANT: Update these paths to where your TCN results are saved --->>>
MODEL_DIR = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\TCN_newResults" # Directory containing TCN model, scaler, encoder
MODEL_FILENAME = os.path.join(MODEL_DIR, 'tcn_har_model.keras') # TCN model file (.keras format)
SCALER_FILENAME = os.path.join(MODEL_DIR, 'tcn_scaler.joblib') # Scaler from TCN training
LABEL_ENCODER_FILENAME = os.path.join(MODEL_DIR, 'tcn_label_encoder.joblib') # LabelEncoder from TCN training
activity_map = {
    "A": "Walking",
    "B": "jogging",
    "C": "using stairs",
    "D": "Sitting",
    "E": "Standing",
}
# --- Windowing Parameters (MUST MATCH TCN TRAINING SCRIPT) ---
SAMPLING_RATE_HZ = 20 # Assuming same as training
WINDOW_SIZE_SAMPLES = 60 # From TCN training script (WINDOW_SIZE)
STRIDE_SAMPLES = 15 # From TCN training script (STRIDE)
NUM_FEATURES = 6 # accel_x/y/z, gyro_x/y/z

# Check if windowing parameters make sense
if STRIDE_SAMPLES <= 0:
    print("⚠️ WARNING: STRIDE_SAMPLES calculated as <= 0. Predictions might happen too frequently or not at all.")
    print("   Using STRIDE_SAMPLES = 1 as a fallback.")
    STRIDE_SAMPLES = 1 # Fallback to avoid zero/negative stride

# --- Global Variables ---
tcn_model = None
scaler = None
label_encoder = None
# Store the buffer associated with a specific client (session ID)
client_buffers = {}
# Keep track of samples received since last prediction for each client
samples_since_last_pred = {}
# Feature order expected by the TCN model and scaler (based on TCN training script)
# The incoming data keys are different, we need to map them correctly.
MODEL_FEATURE_ORDER = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']
# Map incoming keys (from phone app) to the order the model expects
INCOMING_TO_MODEL_MAP = {
    'accel_x': 'x_accel',
    'accel_y': 'y_accel',
    'accel_z': 'z_accel',
    'gyro_x': 'x_gyro',
    'gyro_y': 'y_gyro',
    'gyro_z': 'z_gyro'
}
# Order to extract from incoming data dict to match MODEL_FEATURE_ORDER
EXTRACTION_ORDER = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']


# --- Load Model, Scaler, and Label Encoder ---
def load_model_scaler_encoder():
    """Load the trained TCN model, scaler, and label encoder."""
    global tcn_model, scaler, label_encoder
    print("\n--- Loading TCN Prediction Components ---")
    try:
        # 1. Load TCN Model
        print(f"Attempting to load model from: {MODEL_FILENAME}")
        if not os.path.exists(MODEL_FILENAME):
             raise FileNotFoundError(f"Model file not found at {MODEL_FILENAME}")
        tcn_model = tf.keras.models.load_model(MODEL_FILENAME)
        print(f"  TCN Model loaded successfully.")
        # Optionally print model summary
        # tcn_model.summary()

        # 2. Load Scaler
        print(f"Attempting to load scaler from: {SCALER_FILENAME}")
        if not os.path.exists(SCALER_FILENAME):
            raise FileNotFoundError(f"Scaler file not found at {SCALER_FILENAME}")
        scaler = joblib.load(SCALER_FILENAME)
        print(f"  Scaler loaded successfully.")
        if hasattr(scaler, 'n_features_in_'):
             print(f"   Scaler expects {scaler.n_features_in_} features.")
        if scaler.n_features_in_ != NUM_FEATURES:
             print(f"⚠️ WARNING: Scaler expected {scaler.n_features_in_} features, but NUM_FEATURES is set to {NUM_FEATURES}.")


        # 3. Load Label Encoder
        print(f"Attempting to load label encoder from: {LABEL_ENCODER_FILENAME}")
        if not os.path.exists(LABEL_ENCODER_FILENAME):
             raise FileNotFoundError(f"Label encoder file not found at {LABEL_ENCODER_FILENAME}")
        label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        print(f"  Label Encoder loaded successfully.")
        print(f"   Classes known by encoder: {label_encoder.classes_}")
        print("-----------------------------------------\n")


    except FileNotFoundError as fnf_error:
        print(f"❌ Error: {fnf_error}")
        print("   Please ensure the TCN training script ran successfully and the files")
        print(f"   '{os.path.basename(MODEL_FILENAME)}', '{os.path.basename(SCALER_FILENAME)}', and '{os.path.basename(LABEL_ENCODER_FILENAME)}'")
        print(f"   are present in the '{MODEL_DIR}' directory relative to this script.")
        exit() # Exit if essential components can't be loaded
    except Exception as e:
        print(f"❌ Error loading model, scaler, or encoder: {e}")
        exit()

# --- Flask Routes ---
@app.route('/')
def index():
    return "HAR TCN Real-time Predictor is running."

# --- WebSocket Events ---
@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    client_sid = request.sid # Get the unique session ID for this client
    print(f"\n📱 Mobile client connected! SID: {client_sid}")

    # Initialize buffer and counter for this specific client
    # Using a standard list; will manage size manually for window extraction
    client_buffers[client_sid] = []
    samples_since_last_pred[client_sid] = 0

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
    """Handle incoming sensor data and perform prediction using TCN model"""
    client_sid = request.sid

    # Ensure buffer exists for this client
    if client_sid not in client_buffers:
        print(f"⚠️ Warning: Received data from unknown SID {client_sid}. Ignoring.")
        # Optionally, reconnect or re-initialize here if needed
        # handle_connect() # Be careful with unintended side effects
        return

    if tcn_model is None or scaler is None or label_encoder is None:
        print(f"⏳ Model/Scaler/Encoder not loaded yet. Waiting for SID {client_sid[-5:]}...")
        # Optionally emit a waiting status
        # socketio.emit('status', {'message': 'Initializing...'}, room=client_sid)
        time.sleep(0.1) # Small delay
        return

    try:
        # 1. Extract Sensor Values in the Correct Order
        # Ensure all expected keys are present in the incoming data
        if not all(key in data_point for key in EXTRACTION_ORDER):
            missing_keys = [key for key in EXTRACTION_ORDER if key not in data_point]
            print(f"❌ Invalid data point received from SID {client_sid}: Missing keys {missing_keys}. Data: {data_point}")
            return

        # Extract values according to the order needed by the model/scaler
        sensor_values_ordered = [data_point[key] for key in EXTRACTION_ORDER]

        # Optionally include timestamp if needed for debugging or advanced logic
        timestamp_ms = data_point.get('timestamp', time.time() * 1000) # Use provided timestamp or current time

        # Store sensor data only (order matters!)
        data_to_store = sensor_values_ordered
        client_buffers[client_sid].append(data_to_store)
        samples_since_last_pred[client_sid] += 1

        current_buffer = client_buffers[client_sid]
        buffer_len = len(current_buffer)

        # --- Prediction Logic ---
        # Check if we have enough data for a full window AND enough new samples since last prediction
        if buffer_len >= WINDOW_SIZE_SAMPLES and samples_since_last_pred[client_sid] >= STRIDE_SAMPLES:
            start_pred_time = time.time()

            # 2. Get the latest window data (list of lists)
            window_data_list = current_buffer[-WINDOW_SIZE_SAMPLES:]

            # 3. Convert to NumPy array (Shape: [WINDOW_SIZE_SAMPLES, NUM_FEATURES])
            window_np = np.array(window_data_list, dtype=np.float32)

            if window_np.shape != (WINDOW_SIZE_SAMPLES, NUM_FEATURES):
                print(f"❌ Error: Window data shape mismatch for SID {client_sid}. Expected {(WINDOW_SIZE_SAMPLES, NUM_FEATURES)}, Got {window_np.shape}. Skipping prediction.")
                # Reset counter even on error to avoid repeated attempts with bad data
                samples_since_last_pred[client_sid] = 0
                # Consider trimming buffer if it grows too large due to errors
                if buffer_len > WINDOW_SIZE_SAMPLES * 2:
                     keep_index = buffer_len - WINDOW_SIZE_SAMPLES # Keep just one window size
                     client_buffers[client_sid] = current_buffer[keep_index:]
                return


            # 4. Scale the window data
            # Scaler expects [n_samples, n_features], which matches window_np shape
            try:
                scaled_window = scaler.transform(window_np)
            except Exception as e:
                 print(f"❌ Error during scaling for SID {client_sid}: {e}")
                 samples_since_last_pred[client_sid] = 0 # Reset counter
                 return # Stop processing this window

            # 5. Reshape for TCN Model input
            # TCN expects [batch_size, timesteps, features]
            # Here, batch_size=1, timesteps=WINDOW_SIZE_SAMPLES
            scaled_window_reshaped = scaled_window.reshape(1, WINDOW_SIZE_SAMPLES, NUM_FEATURES)

            # 6. Predict using TCN Model
            try:
                pred_proba = tcn_model.predict(scaled_window_reshaped, verbose=0) # verbose=0 suppresses progress bar
                # pred_proba shape is [1, num_classes]

                # 7. Get the predicted class index
                prediction_index = np.argmax(pred_proba, axis=1)[0]

                # 8. Convert index to activity label
                prediction_label = label_encoder.inverse_transform([prediction_index])[0]
                prediction_confidence = pred_proba[0][prediction_index] # Confidence of the chosen class

                end_pred_time = time.time()
                pred_duration = (end_pred_time - start_pred_time) * 1000 # ms

                # 9. Emit Prediction to client
                print(f"SID {client_sid[-5:]}: Pred -> {prediction_label} ({prediction_confidence*100:.1f}%) | Took {pred_duration:.1f} ms")
                socketio.emit('prediction', {
                    'activity': prediction_label,
                    'confidence': float(prediction_confidence) # Ensure JSON serializable
                    }, room=client_sid)

                # 10. Reset sample counter for this client
                samples_since_last_pred[client_sid] = 0

            except Exception as e:
                print(f"❌ Error during TCN prediction or label conversion for SID {client_sid}: {e}")
                samples_since_last_pred[client_sid] = 0 # Reset counter


        # Optional: Trim buffer periodically to prevent excessive memory usage
        # Keep slightly more than one window to ensure overlap is always possible
        max_buffer_len = WINDOW_SIZE_SAMPLES + STRIDE_SAMPLES * 2 # Heuristic
        if buffer_len > max_buffer_len:
             keep_index = buffer_len - WINDOW_SIZE_SAMPLES # Keep exactly one window's worth
             client_buffers[client_sid] = current_buffer[keep_index:]
             # print(f"DBG: Trimmed buffer for SID {client_sid} to {len(client_buffers[client_sid])} samples.")


    except KeyError as e:
        print(f"❌ Invalid data point received from SID {client_sid}: Missing key {e}. Data: {data_point}")
    except Exception as e:
        print(f"❌ Unexpected error handling sensor data for SID {client_sid}: {e}")
        # Consider resetting counter or clearing buffer in case of persistent errors
        # samples_since_last_pred[client_sid] = 0
        # client_buffers[client_sid] = []


if __name__ == '__main__':
    print("\n🚀 Starting HAR TCN Real-time Predictor on 0.0.0.0:80")
    # --- IMPORTANT ---
    # Ensure TensorFlow uses CPU if GPU is causing issues or not intended
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # tf.config.set_visible_devices([], 'GPU') # Alternative TF 2.x way

    load_model_scaler_encoder() # Load model, scaler, and encoder at startup

    if tcn_model is None or scaler is None or label_encoder is None:
         print("❌ Exiting due to essential component loading failure.")
    else:
        print("\n  Connect your Android app. Waiting for sensor data...")
        # Use SocketIO's development server
        # debug=False is recommended for stability, especially with background threads
        socketio.run(app, host='0.0.0.0', port=80, debug=False, use_reloader=False)