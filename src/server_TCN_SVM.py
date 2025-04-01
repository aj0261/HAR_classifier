import os
import pandas as pd
import numpy as np
import time
import joblib # For loading scaler and label encoder AND SVM model/scaler
import tensorflow as tf # For loading TCN model
from collections import deque # For efficient buffering
from scipy.stats import iqr # Needed for SVM feature extraction
from scipy.fft import rfft, rfftfreq # Needed for SVM feature extraction
from sklearn.preprocessing import StandardScaler # Need the class definition
from sklearn.preprocessing import LabelEncoder # Need the class definition
from sklearn.svm import SVC # Need the class definition for loading SVM

from flask import Flask
from flask_socketio import SocketIO, emit
from flask import request # Import request object to get session ID

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_tcn_svm!' # Replace with a real secret key
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Configuration & Constants ---

# --- TCN Model Configuration ---
TCN_MODEL_DIR = r"C:\Users\gontu\OneDrive\Documents\HAR Classification\TCN_newResults" # Directory containing TCN model, scaler, encoder
TCN_MODEL_FILENAME = os.path.join(TCN_MODEL_DIR, 'tcn_har_model.keras')
TCN_SCALER_FILENAME = os.path.join(TCN_MODEL_DIR, 'tcn_scaler.joblib')
TCN_LABEL_ENCODER_FILENAME = os.path.join(TCN_MODEL_DIR, 'tcn_label_encoder.joblib')
TCN_NUM_FEATURES = 6 # accel_x/y/z, gyro_x/y/z
# TCN feature order (mapping from incoming keys)
TCN_MODEL_FEATURE_ORDER = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']
INCOMING_KEYS_FOR_TCN = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

# --- SVM Model Configuration ---
SVM_MODEL_DIR = 'models' # Directory containing SVM model and scaler
SVM_MODEL_FILENAME = os.path.join(SVM_MODEL_DIR, 'svm_har_model.joblib')
SVM_SCALER_FILENAME = os.path.join(SVM_MODEL_DIR, 'scaler.joblib')
SVM_NUM_FEATURES = 53 # Expected number of features for SVM (adjust if different)
# Columns needed for SVM feature extraction (matching incoming keys)
SVM_FEATURE_COLUMNS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
# <<<--- IMPORTANT: Update this map with your actual SVM class labels --->>>
SVM_LABEL_MAP = {0: "Walking", 1: "Stairs"} # Example: {0: "Walking", 1: "Stairs"}
# TCN classes that trigger the SVM model check
SVM_TRIGGER_CLASSES = ['A', 'C'] # Add other classes if needed

# --- Windowing Parameters (MUST MATCH BOTH TCN & SVM TRAINING) ---
SAMPLING_RATE_HZ = 20
# Window Size (choose one consistent value used for BOTH models during training)
# Assuming 3 seconds * 20 Hz = 60 samples was used for both
WINDOW_SIZE_SAMPLES = 60
# Stride (use the TCN stride, as it's likely determining the prediction frequency)
STRIDE_SAMPLES = 15 # From TCN training script (STRIDE)

# --- Sanity Checks ---
if len(SVM_FEATURE_COLUMNS) != TCN_NUM_FEATURES:
    print("⚠️ WARNING: Number of columns for SVM feature extraction differs from TCN input features.")
    print("   Ensure INCOMING_KEYS_FOR_TCN provides all necessary data for both.")
if len(INCOMING_KEYS_FOR_TCN) < max(len(SVM_FEATURE_COLUMNS), TCN_NUM_FEATURES):
     print("❌ ERROR: INCOMING_KEYS_FOR_TCN does not contain enough keys for both models.")
     exit()

# --- Global Variables ---
# TCN components
tcn_model = None
tcn_scaler = None
label_encoder = None
# SVM components
svm_model = None
svm_scaler = None
# Client data buffers
client_buffers = {}
samples_since_last_pred = {}
# Performance tracking
# last_prediction_time = {} # Optional: for explicit time-based throttling

# --- Feature Extraction Function (for SVM) ---
def extract_svm_features(window_df, expected_cols):
    """Extracts features for the SVM model."""
    # Check if required columns exist
    if not all(col in window_df.columns for col in expected_cols):
        missing = [col for col in expected_cols if col not in window_df.columns]
        print(f"❌ ERROR in extract_svm_features: Missing columns {missing}")
        return np.full(SVM_NUM_FEATURES, np.nan) # Return NaNs

    features = []
    if window_df.empty or len(window_df) < 10: # Basic check
         return np.full(SVM_NUM_FEATURES, np.nan)

    axes = expected_cols # Use the provided column names
    for axis in axes:
        data = window_df[axis]
        # Handle potential all-NaN slices which cause errors in stats
        if data.isnull().all():
            features.extend([np.nan] * 7) # Add NaNs for all stats
            continue
        try:
            features.extend([
                data.mean(skipna=True), data.std(skipna=True), data.var(skipna=True),
                data.min(skipna=True), data.max(skipna=True), data.median(skipna=True),
                iqr(data.dropna()) # iqr needs non-NaN data
            ])
        except Exception as e:
            print(f"⚠️ Warning extracting basic stats for {axis}: {e}")
            features.extend([np.nan] * 7)


    # Calculate magnitude using only accel columns present
    accel_cols = [col for col in ['accel_x', 'accel_y', 'accel_z'] if col in window_df.columns]
    if len(accel_cols) > 0:
        accel_mag = np.sqrt((window_df[accel_cols].fillna(0)**2).sum(axis=1)) # Fill NaN before squaring
        features.extend([accel_mag.mean(), accel_mag.std()])
    else:
        features.extend([np.nan, np.nan])


    # Correlation (check columns exist and have variance)
    def safe_corr(col1, col2):
        if col1 in window_df.columns and col2 in window_df.columns:
            data1 = window_df[col1].dropna()
            data2 = window_df[col2].dropna()
            # Check for sufficient data points and variance after dropping NaNs
            if len(data1) > 1 and len(data2) > 1 and data1.std() > 0 and data2.std() > 0:
                 # Align data in case NaNs were dropped differently
                aligned_data1, aligned_data2 = data1.align(data2, join='inner')
                if len(aligned_data1) > 1: # Ensure overlap exists
                    return aligned_data1.corr(aligned_data2)
        return 0.0 # Return 0 if calculation isn't possible

    features.append(safe_corr('accel_x', 'accel_y'))
    features.append(safe_corr('accel_x', 'accel_z'))
    features.append(safe_corr('accel_y', 'accel_z'))


    # FFT (handle potential NaNs and short data)
    N = len(window_df)
    if N == 0: # Prevent errors if window_df is empty after checks
        features.extend([np.nan] * 6) # Add NaNs for all FFT features
    else:
        xf_freq = rfftfreq(N, 1 / SAMPLING_RATE_HZ) if N > 0 else np.array([])

        def get_fft_features(data_series):
            if N < 2 or data_series.isnull().all(): return [np.nan, np.nan]
            values = data_series.fillna(0).values # Fill NaNs for FFT
            yf_mag = np.abs(rfft(values))
            # Avoid index error if xf_freq is too short
            if len(xf_freq) > 1 and len(yf_mag) > 1:
                dominant_freq_idx = np.argmax(yf_mag[1:]) + 1
                # Check if index is valid for xf_freq
                dom_freq = xf_freq[dominant_freq_idx] if dominant_freq_idx < len(xf_freq) else np.nan
                energy = np.sum(yf_mag**2) / N if N > 0 else np.nan
                return [dom_freq, energy]
            elif len(xf_freq) == 1 and len(yf_mag) >= 1: # Handle N=1 case (only DC component)
                 return [xf_freq[0], np.sum(yf_mag**2) / N if N > 0 else np.nan]
            else: # Handle N=0 or other unexpected cases
                return [np.nan, np.nan]

        # Calculate FFT features for accel mag, y, z if available
        if 'accel_mag' not in locals() or accel_mag.isnull().all():
             features.extend([np.nan, np.nan])
        else:
             features.extend(get_fft_features(accel_mag))

        if 'accel_y' not in window_df.columns: features.extend([np.nan, np.nan])
        else: features.extend(get_fft_features(window_df['accel_y']))

        if 'accel_z' not in window_df.columns: features.extend([np.nan, np.nan])
        else: features.extend(get_fft_features(window_df['accel_z']))


    result = np.array(features, dtype=float)
    if len(result) != SVM_NUM_FEATURES:
         print(f"❌ ERROR: Feature extraction produced {len(result)} features, expected {SVM_NUM_FEATURES}. Padding/truncating.")
         # Pad or truncate to the expected size (basic fix)
         if len(result) < SVM_NUM_FEATURES:
             result = np.pad(result, (0, SVM_NUM_FEATURES - len(result)), 'constant', constant_values=np.nan)
         else:
             result = result[:SVM_NUM_FEATURES]

    return result


# --- Load All Models and Scalers ---
def load_all_models_scalers():
    """Load TCN and SVM models, scalers, and TCN encoder."""
    global tcn_model, tcn_scaler, label_encoder, svm_model, svm_scaler
    success = True
    print("\n--- Loading Prediction Components ---")

    # 1. Load TCN Components
    try:
        print(f"Attempting to load TCN model from: {TCN_MODEL_FILENAME}")
        if not os.path.exists(TCN_MODEL_FILENAME): raise FileNotFoundError(f"TCN Model file not found at {TCN_MODEL_FILENAME}")
        tcn_model = tf.keras.models.load_model(TCN_MODEL_FILENAME)
        print(f"  TCN Model loaded successfully.")

        print(f"Attempting to load TCN scaler from: {TCN_SCALER_FILENAME}")
        if not os.path.exists(TCN_SCALER_FILENAME): raise FileNotFoundError(f"TCN Scaler file not found at {TCN_SCALER_FILENAME}")
        tcn_scaler = joblib.load(TCN_SCALER_FILENAME)
        print(f"  TCN Scaler loaded successfully.")
        if hasattr(tcn_scaler, 'n_features_in_') and tcn_scaler.n_features_in_ != TCN_NUM_FEATURES:
            print(f"⚠️ WARNING: TCN Scaler expected {tcn_scaler.n_features_in_} features, but config has {TCN_NUM_FEATURES}.")

        print(f"Attempting to load Label Encoder from: {TCN_LABEL_ENCODER_FILENAME}")
        if not os.path.exists(TCN_LABEL_ENCODER_FILENAME): raise FileNotFoundError(f"Label Encoder file not found at {TCN_LABEL_ENCODER_FILENAME}")
        label_encoder = joblib.load(TCN_LABEL_ENCODER_FILENAME)
        print(f"  Label Encoder loaded successfully.")
        print(f"   TCN Classes: {label_encoder.classes_}")

    except Exception as e:
        print(f"❌ Error loading TCN components: {e}")
        success = False

    # 2. Load SVM Components
    try:
        print(f"Attempting to load SVM model from: {SVM_MODEL_FILENAME}")
        if not os.path.exists(SVM_MODEL_FILENAME): raise FileNotFoundError(f"SVM Model file not found at {SVM_MODEL_FILENAME}")
        svm_model = joblib.load(SVM_MODEL_FILENAME)
        print(f"  SVM Model loaded successfully.")
        if hasattr(svm_model, 'kernel'): print(f"   SVM Type: SVC (kernel={svm_model.kernel})")


        print(f"Attempting to load SVM scaler from: {SVM_SCALER_FILENAME}")
        if not os.path.exists(SVM_SCALER_FILENAME): raise FileNotFoundError(f"SVM Scaler file not found at {SVM_SCALER_FILENAME}")
        svm_scaler = joblib.load(SVM_SCALER_FILENAME)
        print(f"  SVM Scaler loaded successfully.")
        if hasattr(svm_scaler, 'n_features_in_') and svm_scaler.n_features_in_ != SVM_NUM_FEATURES:
             print(f"⚠️ WARNING: SVM Scaler expected {svm_scaler.n_features_in_} features, but config expects {SVM_NUM_FEATURES} after extraction.")

    except Exception as e:
        print(f"❌ Error loading SVM components: {e}")
        success = False

    print("-------------------------------------\n")
    return success

# --- Flask Routes ---
@app.route('/')
def index():
    return "HAR TCN+SVM Real-time Predictor is running."

# --- WebSocket Events ---
@socketio.on('connect')
def handle_connect():
    client_sid = request.sid
    print(f"\n📱 Mobile client connected! SID: {client_sid}")
    client_buffers[client_sid] = deque(maxlen=(WINDOW_SIZE_SAMPLES + STRIDE_SAMPLES * 3)) # Use deque, slightly larger buffer
    samples_since_last_pred[client_sid] = 0
    # last_prediction_time[client_sid] = 0 # Optional: for time throttling

@socketio.on('disconnect')
def handle_disconnect():
    client_sid = request.sid
    print(f"\n🚫 Mobile client disconnected! SID: {client_sid}")
    if client_sid in client_buffers: del client_buffers[client_sid]
    if client_sid in samples_since_last_pred: del samples_since_last_pred[client_sid]
    # if client_sid in last_prediction_time: del last_prediction_time[client_sid] # Optional

@socketio.on('sensor_data')
def handle_sensor_data(data_point):
    client_sid = request.sid

    if client_sid not in client_buffers:
        print(f"⚠️ Warning: Received data from unknown SID {client_sid}. Re-initializing.")
        handle_connect() # Try to re-initialize buffer
        # return # Optional: skip processing this data point

    # Check if all models/scalers loaded
    if not all([tcn_model, tcn_scaler, label_encoder, svm_model, svm_scaler]):
        print(f"⏳ Models/Scalers not fully loaded yet. Waiting for SID {client_sid[-5:]}...")
        socketio.emit('status', {'message': 'Initializing...'}, room=client_sid)
        time.sleep(0.1)
        return

    try:
        # 1. Extract Data & Timestamp
        timestamp_ms = data_point.get('timestamp', time.time() * 1000) # Use provided or generate

        # Extract values based on INCOMING_KEYS_FOR_TCN, assuming they cover SVM needs too
        sensor_values_ordered = []
        for key in INCOMING_KEYS_FOR_TCN:
            if key not in data_point:
                 print(f"❌ Invalid data point: Missing key '{key}'. Data: {data_point}")
                 return # Skip this data point
            sensor_values_ordered.append(data_point[key])

        # Store as tuple: (timestamp, [sensor_values])
        data_to_store = (timestamp_ms, sensor_values_ordered)
        client_buffers[client_sid].append(data_to_store)
        samples_since_last_pred[client_sid] += 1

        current_buffer = client_buffers[client_sid]
        buffer_len = len(current_buffer)

        # --- Prediction Logic ---
        # Check buffer size and stride count
        if buffer_len >= WINDOW_SIZE_SAMPLES and samples_since_last_pred[client_sid] >= STRIDE_SAMPLES:
            # --- Optional: Time-based throttling ---
            # current_time_sec = time.time()
            # MIN_PREDICTION_INTERVAL_SEC = 1.0 / 4.0 # Target 4 Hz
            # if current_time_sec - last_prediction_time.get(client_sid, 0) < MIN_PREDICTION_INTERVAL_SEC:
            #     # Too soon since last prediction, skip this cycle but DON'T reset sample counter yet
            #     # print("DBG: Throttling prediction")
            #     return
            # --- End Optional Throttling Check ---

            start_pred_time = time.time()

            # 2. Get Window Data
            # Deque makes slicing efficient, but convert to list for easier access
            window_tuples = list(current_buffer)[-WINDOW_SIZE_SAMPLES:]
            window_timestamps = [item[0] for item in window_tuples]
            window_sensor_data = [item[1] for item in window_tuples] # List of lists

            # Use timestamp of the *last* sample in the window as the reference time
            prediction_timestamp_ms = window_timestamps[-1] if window_timestamps else timestamp_ms

            # 3. TCN Prediction
            final_prediction_label = "Unknown"
            final_confidence = 0.0
            model_used = "None"
            tcn_pred_label = "Error" # Default in case of TCN error

            try:
                # Prepare TCN Input
                window_np = np.array(window_sensor_data, dtype=np.float32)
                if window_np.shape != (WINDOW_SIZE_SAMPLES, TCN_NUM_FEATURES):
                    raise ValueError(f"TCN window shape mismatch. Expected {(WINDOW_SIZE_SAMPLES, TCN_NUM_FEATURES)}, Got {window_np.shape}")

                scaled_window_tcn = tcn_scaler.transform(window_np)
                scaled_window_tcn_reshaped = scaled_window_tcn.reshape(1, WINDOW_SIZE_SAMPLES, TCN_NUM_FEATURES)

                # Predict with TCN
                pred_proba_tcn = tcn_model.predict(scaled_window_tcn_reshaped, verbose=0)
                tcn_pred_index = np.argmax(pred_proba_tcn, axis=1)[0]
                tcn_pred_label = label_encoder.inverse_transform([tcn_pred_index])[0]
                tcn_confidence = float(pred_proba_tcn[0][tcn_pred_index]) # Ensure float

                final_prediction_label = tcn_pred_label
                final_confidence = tcn_confidence
                model_used = "TCN"

            except Exception as e:
                print(f"❌ Error during TCN prediction for SID {client_sid}: {e}")
                # Keep default 'Error' label or handle differently

            # 4. Conditional SVM Prediction (if TCN prediction is in trigger list and no TCN error)
            if model_used == "TCN" and tcn_pred_label in SVM_TRIGGER_CLASSES:
                print(f"DBG: TCN predicted '{tcn_pred_label}', triggering SVM check...")
                try:
                    # Prepare SVM Input (DataFrame needed for feature extraction)
                    # Use SVM_FEATURE_COLUMNS which should match INCOMING_KEYS_FOR_TCN order initially
                    window_df_svm = pd.DataFrame(window_sensor_data, columns=INCOMING_KEYS_FOR_TCN)
                    # Ensure columns match exactly what extract_svm_features expects
                    if list(window_df_svm.columns) != SVM_FEATURE_COLUMNS:
                         print(f"⚠️ Warning: DataFrame columns {list(window_df_svm.columns)} don't match SVM_FEATURE_COLUMNS {SVM_FEATURE_COLUMNS}. Reordering/selecting.")
                         try:
                             window_df_svm = window_df_svm[SVM_FEATURE_COLUMNS]
                         except KeyError as ke:
                             raise ValueError(f"Cannot create DataFrame for SVM features. Missing key: {ke}") from ke


                    # Extract SVM Features
                    svm_features = extract_svm_features(window_df_svm, SVM_FEATURE_COLUMNS)

                    if np.isnan(svm_features).all():
                        print(f"⚠️ Warning: SVM feature extraction resulted in all NaNs for SID {client_sid}. Using TCN result.")
                    else:
                        # Fill remaining NaNs (e.g., with mean if scaler requires it, or 0) - check scaler behavior
                        # Simple approach: fill with 0, but training method matters.
                        # svm_features = np.nan_to_num(svm_features) # Replace NaN with 0
                        # Better: Use imputer fitted during training, or check if scaler handles NaNs

                        # Scale SVM Features
                        svm_features_reshaped = svm_features.reshape(1, -1)
                        # Check if scaler handles NaNs or if we need imputation
                        if np.isnan(svm_features_reshaped).any():
                             print(f"⚠️ Warning: NaNs present in features before SVM scaling. Filling with 0 for prediction. Scaler might behave unexpectedly.")
                             svm_features_reshaped = np.nan_to_num(svm_features_reshaped)

                        scaled_features_svm = svm_scaler.transform(svm_features_reshaped)

                        # Predict with SVM
                        svm_pred_code = svm_model.predict(scaled_features_svm)[0]
                        svm_pred_label = SVM_LABEL_MAP.get(svm_pred_code, f"Unknown_SVM_{svm_pred_code}")

                        # Override the final prediction
                        final_prediction_label = svm_pred_label
                        final_confidence = None # SVM doesn't give probability easily
                        model_used = "SVM (triggered)"
                        print(f"DBG: SVM prediction: '{svm_pred_label}'")


                except Exception as e:
                    print(f"❌ Error during SVM check for SID {client_sid}: {e}. Falling back to TCN result.")
                    # Keep the original TCN prediction if SVM check fails
                    final_prediction_label = tcn_pred_label
                    final_confidence = tcn_confidence
                    model_used = "TCN (SVM Failed)"


            # 5. Emit Final Prediction
            end_pred_time = time.time()
            pred_duration = (end_pred_time - start_pred_time) * 1000 # ms

            print(f"SID {client_sid[-5:]}: Final Pred -> {final_prediction_label} "
                  f"(Conf: {f'{final_confidence*100:.1f}%' if final_confidence is not None else 'N/A'}) "
                  f"| Model: {model_used} | Time: {prediction_timestamp_ms} | Dur: {pred_duration:.1f} ms")

            socketio.emit('prediction', {
                'activity': final_prediction_label,
                'confidence': final_confidence, # Can be None for SVM
                'model_used': model_used,
                'timestamp_ms': prediction_timestamp_ms # Timestamp of last sample in window
                }, room=client_sid)

            # 6. Reset sample counter & update optional time throttle
            samples_since_last_pred[client_sid] = 0
            # last_prediction_time[client_sid] = current_time_sec # Optional

        # --- Buffer Trimming (Using deque's maxlen handles this automatically) ---
        # No explicit trimming needed if deque(maxlen=...) is used

    except KeyError as e:
        print(f"❌ Invalid data point structure from SID {client_sid}: Missing key {e}.")
    except Exception as e:
        print(f"❌ Unexpected error handling sensor data for SID {client_sid}: {type(e).__name__} - {e}")
        # Consider resetting counter or buffer on persistent errors


if __name__ == '__main__':
    print("\n🚀 Starting HAR TCN+SVM Real-time Predictor on 0.0.0.0:80")
    # Optional: Force CPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # tf.config.set_visible_devices([], 'GPU')

    if load_all_models_scalers():
        print("\n  Connect your Android app. Waiting for sensor data...")
        socketio.run(app, host='0.0.0.0', port=80, debug=False, use_reloader=False)
    else:
        print("❌ Exiting due to component loading failure.")