# config.py
import os

# --- Base Directory ---
# Assumes this config.py is in the root of 'servers for best Models'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- TCN Model Configuration ---
TCN_MODEL_DIR = os.path.join(BASE_DIR, 'TCN_newResults')
TCN_MODEL_FILENAME = os.path.join(TCN_MODEL_DIR, 'tcn_har_model.keras')
TCN_SCALER_FILENAME = os.path.join(TCN_MODEL_DIR, 'tcn_scaler.joblib')
TCN_LABEL_ENCODER_FILENAME = os.path.join(TCN_MODEL_DIR, 'tcn_label_encoder.joblib')
TCN_NUM_FEATURES = 6 # accel_x/y/z, gyro_x/y/z
# TCN feature order (mapping from incoming keys)
TCN_MODEL_FEATURE_ORDER = ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro']
# Actual keys expected from the mobile app data point
INCOMING_SENSOR_KEYS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']


# --- SVM Model Configuration ---
SVM_MODEL_DIR = os.path.join(BASE_DIR, 'models')
SVM_MODEL_FILENAME = os.path.join(SVM_MODEL_DIR, 'svm_har_model.joblib')
SVM_SCALER_FILENAME = os.path.join(SVM_MODEL_DIR, 'scaler.joblib')
SVM_NUM_FEATURES_EXPECTED = 53 # Expected number of features *after* extraction for SVM
# Columns needed from raw data for SVM feature extraction
SVM_FEATURE_EXTRACTION_COLS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
# Map SVM numerical output codes to meaningful labels
SVM_LABEL_MAP = {0: "Walking", 1: "Stairs"} # <<<--- IMPORTANT: Update this map
# TCN classes (as strings from LabelEncoder) that trigger the SVM model check
SVM_TRIGGER_CLASSES = ['A', 'C'] # Add other classes if needed


# --- Windowing Parameters (MUST MATCH TRAINING) ---
SAMPLING_RATE_HZ = 20
WINDOW_SIZE_SAMPLES = 60 # Should be consistent for both models if features depend on it
STRIDE_SAMPLES = 15 # Determines prediction frequency based on samples


# In config.py, find the "Sanity Checks" section

# Ensure incoming keys cover TCN needs
# --- COMMENT OUT OR DELETE THIS BLOCK ---
# if not set(TCN_MODEL_FEATURE_ORDER).issubset(set(INCOMING_SENSOR_KEYS)):
#     missing = set(TCN_MODEL_FEATURE_ORDER) - set(INCOMING_SENSOR_KEYS)
#     print(f"❌ Config Error: Missing required TCN keys in INCOMING_SENSOR_KEYS: {missing}")
#     exit() # <<<--- REMOVE THIS EXIT CALL
# --- END BLOCK TO REMOVE/COMMENT ---

# Ensure incoming keys cover SVM feature extraction needs (This check is likely OK)
if not set(SVM_FEATURE_EXTRACTION_COLS).issubset(set(INCOMING_SENSOR_KEYS)):
     missing = set(SVM_FEATURE_EXTRACTION_COLS) - set(INCOMING_SENSOR_KEYS)
     print(f"❌ Config Error: Missing required SVM feature extraction keys in INCOMING_SENSOR_KEYS: {missing}")
     exit()

# --- Server Configuration ---
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 80