# prediction/svm_handler.py
import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import iqr
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler # For type hints
from sklearn.svm import SVC # For type hints

def load_svm_components(config):
    """Loads the SVM model and scaler."""
    print("--- Loading SVM Components ---")
    try:
        if not os.path.exists(config.SVM_MODEL_FILENAME): raise FileNotFoundError(f"SVM Model file not found: {config.SVM_MODEL_FILENAME}")
        model = joblib.load(config.SVM_MODEL_FILENAME)
        print(f"  SVM Model loaded ({os.path.basename(config.SVM_MODEL_FILENAME)})")
        if hasattr(model, 'kernel'): print(f"   SVM Type: SVC (kernel={model.kernel})")

        if not os.path.exists(config.SVM_SCALER_FILENAME): raise FileNotFoundError(f"SVM Scaler file not found: {config.SVM_SCALER_FILENAME}")
        scaler = joblib.load(config.SVM_SCALER_FILENAME)
        print(f"  SVM Scaler loaded ({os.path.basename(config.SVM_SCALER_FILENAME)})")
        if hasattr(scaler, 'n_features_in_'):
             print(f"   SVM Scaler expects {scaler.n_features_in_} features.")
             if scaler.n_features_in_ != config.SVM_NUM_FEATURES_EXPECTED:
                  print(f"   ⚠️ Config/Scaler Mismatch: Scaler expects {scaler.n_features_in_}, config expects {config.SVM_NUM_FEATURES_EXPECTED} after extraction.")

        return model, scaler

    except Exception as e:
        print(f"❌❌❌ Error loading SVM components: {e}")
        raise # Re-raise

def extract_svm_features(window_df, config):
    """
    Extracts features for the SVM model from a DataFrame window.

    Args:
        window_df (pd.DataFrame): DataFrame containing sensor data for the window.
                                  Columns should match config.SVM_FEATURE_EXTRACTION_COLS.
        config: The configuration object.

    Returns:
        np.array: Array of extracted features, or array of NaNs if errors occur.
                  Guaranteed to have length config.SVM_NUM_FEATURES_EXPECTED.
    """
    expected_cols = config.SVM_FEATURE_EXTRACTION_COLS
    num_expected_features = config.SVM_NUM_FEATURES_EXPECTED
    sampling_rate = config.SAMPLING_RATE_HZ

    # Check if required columns exist
    if not all(col in window_df.columns for col in expected_cols):
        missing = [col for col in expected_cols if col not in window_df.columns]
        print(f"❌ SVM Extract Error: Missing columns {missing}")
        return np.full(num_expected_features, np.nan)

    features = []
    if window_df.empty or len(window_df) < 5: # Reduced min length slightly
         print("⚠️ SVM Extract Warning: Window too short or empty.")
         return np.full(num_expected_features, np.nan)

    # --- Feature Calculation (with robust NaN handling) ---
    axes = expected_cols
    for axis in axes:
        data = window_df[axis]
        if data.isnull().all() or len(data.dropna()) == 0:
            features.extend([np.nan] * 7)
            continue
        try:
            # Dropna before calculating stats that require it
            data_clean = data.dropna()
            if len(data_clean) == 0: # Check again after dropna
                 features.extend([np.nan] * 7)
                 continue
            iqr_val = iqr(data_clean) if len(data_clean) > 0 else np.nan
            features.extend([
                data.mean(skipna=True), data.std(skipna=True), data.var(skipna=True),
                data.min(skipna=True), data.max(skipna=True), data.median(skipna=True),
                iqr_val
            ])
        except Exception as e:
            print(f"⚠️ SVM Extract Warning (Stat calc for {axis}): {e}")
            features.extend([np.nan] * 7)

    # Accel Magnitude
    accel_cols = [col for col in ['accel_x', 'accel_y', 'accel_z'] if col in window_df.columns]
    if len(accel_cols) > 0:
        # Fill NaNs with 0 *before* squaring to avoid propagation
        accel_mag = np.sqrt((window_df[accel_cols].fillna(0)**2).sum(axis=1))
        features.extend([accel_mag.mean(), accel_mag.std()])
    else:
        features.extend([np.nan, np.nan])

    # Correlation
    def safe_corr(col1, col2):
        if col1 in window_df.columns and col2 in window_df.columns:
            data1 = window_df[col1].dropna()
            data2 = window_df[col2].dropna()
            if len(data1) > 1 and len(data2) > 1 and data1.std() > 1e-6 and data2.std() > 1e-6: # Added variance check
                 aligned_data1, aligned_data2 = data1.align(data2, join='inner')
                 if len(aligned_data1) > 1:
                     # Use np.corrcoef for potentially more robust handling
                     corr_matrix = np.corrcoef(aligned_data1, aligned_data2)
                     return corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        return 0.0

    features.append(safe_corr('accel_x', 'accel_y'))
    features.append(safe_corr('accel_x', 'accel_z'))
    features.append(safe_corr('accel_y', 'accel_z'))

    # FFT
    N = len(window_df)
    xf_freq = rfftfreq(N, 1 / sampling_rate) if N > 0 else np.array([])

    def get_fft_features(data_series):
        if N < 2 or data_series.isnull().all(): return [np.nan, np.nan]
        values = data_series.fillna(0).values
        yf_mag = np.abs(rfft(values))
        if len(xf_freq) > 1 and len(yf_mag) > 1:
            dominant_freq_idx = np.argmax(yf_mag[1:]) + 1
            dom_freq = xf_freq[dominant_freq_idx] if dominant_freq_idx < len(xf_freq) else np.nan
            energy = np.sum(yf_mag**2) / N if N > 0 else np.nan
            return [dom_freq, energy]
        elif len(xf_freq) >= 1 and len(yf_mag) >= 1: # Handle N=1 (only DC) or N=0
             energy = np.sum(yf_mag**2) / N if N > 0 else np.nan
             return [xf_freq[0] if len(xf_freq) > 0 else np.nan, energy] # Return DC freq if N=1
        else:
            return [np.nan, np.nan]

    if 'accel_mag' not in locals() or accel_mag.isnull().all(): features.extend([np.nan, np.nan])
    else: features.extend(get_fft_features(accel_mag))

    if 'accel_y' not in window_df.columns: features.extend([np.nan, np.nan])
    else: features.extend(get_fft_features(window_df['accel_y']))

    if 'accel_z' not in window_df.columns: features.extend([np.nan, np.nan])
    else: features.extend(get_fft_features(window_df['accel_z']))
    # --- End Feature Calculation ---

    result = np.array(features, dtype=float)

    # Final check for correct feature count
    if len(result) != num_expected_features:
         print(f"❌ SVM Extract Error: Feature count mismatch. Got {len(result)}, Expected {num_expected_features}. Padding/truncating.")
         if len(result) < num_expected_features:
             result = np.pad(result, (0, num_expected_features - len(result)), 'constant', constant_values=np.nan)
         else:
             result = result[:num_expected_features]

    # Check for NaNs before returning (caller might need to handle)
    if np.isnan(result).all():
        print("⚠️ SVM Extract Warning: All extracted features are NaN.")
    elif np.isnan(result).any():
         print(f"⚠️ SVM Extract Warning: Contains {np.isnan(result).sum()} NaN values.")

    return result

def predict_svm(model, scaler, features_np, config):
    """
    Performs prediction using the loaded SVM model.

    Args:
        model: The loaded scikit-learn SVM model.
        scaler: The loaded StandardScaler for SVM features.
        features_np: NumPy array of *extracted* SVM features.
        config: The configuration object.

    Returns:
        Predicted class code (e.g., 0, 1) or None if prediction fails.
    """
    try:
        # 1. Validate shape (optional, but good practice)
        if features_np.ndim == 1:
            features_reshaped = features_np.reshape(1, -1)
        elif features_np.ndim == 2 and features_np.shape[0] == 1:
            features_reshaped = features_np
        else:
             print(f"❌ SVM Predict Error: Unexpected feature array shape {features_np.shape}")
             return None

        if features_reshaped.shape[1] != config.SVM_NUM_FEATURES_EXPECTED:
             print(f"❌ SVM Predict Error: Feature count mismatch. Expected {config.SVM_NUM_FEATURES_EXPECTED}, Got {features_reshaped.shape[1]}")
             return None

        # 2. Handle NaNs before scaling (CRITICAL)
        # Option 1: Fill with 0 (simple, but maybe suboptimal)
        if np.isnan(features_reshaped).any():
             print(f"   ⚠️ SVM Predict Warning: Filling {np.isnan(features_reshaped).sum()} NaN(s) with 0 before scaling.")
             features_imputed = np.nan_to_num(features_reshaped)
        else:
            features_imputed = features_reshaped
        # Option 2: Use an imputer fitted during training (better)
        # features_imputed = svm_imputer.transform(features_reshaped) # If you have svm_imputer

        # 3. Scale
        scaled_features = scaler.transform(features_imputed)

        # 4. Predict
        prediction_code = model.predict(scaled_features)[0]
        return prediction_code

    except Exception as e:
        print(f"❌ Error during SVM prediction: {e}")
        return None