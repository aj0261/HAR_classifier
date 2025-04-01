# prediction/tcn_handler.py
import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder # For type hints/checking

def load_tcn_components(config):
    """Loads the TCN model, scaler, and label encoder."""
    print("--- Loading TCN Components ---")
    try:
        if not os.path.exists(config.TCN_MODEL_FILENAME): raise FileNotFoundError(f"TCN Model file not found: {config.TCN_MODEL_FILENAME}")
        model = tf.keras.models.load_model(config.TCN_MODEL_FILENAME)
        print(f"  TCN Model loaded ({os.path.basename(config.TCN_MODEL_FILENAME)})")

        if not os.path.exists(config.TCN_SCALER_FILENAME): raise FileNotFoundError(f"TCN Scaler file not found: {config.TCN_SCALER_FILENAME}")
        scaler = joblib.load(config.TCN_SCALER_FILENAME)
        print(f"  TCN Scaler loaded ({os.path.basename(config.TCN_SCALER_FILENAME)})")
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != config.TCN_NUM_FEATURES:
            print(f"   ⚠️ Config/Scaler Mismatch: Scaler expects {scaler.n_features_in_}, config says {config.TCN_NUM_FEATURES}")

        if not os.path.exists(config.TCN_LABEL_ENCODER_FILENAME): raise FileNotFoundError(f"Label Encoder file not found: {config.TCN_LABEL_ENCODER_FILENAME}")
        label_encoder = joblib.load(config.TCN_LABEL_ENCODER_FILENAME)
        print(f"  Label Encoder loaded ({os.path.basename(config.TCN_LABEL_ENCODER_FILENAME)})")
        print(f"   TCN Classes: {label_encoder.classes_}")

        return model, scaler, label_encoder

    except Exception as e:
        print(f"❌❌❌ Error loading TCN components: {e}")
        raise # Re-raise the exception to be caught by the main loader

def predict_tcn(model, scaler, window_data_np, config):
    """
    Performs prediction using the loaded TCN model.

    Args:
        model: The loaded Keras TCN model.
        scaler: The loaded StandardScaler for TCN.
        window_data_np: NumPy array of window data (shape: [WINDOW_SIZE, TCN_NUM_FEATURES]).
        config: The configuration object.

    Returns:
        A tuple: (predicted_class_index, prediction_probability)
        Returns (None, None) if prediction fails.
    """
    try:
        # 1. Validate shape
        expected_shape = (config.WINDOW_SIZE_SAMPLES, config.TCN_NUM_FEATURES)
        if window_data_np.shape != expected_shape:
            print(f"❌ TCN Predict Error: Input data shape mismatch. Expected {expected_shape}, Got {window_data_np.shape}")
            return None, None

        # 2. Scale
        scaled_window = scaler.transform(window_data_np)

        # 3. Reshape for TCN (batch_size=1, timesteps, features)
        scaled_window_reshaped = scaled_window.reshape(1, config.WINDOW_SIZE_SAMPLES, config.TCN_NUM_FEATURES)

        # 4. Predict probabilities
        pred_proba = model.predict(scaled_window_reshaped, verbose=0) # [1, num_classes]

        # 5. Get best class index and its probability
        pred_index = np.argmax(pred_proba, axis=1)[0]
        pred_prob = float(pred_proba[0][pred_index]) # Convert to float

        return pred_index, pred_prob

    except Exception as e:
        print(f"❌ Error during TCN prediction: {e}")
        return None, None