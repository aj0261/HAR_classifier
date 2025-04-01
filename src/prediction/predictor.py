# prediction/predictor.py
import time
import numpy as np
import pandas as pd # Make sure pandas is imported
from . import tcn_handler
from . import svm_handler

class Predictor:
    """Manages loading models and orchestrating the prediction process."""

    def __init__(self, config):
        """Loads all models and necessary components."""
        self.config = config
        self.tcn_model = None
        self.tcn_scaler = None
        self.label_encoder = None
        self.svm_model = None
        self.svm_scaler = None
        self.is_ready = False

        try:
            self.tcn_model, self.tcn_scaler, self.label_encoder = tcn_handler.load_tcn_components(config)
            self.svm_model, self.svm_scaler = svm_handler.load_svm_components(config)
            self.is_ready = True
            print("\n    Predictor initialized successfully. All components loaded.")
        except Exception as e:
            print(f"\n❌❌❌ Predictor initialization failed: {e}")
            print("   Server might run, but predictions will fail until components are loaded.")

    def predict_activity(self, window_sensor_data, window_timestamps):
        """
        Performs activity prediction using TCN and potentially SVM.

        Args:
            window_sensor_data (list): List of lists containing sensor data for the window.
                                      Order must match config.INCOMING_SENSOR_KEYS.
            window_timestamps (list): List of timestamps corresponding to the sensor data.

        Returns:
            dict: A dictionary containing prediction results:
                  {'activity': str, 'confidence': float|None, 'model_used': str,
                   'timestamp_ms': float, 'processing_time_ms': float, 'error': str|None}
        """
        start_time = time.time()
        result = {
            'activity': 'Error',
            'confidence': None,
            'model_used': 'None',
            'timestamp_ms': window_timestamps[-1] if window_timestamps else time.time() * 1000,
            'processing_time_ms': 0,
            'error': None
        }

        if not self.is_ready:
            result['error'] = "Predictor not ready (models failed to load)."
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            return result

        # --- TCN Prediction ---
        tcn_pred_label = "Error"
        tcn_confidence = None
        model_used = "None"

        try:
            # --- START: Data Reordering for TCN ---
            # 1. Create DataFrame with columns named according to incoming data keys
            window_df = pd.DataFrame(window_sensor_data, columns=self.config.INCOMING_SENSOR_KEYS)

            # 2. Create a mapping from TCN expected order to incoming keys if names differ
            #    Example: TCN expects 'x_accel', incoming is 'accel_x'
            #    This step assumes the *meaning* is the same, just the name differs.
            #    If TCN_MODEL_FEATURE_ORDER and INCOMING_SENSOR_KEYS are identical *after* fixing config,
            #    this explicit remapping might not be strictly necessary, but it's safer.

            # Check if reordering/renaming is actually needed
            if self.config.TCN_MODEL_FEATURE_ORDER != self.config.INCOMING_SENSOR_KEYS:
                 # Build a rename mapping if keys differ but represent the same signals
                 # This assumes a one-to-one correspondence based on position or meaning
                 # Example: TCN's 'x_accel' corresponds to incoming 'accel_x'
                 rename_map = dict(zip(self.config.INCOMING_SENSOR_KEYS, self.config.TCN_MODEL_FEATURE_ORDER))
                 # Example rename_map: {'accel_x': 'x_accel', 'accel_y': 'y_accel', ...}

                 # Apply renaming first
                 window_df_renamed = window_df.rename(columns=rename_map)
                 # Then select columns in the TCN model's required order
                 window_df_tcn_ordered = window_df_renamed[self.config.TCN_MODEL_FEATURE_ORDER]
            else:
                 # If keys are identical, just ensure order (though Dataframe creation from list of lists usually preserves order)
                 window_df_tcn_ordered = window_df[self.config.TCN_MODEL_FEATURE_ORDER]


            # 3. Convert the correctly ordered DataFrame to a NumPy array for the TCN handler
            window_np_tcn_ordered = window_df_tcn_ordered.to_numpy(dtype=np.float32)
            # --- END: Data Reordering for TCN ---


            # Pass the correctly ordered NumPy array to the TCN handler
            tcn_pred_index, tcn_confidence = tcn_handler.predict_tcn(
                self.tcn_model, self.tcn_scaler, window_np_tcn_ordered, self.config # Use the ORDERED array
            )

            if tcn_pred_index is not None:
                tcn_pred_label = self.label_encoder.inverse_transform([tcn_pred_index])[0]
                model_used = "TCN"
                result['activity'] = tcn_pred_label
                result['confidence'] = tcn_confidence
                result['model_used'] = model_used
            else:
                result['error'] = "TCN prediction failed."
                tcn_pred_label = "Error" # Ensure it's marked as error

        except KeyError as e:
             print(f"❌ Key Error during TCN data preparation: Missing key {e}. Check config mappings.")
             result['error'] = f"TCN Data Prep Error: Missing key {e}"
             tcn_pred_label = "Error"
        except Exception as e:
            print(f"❌ Unexpected Error in TCN phase: {type(e).__name__} - {e}")
            result['error'] = f"TCN phase error: {e}"
            tcn_pred_label = "Error"


        # --- Conditional SVM Prediction ---
        # Only proceed if TCN prediction was successful and triggered
        if model_used == "TCN" and tcn_pred_label in self.config.SVM_TRIGGER_CLASSES:
            print(f"DBG: TCN='{tcn_pred_label}', triggering SVM...")
            try:
                # Create DataFrame with correct columns for feature extraction
                # Use the *original* DataFrame created from incoming data
                # as SVM feature extraction likely uses the 'accel_x' style keys
                window_df_svm = pd.DataFrame(window_sensor_data, columns=self.config.INCOMING_SENSOR_KEYS)

                # Select only columns needed for SVM features, in the correct order defined in config
                # This ensures extract_svm_features gets exactly what it expects
                window_df_svm_input = window_df_svm[self.config.SVM_FEATURE_EXTRACTION_COLS]

                # Extract features
                svm_features = svm_handler.extract_svm_features(window_df_svm_input, self.config)

                if not np.isnan(svm_features).all():
                    # Predict using SVM
                    svm_pred_code = svm_handler.predict_svm(
                        self.svm_model, self.svm_scaler, svm_features, self.config
                    )

                    if svm_pred_code is not None:
                        svm_pred_label = self.config.SVM_LABEL_MAP.get(svm_pred_code, f"Unknown_SVM_{svm_pred_code}")
                        print(f"DBG: SVM prediction: '{svm_pred_label}' (Code: {svm_pred_code})")
                        # Override TCN result
                        result['activity'] = svm_pred_label
                        result['confidence'] = None # SVM confidence not readily available
                        result['model_used'] = "SVM (triggered)"
                    else:
                        print("⚠️ SVM prediction failed. Using TCN result.")
                        result['model_used'] = "TCN (SVM Predict Failed)"
                        # Keep TCN results already in 'result' dict
                else:
                    print("⚠️ SVM feature extraction failed (all NaNs). Using TCN result.")
                    result['model_used'] = "TCN (SVM Feature Fail)"
                    # Keep TCN results

            except KeyError as e:
                 print(f"❌ Key Error during SVM data preparation: Missing key {e}. Check config mappings.")
                 result['error'] = f"SVM Data Prep Error: Missing key {e}"
                 result['model_used'] = "TCN (SVM Phase Error)" # Revert model used status
            except Exception as e:
                print(f"❌ Unexpected Error in SVM phase: {type(e).__name__} - {e}")
                result['error'] = f"SVM phase error: {e}"
                result['model_used'] = "TCN (SVM Phase Error)" # Revert model used status
                # Keep TCN results

        result['processing_time_ms'] = (time.time() - start_time) * 1000
        return result