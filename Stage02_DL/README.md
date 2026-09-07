# STAGE 02 — DEEP LEARNING (Disaster Response Coordination)

## 1. Stage 02 Overview
This stage implements deep learning models for visual and chronological analysis of disaster data.

## 2. Mission
"Go from patterns to instincts." The system visually analyzes imagery and sequences temporal sensor data to detect subtle risks that basic thresholds miss.

## 3. Problem Being Solved
Stage 01 relies on hard numerical thresholds (e.g., river level > 12m). Stage 02 detects flooded roadways visually before sensors trigger, and uses sequential forecasting to predict river level breaches hours in advance.

## 4. Visual Data
The CNN uses images of flooded and unflooded street conditions (SDSU Midwest Flood 2019 dataset).

## 5. Satellite/Time-Series Data
Not implemented directly in DL pipeline due to memory constraints; relying on temporal gauge data instead.

## 6. Temporal Sensor Data
Sequential river water level readings (`Engineered_History_Trend_Dataset.csv`), sampled hourly with a 24-hour lookback.

## 7. CNN Architecture
A baseline PyTorch Convolutional Neural Network (CNN) with 3 convolutional blocks and adaptive average pooling, ending in a dense classifier.

## 8. CNN Preprocessing
Images resized to 128x128. Normalized using ImageNet stats: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.

## 9. Data Augmentation
`RandomHorizontalFlip()` applied to the training split. *Note: Brightness variation/noise for night/rain conditions is not currently implemented in the pipeline.*

## 10. CNN Training
- **Loss:** Weighted CrossEntropyLoss
- **Optimizer:** Adam (lr=1e-3)
- **Batch Size:** 32

## 11. CNN Evaluation
- **Test Accuracy:** 0.8000
- **Macro F1:** 0.4444
- **Precision (Macro):** 0.4000
- **Recall (Macro):** 0.5000
*Note: Due to severe class imbalance, the F1 score for the 'flooded' class is low.*

## 12. Confusion Matrix
See `data/outputs/cnn_confusion_matrix.png`. The model heavily biases toward one class.

## 13. Wet-Asphalt Failure Analysis
*Limitation: Due to missing localized image annotations, wet-asphalt false positives could not be numerically isolated in the current test set.*

## 14. Saliency/Grad-CAM Analysis
*Limitation: A true Grad-CAM is missing. `02_eda_engineer.py` attempts a basic "dark-pixel overlap" screening, but it cannot guarantee the model is focusing on water rather than shadows.*

## 15. LSTM/Transformer Architecture
A 2-layer PyTorch LSTM with 64 hidden units, taking 1-dimensional features.

## 16. Time-Series Preprocessing
StandardScaler fitted *only* on the training data. Data partitioned chronologically (70/15/15) with no temporal leakage.

## 17. 3-Hour Forecasting Approach
The model is trained for 1-step forecasting, and the inference API (`forecast_water_levels`) autoregressively rolls the prediction forward for the specified horizon.

## 18. Forecast Metrics (LSTM)
- **MAE:** 23.45 m
- **RMSE:** 49.39 m
*Note: High error indicates the model struggles to accurately forecast the exact scale of water level changes without exogenous variables (like rainfall).*

## 19. Forecast Examples/Results
See `data/outputs/lstm_actual_vs_predicted.png`.

## 20. ML vs DL Benchmark
*Limitation: A direct side-by-side Visual Benchmark Reel between Stage 01 and Stage 02 is missing from the evaluation script.*

## 21. Model Save/Load
Models are saved properly as PyTorch state dictionaries (`disaster_cnn.pt`, `water_level_lstm.pt`) and Scikit-Learn pipelines (`severity_model.joblib`).

## 22. Inference Workflow
`05_integration_engineer.py` isolates inference from training, ensuring only the necessary preprocessing steps run during deployment.

## 23. Unified Alert API
The `DLIntegrationEngine` provides a robust API wrapping all three models.

## 24. API Example Request
```python
engine = DLIntegrationEngine()
engine.forecast_water_levels([10.1, 10.2, ... 10.5], horizon=3)
```

## 25. API Example Response
```json
{
    "forecast_water_level": 11.2,
    "forecast_water_levels": [10.8, 11.0, 11.2],
    "horizon": 3,
    "lookback": 24
}
```

## 26. Automated Tests
A suite of Pytest checks (`test/test_stage02.py`) validates sequence shapes, CNN tensor shapes, and API exception handling (e.g., rejecting invalid inputs).

## 27. Project Structure
```text
Stage02_DL/
  ├── 01_data_engineer.py (Broken locally due to Colab paths)
  ├── 02_eda_engineer.py
  ├── 03_dl_engineer.py
  ├── 04_evaluation_engineer.py
  ├── 05_integration_engineer.py
  ├── test/test_stage02.py
  └── data/
```

## 28. How to run Stage 02
1. Run `python 03_dl_engineer.py` to train DL models.
2. Run `python 04_evaluation_engineer.py` to generate reports.
3. Run `pytest test/ -v` to verify endpoints.

## 29. Known Limitations
1. `01_data_engineer.py` is hardcoded for Google Colab and relies on Keras instead of PyTorch.
2. Missing true Grad-CAM implementation.
3. Missing heavy rain / night data augmentation.
4. Missing Visual Benchmark Reel.

## 30. Final Completion Status
**PARTIALLY COMPLETE / READY WITH MINOR LIMITATIONS**
The core ML integration API and model files function properly on valid inputs, but the evaluation pipeline lacks deep interpretability and robust augmentation.
