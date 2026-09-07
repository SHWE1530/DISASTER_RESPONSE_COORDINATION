# Stage02 Deep Learning Evaluation Report

## Workflow

Raw test data -> saved model -> inference -> predictions -> metrics -> error analysis -> recommendation.

The CNN test partition contains 75 unseen images. The LSTM test partition contains
1314 chronological one-step sequences. The deterministic 70/15/15 split is reconstructed
from the raw data with seed 42; split manifests were not saved by the DL training script, so exact replay
depends on the raw files and folder ordering remaining unchanged.

## CNN

| Metric | Result |
| --- | ---: |
| Accuracy | 0.8933 |
| Macro precision | 0.8443 |
| Macro recall | 0.8083 |
| Macro F1 | 0.8244 |
| Best Threshold | 0.7500 |
| Flooded ROC-AUC | 0.8744 |
| Flooded PR-AUC | 0.7990 |
| Average latency (ms/sample) | 13.0276 |

The confusion matrix and `cnn_test_predictions.csv` show the error pattern. The flooded class recall is
0.6667.

## LSTM

| Metric | Result |
| --- | ---: |
| MAE | 0.2306 |
| MSE | 0.5026 |
| RMSE | 0.7090 |
| Average latency (ms/sample) | 0.1704 |

LSTM errors are stored in `lstm_test_predictions.csv`; larger absolute errors identify difficult periods.

## Robustness

The LSTM perturbation test used a small scaled-space Gaussian noise standard deviation of 0.05.
Its perturbed MAE was 0.2309 and RMSE was 0.7109. These are sensitivity
measurements, not replacements for the official test metrics.

## Artifacts

- `evaluation_cnn_confusion_matrix.png`
- `evaluation_lstm_actual_vs_predicted.png`
- `cnn_test_predictions.csv`
- `lstm_test_predictions.csv`
- `evaluation_metrics.json`
- `evaluation_comparison.csv`
