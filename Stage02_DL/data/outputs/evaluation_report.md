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
| Accuracy | 0.6933 |
| Macro precision | 0.6526 |
| Macro recall | 0.7333 |
| Macro F1 | 0.6437 |
| Flooded ROC-AUC | 0.8767 |
| Flooded PR-AUC | 0.7843 |
| Average latency (ms/sample) | 3.5149 |

The confusion matrix and `cnn_test_predictions.csv` show the error pattern. The flooded class recall is
0.8000; accuracy alone is therefore misleading because the test set is imbalanced.

## LSTM

| Metric | Result |
| --- | ---: |
| MAE | 0.2950 |
| MSE | 0.5540 |
| RMSE | 0.7443 |
| Average latency (ms/sample) | 0.1035 |

LSTM errors are stored in `lstm_test_predictions.csv`; larger absolute errors identify difficult periods.

## Robustness

The LSTM perturbation test used a small scaled-space Gaussian noise standard deviation of 0.05.
Its perturbed MAE was 0.2972 and RMSE was 0.7493. These are sensitivity
measurements, not replacements for the official test metrics.

## Generalization and model comparison

The saved checkpoints do not contain epoch-by-epoch histories, so a numeric training-versus-validation
generalization gap cannot be recomputed without retraining. Existing training curves are preserved in
`cnn_training_curves.png`; the evaluation pipeline does not alter the models.

CNN and LSTM solve different tasks and must not be ranked by accuracy against regression error. The CNN
produces a visual flood class, while the LSTM forecasts a continuous water level. Both outputs are useful
signals for a later response component, but this evaluation provides no evidence that one replaces the other.

## Artifacts

- `evaluation_cnn_confusion_matrix.png`
- `evaluation_lstm_actual_vs_predicted.png`
- `cnn_test_predictions.csv`
- `lstm_test_predictions.csv`
- `evaluation_metrics.json`
- `evaluation_comparison.csv`
