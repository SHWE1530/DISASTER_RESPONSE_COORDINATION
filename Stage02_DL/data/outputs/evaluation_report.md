# Stage02 Deep Learning Evaluation Report

## Workflow

Raw test data -> saved model -> inference -> predictions -> metrics -> error analysis -> recommendation.

The CNN test partition contains 75 unseen images. The LSTM test partition contains
1314 chronological one-step sequences. The deterministic 70/15/15 split is reconstructed
is replayed from `cnn_split_manifest.json` / `lstm_split_manifest.json`, written by the training
script, so the held-out partition is reproduced exactly by file path rather than re-derived from a
seed and the filesystem's directory ordering.

The CNN decision threshold is the validation-tuned value stored in the checkpoint. It is NOT
re-tuned here: optimising a threshold on the test set and then reporting that set's score is not
an independent estimate.

## CNN

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9600 |
| Macro precision | 0.9290 |
| Macro recall | 0.9500 |
| Macro F1 | 0.9390 |
| Decision threshold (validation-tuned) | 0.5000 |
| Test set size | 75 |
| Flooded positives in test set | 15 |
| Flooded ROC-AUC | 0.9856 |
| Flooded PR-AUC | 0.9578 |
| Average latency (ms/sample) | 34.7500 |

The confusion matrix and `cnn_test_predictions.csv` show the error pattern. The flooded class recall is
0.9333.

## LSTM

| Metric | Result |
| --- | ---: |
| MAE | 0.2284 |
| MSE | 0.4997 |
| RMSE | 0.7069 |
| Average latency (ms/sample) | 0.1717 |

LSTM errors are stored in `lstm_test_predictions.csv`; larger absolute errors identify difficult periods.

## Robustness

The LSTM perturbation test used a small scaled-space Gaussian noise standard deviation of 0.05.
Its perturbed MAE was 0.2284 and RMSE was 0.7084. These are sensitivity
measurements, not replacements for the official test metrics.

## Artifacts

- `evaluation_cnn_confusion_matrix.png`
- `evaluation_lstm_actual_vs_predicted.png`
- `cnn_test_predictions.csv`
- `lstm_test_predictions.csv`
- `evaluation_metrics.json`
- `evaluation_comparison.csv`
