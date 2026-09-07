## Stage02 Deep Learning

This stage uses the real datasets under `data/raw` and provides three trained
components:

1. A PyTorch CNN for `flooded` versus `unflooded` images in
	`data/raw/Flood_Image_Dataset`.
2. A PyTorch LSTM for recursive future water-level forecasting from
	`data/Engineered_History_Trend_Dataset.csv`. It uses the engineered hourly
	`timestamp` and `river_level_m` history after chronological sorting.
3. A labelled zone-risk model from `data/raw/04_MASTER_DATASET/Master_Dataset.csv`.
	Its real labels are `Low`, `Moderate`, and `Severe`.

### Run

Install the runtime dependencies in the selected Python environment:

```bash
python -m pip install torch torchvision pandas numpy scikit-learn pillow joblib seaborn matplotlib
```

Run training and evaluation from the repository root:

```bash
python Stage02_DL/03_dl_engineer.py
```

`STAGE02_EPOCHS` and `STAGE02_LOOKBACK` can override the defaults of 5 epochs
and 24 historical observations. The CNN uses class-balanced sampling and
weighted cross-entropy so the minority `flooded` class is not ignored. The
splits are reproducible with seed 42. The LSTM split is chronological and its
scaler is fit on the training partition.

### Artifacts

Models are written to `data/models`:

- `disaster_cnn.pt`
- `water_level_lstm.pt`
- `water_level_scaler.joblib`
- `severity_model.joblib`

Evaluation files and plots are written to `data/outputs`:

- `dl_training_metrics.json`
- `cnn_confusion_matrix.png`
- `cnn_training_curves.png`
- `lstm_actual_vs_predicted.png`
- `severity_confusion_matrix.png`

The `03_dl_engineer.py` module also exposes `predict_image`,
`forecast_water_level`, `forecast_water_levels`, and `predict_severity` for
inference without retraining. `forecast_water_levels` recursively predicts a
future horizon of 1 to 168 steps from the latest 24 readings.

### Limitations

The image benchmark is imbalanced: the current held-out test split contains
many more `unflooded` than `flooded` images. Reported CNN accuracy must be read
with its per-class precision/recall and confusion matrix; the trained baseline
does not yet provide reliable flooded-class recall. The satellite Arrow data
is a separate Hugging Face image/mask benchmark and is not used as road-blockage
ground truth. No severity label is fabricated: the severity component uses the
real `zone_risk` labels in the master dataset.
