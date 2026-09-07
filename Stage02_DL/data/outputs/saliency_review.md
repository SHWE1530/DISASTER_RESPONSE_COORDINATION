# CNN Saliency Review

## Purpose

This EDA step checks whether the trained CNN's strongest gradient-saliency
regions appear to rely on dark image regions. It is a screening step for the
failure mode of learning dark shadows instead of floodwater.

## Samples

- Images reviewed: 6
- Flooded examples: 3
- Unflooded examples: 3
- Mean dark-pixel overlap in the top 10% saliency: 0.0005
- Samples above the 0.30 screening threshold: 0

## Interpretation

The overlays in `data/outputs/saliency_maps/` must be visually reviewed for
water-shaped regions, edges, roads, buildings, and shadows. The dark-pixel
overlap is not a water mask and cannot prove semantic floodwater focus.
Because this dataset has no pixel-level water/shadow annotations, this audit
must report evidence rather than claim verification.

The CNN's flooded-class recall and prediction errors must be considered with
this review. A low dark-pixel overlap does not guarantee reliable floodwater
recognition, and a high overlap is a warning that shadow-related shortcuts may
be present.
