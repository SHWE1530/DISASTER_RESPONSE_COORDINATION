
===============================================================
STAGE 2 — COMPLETE DEEP LEARNING DATA PACKAGE
===============================================================

Project:
Disaster Response Coordination —
Building an Autonomous, Multi-Agent Decision Engine for Urban Crises

===============================================================
DATA COMPONENTS
===============================================================

1. SATELLITE FLOOD DATASET
---------------------------------------------------------------
Source:
SDSU MidWest Flood 2019

Images:
500 labelled satellite image + flood mask pairs

Purpose:
Flooded / Non-Flooded visual classification

---------------------------------------------------------------

2. RAINFALL DATASET
---------------------------------------------------------------
Extracted rainfall dataset files:
13

Purpose:
Rainfall-based temporal disaster signal

---------------------------------------------------------------

3. RIVER WATER LEVEL DATASET
---------------------------------------------------------------
Extracted RWL CSV files:
5

Purpose:
River water level temporal monitoring and forecasting

---------------------------------------------------------------

4. MASTER DATASET
---------------------------------------------------------------
Existing Master_Dataset.csv:
AVAILABLE

Purpose:
Temporal sequence modelling

---------------------------------------------------------------

5. METADATA
---------------------------------------------------------------
DATASET_INVENTORY.csv

Contains:
- File name
- Relative path
- Row count
- Column count
- Column names

===============================================================
DEEP LEARNING USAGE
===============================================================

Satellite Images
        ↓
       CNN
        ↓
Flood Visual Signal

Rainfall + River Water Level
        ↓
   LSTM / Transformer
        ↓
Temporal Flood Signal

===============================================================
IMPORTANT
===============================================================

The satellite benchmark dataset is not India-specific.
Therefore, CNN validation results should be reported as
benchmark/prototype results and not as India-specific accuracy.

The current CNN performs:
Flooded vs Non-Flooded classification.

Road blockage detection requires separate road-level
ground-truth labels and should not be claimed from flood masks alone.

===============================================================
STAGE 2 DATA PACKAGE
===============================================================
