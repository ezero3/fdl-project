# Specification 01: Explore and Validate WM-811K

## Goal

Understand the original WM-811K dataset before choosing preprocessing operations or a deep learning model. The analysis must treat wafer maps as discrete spatial data, preserve the source observations, and verify assumptions empirically.

## Required analyses

- Load the original `data/MIR-WM811K/WM811K.pkl` dataset and inspect its schema, fields, types, and object-valued entries.
- Determine how failure labels and train/test assignments are represented, then distinguish labeled from unlabeled observations.
- Verify the wafer-map representation and encoding across the complete dataset, including unexpected values, invalid types, empty arrays, non-2D maps, NaNs, and infinities.
- Report labeled class counts, relative frequencies, the dominant and rarest classes, and the majority/minority imbalance ratio.
- Analyze wafer height, width, area, aspect ratio, exact shape frequencies, and unusual dimension tails without resizing or removing maps.
- Quantify background, functional, defective, and active dies, including defective-die and occupancy ratios.
- Visualize several representative maps from every labeled class using a consistent discrete color encoding.
- Check for malformed or unusual maps, metadata inconsistencies, inactive wafers, and duplicate `(lotName, waferIndex)` identifiers.
- Investigate the dominant `none` class separately, including its defect content, variability, and relationship to named failure classes.
- Separate observed facts from proposed preprocessing and modeling responses.

## Deliverables

- An executable and documented EDA notebook under `notebooks/`.
- Reproducible tables and interpretations for every required analysis.
- Descriptively named figures saved under `output/images/`.
- A concise report of findings and their implications for subsequent preprocessing, training, and evaluation.

No preprocessing strategy, cleaning rule, or model architecture should be selected solely by assumption during this task.
