# Task 01: Explore and Validate the WM-811K Dataset

Class counts, labeled/unlabeled counts, wafer dimensions, examples per class, data encoding, anomalies, etc.

## Goal

Understand the original WM-811K dataset before choosing preprocessing operations or a deep learning model. The analysis must treat wafer maps as discrete spatial data, preserve the source observations, and verify assumptions empirically without selecting strategies solely by assumption.

## Tasks & Checklist

- [x] **Load original dataset**: Load `data/MIR-WM811K/WM811K.pkl` and inspect schema, fields, types, and object-valued entries.
- [x] **Label availability**: Differentiate labeled from unlabeled observations and inspect train/test representation.
- [x] **Encoding validation**: Verify wafer-map encoding across the complete dataset (check invalid types, empty arrays, non-2D maps, NaNs, infinities).
- [x] **Class distribution & imbalance**: Report labeled class counts, relative frequencies, dominant/rarest classes, and imbalance ratios.
- [x] **Dimension analysis**: Analyze wafer height, width, area, aspect ratio, and exact shape frequencies without resizing or dropping maps.
- [x] **Die statistics**: Quantify background, functional, defective, and active dies (defective-die and occupancy ratios).
- [x] **Visualizations**: Visualize representative maps from every class using consistent discrete color encoding.
- [x] **Sanity checks**: Check for malformed maps, metadata inconsistencies, inactive wafers, and duplicate identifiers.
- [x] **Analyze `none` class**: Examine defect content, variability, and relationship to named failure classes.
- [x] **Document observations**: Separate empirical findings from proposed preprocessing and modeling responses.

## Deliverables

- Executable and documented EDA notebook: [`notebooks/preparation/eda_wm811k_validation.ipynb`](../../notebooks/preparation/eda_wm811k_validation.ipynb).
- Reproducible tables and interpretations.
- Descriptively named figures saved under `output/images/`.
- Written summary report: [`docs/reports/eda_report.md`](../reports/eda_report.md).
