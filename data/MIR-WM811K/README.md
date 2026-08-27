# MIR WM-811K — Local Dataset Files

This directory contains the Python version of the MIR WM-811K wafer-map dataset.
The dataset is stored as one serialized table, not as PNG/JPEG files grouped into class folders.

## Files

| File | Purpose |
|---|---|
| `WM811K.pkl` | Main dataset: a pandas DataFrame containing 811,457 wafer records and their wafer-map matrices. |
| `example.py` | Minimal script supplied with the dataset to load the pickle and display one map. |
| `license.txt` | Original license, citation requirements, and source information. |

## How to access the data

```python
import pandas as pd

df = pd.read_pickle("data/MIR-WM811K/WM811K.pkl")
print(df.shape)
print(df.columns)
```

Loading the pickle creates a DataFrame in memory; it does **not** extract image files.
Each row represents one wafer, and `waferMap` contains its image as a 2D numeric array.
Typical additional fields describe the failure type, train/test assignment, lot, and wafer size.

```python
import matplotlib.pyplot as plt

plt.imshow(df.iloc[0]["waferMap"])
plt.axis("off")
plt.show()
```

The original archive also provides `WM811K.mat` for MATLAB users.
It is an alternative representation of the same dataset and is not required for this Python project.
Keep the original `license.txt` because it contains the dataset license and mandatory citations.
