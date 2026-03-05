# Data Folder

Place the UCI dataset CSV in this folder as `dataset.csv`.

Steps:
1. Download the ZIP from the UCI repository: "Predict Students' Dropout and Academic Success".
2. Extract `data.csv` from the ZIP.
3. Rename `data.csv` to `dataset.csv`.
4. Move `dataset.csv` into this `data/` folder.

The training script `model/train_model.py` expects the file to be at `data/dataset.csv` with semicolon (`;`) as the delimiter.
