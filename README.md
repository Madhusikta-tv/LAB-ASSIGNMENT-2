# Assignment 02 — NYC Taxi Fare Prediction (Deep Feedforward Neural Network)

## Status
- [x] Model notebook built: `notebooks/01_nyc_taxi_fare_model.ipynb`
- [ ] Full-dataset run executed on Colab (produces `models/nyc_taxi_dnn.keras` + `models/deployment_config.json`)
- [ ] Deployment app (Streamlit) — next step, after the model artifacts exist

## How to run the notebook (Google Colab)

1. Upload `notebooks/01_nyc_taxi_fare_model.ipynb` to Google Colab (or open it directly from Drive/GitHub).
2. `Runtime → Change runtime type → GPU (T4)`.
3. Run cells top to bottom. On first run leave `QUICK_TEST_MODE = True` (Section 1) — this
   caps the streaming pass to ~1M raw rows and runs a tiny tuning/training budget so the *entire*
   pipeline (download → clean → engineer → shard → train → tune → evaluate → save) finishes in a
   few minutes and you can confirm nothing is broken before committing to the full run.
4. You will be prompted to upload your `kaggle.json` API token (Kaggle → Account → Create New API
   Token) — needed to download the ~5.7 GB competition file directly into the Colab VM.
5. Once the quick test passes end-to-end, set `QUICK_TEST_MODE = False` and re-run from the top.
   This is the real ~55M-row run — expect the streaming pass and full training to take a while;
   that's expected for a genuine full-dataset deep learning pipeline.
6. Section 22 saves the trained model + a `deployment_config.json` (scaler stats, feature list,
   NYC bounds, airport coordinates, etc. — everything needed to reproduce training-time
   preprocessing at inference time) and zips them into `deployment_artifacts.zip`. Download that
   zip and unzip it into this project's `models/` folder locally.

## Why this pipeline is built the way it is

The full Kaggle `train.csv` is ~55.4M rows / 5.7 GB — too large to comfortably hold in memory
repeatedly. The notebook makes **one single streaming pass** over the raw CSV that simultaneously
profiles data quality, samples rows for EDA plots, cleans, engineers features, computes scaler
statistics (Welford's algorithm, training split only), and writes compact `float32` Parquet
shards. Model training then reads those shards through a `tf.data` streaming pipeline, so peak RAM
usage is bounded by the shard size, not by the dataset's total size — see the notebook's Section 0
markdown cell and Section 7 for full details.

## Folder layout

```
Assignment_02_NYC_Taxi_Fare/
├── notebooks/
│   └── 01_nyc_taxi_fare_model.ipynb   # full EDA + preprocessing + DNN + tuning + evaluation
├── models/                            # trained model + deployment_config.json land here (from Colab)
├── app/                               # Streamlit deployment app (next step)
└── report/                            # lab report
```

## Next step
Once you've run the full-dataset notebook on Colab and placed the artifacts in `models/`, say the
word and we'll build the Streamlit deployment app in `app/`.
