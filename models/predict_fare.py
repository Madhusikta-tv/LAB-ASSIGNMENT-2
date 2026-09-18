# Standalone NYC taxi fare inference using the selected ensemble (DNN + LightGBM).
# Requires: tensorflow, lightgbm, pandas, numpy, joblib, scikit-learn.
import json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, joblib
from tensorflow import keras

_H = Path(__file__).parent
_C = json.load(open(_H / "deployment_config.json"))
_LGBM = joblib.load(_H / "nyc_taxi_lgbm.joblib")
_DNN = keras.models.load_model(_H / "nyc_taxi_dnn.keras")
_KP = joblib.load(_H / "kmeans_pickup.joblib")
_KD = joblib.load(_H / "kmeans_dropoff.joblib")
_FEAT = _C["feature_order"]; _NUM = _C["num_cols"]; _CAT = _C["cat_cols"]
_MEAN = np.array(_C["feature_mean"], np.float32); _STD = np.array(_C["feature_std"], np.float32)
_RUSH = set(_C["rush_hours"]); _YEARS = _C["years"]
_W = _C["ensemble_weight_dnn"]; _LOG = _C["use_log"]
def _inv(p): return np.expm1(p) if _LOG else p

def _hav(a, b, c, d):
    R = 6371.0; a, b, c, d = map(np.radians, [a, b, c, d]); dl, do = c - a, d - b
    x = np.sin(dl/2)**2 + np.cos(a)*np.cos(c)*np.sin(do/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(x, 0, 1)))

def _bearing(a, b, c, d):
    a, c = np.radians(a), np.radians(c); do = np.radians(d - b)
    return np.degrees(np.arctan2(np.sin(do)*np.cos(c), np.cos(a)*np.sin(c) - np.sin(a)*np.cos(c)*np.cos(do)))

def _features(pickup_datetime, plon, plat, dlon, dlat, passengers):
    dt = datetime.fromisoformat(str(pickup_datetime)); hour, dow = dt.hour, dt.weekday()
    f = {"pickup_hour": hour, "pickup_dow": dow, "pickup_month": dt.month,
         "year_idx": _YEARS.index(dt.year) if dt.year in _YEARS else 0,
         "pickup_cluster": int(_KP.predict(np.array([[plat, plon]], dtype=np.float64))[0]),
         "dropoff_cluster": int(_KD.predict(np.array([[dlat, dlon]], dtype=np.float64))[0]),
         "passenger_count": passengers, "pickup_longitude": plon, "pickup_latitude": plat,
         "dropoff_longitude": dlon, "dropoff_latitude": dlat,
         "distance_km": float(_hav(plat, plon, dlat, dlon)),
         "manhattan_km": float(_hav(plat, plon, dlat, plon) + _hav(dlat, plon, dlat, dlon)),
         "abs_lat_diff": abs(dlat - plat), "abs_lon_diff": abs(dlon - plon),
         "euclid_deg": float(((dlat - plat)**2 + (dlon - plon)**2) ** 0.5),
         "bearing": float(_bearing(plat, plon, dlat, dlon)),
         "distance_per_passenger": float(_hav(plat, plon, dlat, dlon)) / max(passengers, 1),
         "is_weekend": int(dow >= 5), "is_rush_hour": int(hour in _RUSH)}
    for n, (la, lo) in _C["airports"].items():
        f["%s_pickup" % n] = float(_hav(plat, plon, la, lo)); f["%s_dropoff" % n] = float(_hav(dlat, dlon, la, lo))
    for n, (la, lo) in _C["landmarks"].items():
        f["%s_pickup" % n] = float(_hav(plat, plon, la, lo)); f["%s_dropoff" % n] = float(_hav(dlat, dlon, la, lo))
    return f

def predict_fare(pickup_datetime, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat, passengers=1):
    f = _features(pickup_datetime, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat, passengers)
    X_lgb = pd.DataFrame([[f[c] for c in _FEAT]], columns=_FEAT)
    x_dnn = {c: np.array([[f[c]]], np.int32) for c in _CAT}
    x_dnn["num"] = (np.array([[f[c] for c in _NUM]], np.float32) - _MEAN) / _STD
    p_lgb = _inv(_LGBM.predict(X_lgb)[0])
    p_dnn = _inv(_DNN.predict(x_dnn, verbose=0).ravel()[0])
    return float(_W * p_dnn + (1 - _W) * p_lgb)

if __name__ == "__main__":
    print("Estimated fare: $%.2f" % predict_fare("2013-07-06 17:18:00", -73.9827, 40.7680, -73.9855, 40.7484, 1))
