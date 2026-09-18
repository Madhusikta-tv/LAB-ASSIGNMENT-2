# Standalone NYC taxi fare inference using LightGBM only.
#
# The trained ensemble was DNN + LightGBM, but the saved metrics showed the DNN
# contributed almost nothing (ensemble MAE $1.355 vs LightGBM-alone MAE $1.3545 -
# a $0.0005 difference) while requiring a full TensorFlow runtime plus loading the
# .keras model in memory. For deployment on a memory-constrained free-tier host,
# that trade isn't worth it, so this drops the DNN and serves LightGBM directly.
# The trained nyc_taxi_dnn.keras is kept in this folder for the report/notebook
# record, but is intentionally not loaded here.
#
# Requires: lightgbm, pandas, numpy, joblib, scikit-learn.
import json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, joblib

_H = Path(__file__).parent
_C = json.load(open(_H / "deployment_config.json"))
_LGBM = joblib.load(_H / "nyc_taxi_lgbm.joblib")
_KP = joblib.load(_H / "kmeans_pickup.joblib")
_KD = joblib.load(_H / "kmeans_dropoff.joblib")
_FEAT = _C["feature_order"]
_RUSH = set(_C["rush_hours"]); _YEARS = _C["years"]
_LOG = _C["use_log"]
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
    return float(_inv(_LGBM.predict(X_lgb)[0]))

if __name__ == "__main__":
    print("Estimated fare: $%.2f" % predict_fare("2013-07-06 17:18:00", -73.9827, 40.7680, -73.9855, 40.7484, 1))
