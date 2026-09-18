"""NYC Taxi Fare Predictor - Streamlit deployment app.

Loads the trained ensemble (DNN + LightGBM, geo-clustered with KMeans) from
../models/ via predict_fare.py and serves an interactive click-a-map interface.
"""
import json
import sys
from datetime import datetime, date, time as dtime
from pathlib import Path

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
sys.path.insert(0, str(MODELS_DIR))

st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="\U0001F695", layout="wide")

# NYC metro bounding box used only for client-side input sanity-checking / warnings
# (the model itself will still produce a number outside this box, just unreliably).
NYC_BOUNDS = dict(min_lat=40.40, max_lat=41.10, min_lon=-74.35, max_lon=-73.60)
DEFAULT_CENTER = [40.7549, -73.9840]  # Midtown Manhattan


@st.cache_resource(show_spinner="Loading trained models (DNN + LightGBM + geo-clusters)...")
def load_predictor():
    import predict_fare  # noqa: local module in models/, resolves its sibling files via __file__
    return predict_fare


@st.cache_data
def load_config():
    with open(MODELS_DIR / "deployment_config.json") as f:
        return json.load(f)


pf = load_predictor()
CONFIG = load_config()
LANDMARKS = {**CONFIG.get("airports", {}), **CONFIG.get("landmarks", {})}
LANDMARK_LABELS = {
    "jfk": "JFK Airport", "lga": "LaGuardia Airport", "ewr": "Newark Airport",
    "midtown": "Midtown", "downtown": "Downtown / Financial District",
    "central_park": "Central Park",
}

for key, default in [("pickup", None), ("dropoff", None), ("click_target", "Pickup")]:
    if key not in st.session_state:
        st.session_state[key] = default


def in_nyc_bounds(lat, lon):
    return (NYC_BOUNDS["min_lat"] <= lat <= NYC_BOUNDS["max_lat"]
            and NYC_BOUNDS["min_lon"] <= lon <= NYC_BOUNDS["max_lon"])


def haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(min(1, a ** 0.5))


def set_point(role, lat, lon):
    st.session_state[role.lower()] = [round(lat, 6), round(lon, 6)]


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address(query):
    """Free-text address/place lookup via OSM Nominatim, biased to the NYC metro area.
    Returns (lat, lon, display_name) or None if nothing matched."""
    if not query or not query.strip():
        return None
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "bounded": 1,
        "viewbox": (f"{NYC_BOUNDS['min_lon']},{NYC_BOUNDS['max_lat']},"
                    f"{NYC_BOUNDS['max_lon']},{NYC_BOUNDS['min_lat']}"),
    }
    headers = {"User-Agent": "nyc-taxi-fare-predictor-lab-assignment/1.0"}
    resp = requests.get("https://nominatim.openstreetmap.org/search",
                         params=params, headers=headers, timeout=6)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    r = results[0]
    return float(r["lat"]), float(r["lon"]), r.get("display_name", query)


def apply_scenario(name):
    scenarios = {
        "short": dict(pickup=[40.7549, -73.9840], dropoff=[40.7484, -73.9857],
                      d=date(2013, 6, 12), t=dtime(13, 30), p=1),
        "airport": dict(pickup=[40.7549, -73.9840], dropoff=LANDMARKS.get("jfk", [40.6413, -73.7781]),
                         d=date(2013, 8, 20), t=dtime(17, 45), p=2),
        "borderline": dict(pickup=[40.7420, -73.9890], dropoff=[40.7831, -73.9712],
                            d=date(2012, 3, 3), t=dtime(4, 15), p=1),
        "invalid_same": dict(pickup=[40.7549, -73.9840], dropoff=[40.7549, -73.9840],
                              d=date(2013, 1, 1), t=dtime(9, 0), p=1),
    }
    s = scenarios[name]
    st.session_state.pickup = s["pickup"]
    st.session_state.dropoff = s["dropoff"]
    st.session_state.scenario_date = s["d"]
    st.session_state.scenario_time = s["t"]
    st.session_state.scenario_passengers = s["p"]


st.title("\U0001F695 NYC Taxi Fare Predictor")
st.caption(
    "LightGBM model (selected from a DNN + LightGBM ensemble evaluated during training), trained "
    "on the full NYC Taxi Fare Kaggle dataset with geo-clustered pickup/drop-off zones. Click the "
    "map to set pickup and drop-off points."
)

map_col, form_col = st.columns([3, 2], gap="large")

with map_col:
    top = st.columns([1, 1, 1])
    with top[0]:
        st.session_state.click_target = st.radio(
            "Next map click sets:", ["Pickup", "Drop-off"],
            index=0 if st.session_state.click_target == "Pickup" else 1,
            horizontal=True,
        )
    with top[1]:
        if st.button("Clear points"):
            st.session_state.pickup = None
            st.session_state.dropoff = None
            st.session_state["_last_geocode_match"] = None
            st.rerun()
    with top[2]:
        with st.popover("Quick-fill a landmark"):
            for key, label in LANDMARK_LABELS.items():
                if key in LANDMARKS and st.button(label, key=f"lm_{key}"):
                    lat, lon = LANDMARKS[key]
                    set_point(st.session_state.click_target, lat, lon)
                    st.session_state["_last_geocode_match"] = None
                    st.rerun()

    with st.form("address_search_form", clear_on_submit=False):
        addr_col, btn_col = st.columns([4, 1])
        address_query = addr_col.text_input(
            f"Or type an address / place name — sets **{st.session_state.click_target}**",
            placeholder="e.g. Times Square, or 350 5th Ave, New York",
            label_visibility="visible",
        )
        addr_submitted = btn_col.form_submit_button("Find", use_container_width=True)

    if addr_submitted:
        try:
            result = geocode_address(address_query)
        except requests.exceptions.RequestException:
            result = "error"

        if result == "error":
            st.error("Geocoding service is unreachable right now — try again, or click the map instead.")
        elif result is None:
            st.warning(f"No NYC-area location found for “{address_query}”. Try a more specific address.")
        else:
            lat, lon, display_name = result
            set_point(st.session_state.click_target, lat, lon)
            st.session_state["_last_geocode_match"] = display_name
            st.rerun()

    if st.session_state.get("_last_geocode_match"):
        st.caption(f"✅ Matched: {st.session_state['_last_geocode_match']}")

    center = st.session_state.pickup or DEFAULT_CENTER
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
    if st.session_state.pickup:
        folium.Marker(st.session_state.pickup, tooltip="Pickup",
                       icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    if st.session_state.dropoff:
        folium.Marker(st.session_state.dropoff, tooltip="Drop-off",
                       icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(m)
    if st.session_state.pickup and st.session_state.dropoff:
        folium.PolyLine([st.session_state.pickup, st.session_state.dropoff],
                         color="#1e88e5", weight=3, dash_array="6,6").add_to(m)

    map_data = st_folium(m, height=480, use_container_width=True, key="taxi_map",
                          returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        clicked = (round(lat, 6), round(lon, 6))
        if clicked != st.session_state.get("_last_processed_click"):
            st.session_state["_last_processed_click"] = clicked
            set_point(st.session_state.click_target, lat, lon)
            st.session_state["_last_geocode_match"] = None
            st.rerun()

    st.markdown("**Try a preset test scenario** (Section 20 of the assignment brief):")
    sc = st.columns(4)
    if sc[0].button("Short local hop"):
        apply_scenario("short"); st.rerun()
    if sc[1].button("Airport run (JFK)"):
        apply_scenario("airport"); st.rerun()
    if sc[2].button("Borderline (4am, cross-town)"):
        apply_scenario("borderline"); st.rerun()
    if sc[3].button("Invalid: same point"):
        apply_scenario("invalid_same"); st.rerun()

with form_col:
    st.subheader("Trip details")

    p_lat, p_lon = (st.session_state.pickup or [None, None])
    d_lat, d_lon = (st.session_state.dropoff or [None, None])

    c1, c2 = st.columns(2)
    c1.markdown(f"**Pickup**  \n{p_lat:.4f}, {p_lon:.4f}" if p_lat is not None else "**Pickup**  \n*not set*")
    c2.markdown(f"**Drop-off**  \n{d_lat:.4f}, {d_lon:.4f}" if d_lat is not None else "**Drop-off**  \n*not set*")

    trip_date = st.date_input("Pickup date", value=st.session_state.get("scenario_date", date(2013, 6, 15)),
                               min_value=date(2009, 1, 1), max_value=date(2016, 12, 31))
    trip_time = st.time_input("Pickup time", value=st.session_state.get("scenario_time", dtime(12, 0)))
    passengers = st.number_input("Passenger count", min_value=1, max_value=6,
                                  value=st.session_state.get("scenario_passengers", 1), step=1)

    train_years = CONFIG.get("years", [])
    if train_years and trip_date.year not in train_years:
        st.info(
            f"The model was trained on trips from {min(train_years)}-{max(train_years)}. "
            f"A {trip_date.year} date is outside that range, so the year effect falls back to a "
            "default and the estimate may be less reliable.",
            icon="ℹ️",
        )

    ready = st.session_state.pickup is not None and st.session_state.dropoff is not None
    warn_msgs = []
    if ready:
        if st.session_state.pickup == st.session_state.dropoff:
            warn_msgs.append("Pickup and drop-off are the same point (zero-distance trip).")
        for role, pt in [("Pickup", st.session_state.pickup), ("Drop-off", st.session_state.dropoff)]:
            if not in_nyc_bounds(*pt):
                warn_msgs.append(f"{role} is well outside the NYC metro area the model was trained on.")

    for w in warn_msgs:
        st.warning(w, icon="⚠️")

    predict_clicked = st.button("Estimate Fare", type="primary", disabled=not ready,
                                 use_container_width=True)
    if not ready:
        st.caption("Set both a pickup and a drop-off point on the map to enable prediction.")

    if predict_clicked:
        pickup_dt = datetime.combine(trip_date, trip_time)
        try:
            fare = pf.predict_fare(
                pickup_dt.isoformat(sep=" "),
                st.session_state.pickup[1], st.session_state.pickup[0],
                st.session_state.dropoff[1], st.session_state.dropoff[0],
                int(passengers),
            )
            distance_km = haversine_km(st.session_state.pickup[0], st.session_state.pickup[1],
                                        st.session_state.dropoff[0], st.session_state.dropoff[1])
            st.success("Prediction complete")
            r1, r2 = st.columns(2)
            r1.metric("Estimated Taxi Fare", f"${fare:,.2f}")
            r2.metric("Straight-line Distance", f"{distance_km:.2f} km")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    with st.expander("Model performance (held-out test set)"):
        for name, m_key in [("LightGBM (deployed)", "lgbm_metrics"), ("DNN (trained, not deployed)", "dnn_metrics"), ("Ensemble (trained, not deployed)", "ensemble_metrics")]:
            metrics = CONFIG.get(m_key)
            if metrics:
                st.markdown(f"**{name}** — MAE: ${metrics['MAE']:.3f} | RMSE: ${metrics['RMSE']:.3f} | R²: {metrics['R2']:.4f}")
        note = CONFIG.get("deployment_note")
        if note:
            st.caption(note)
