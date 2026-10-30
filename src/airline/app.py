
"""BTS Airline — Flight Intelligence Dashboard."""

from datetime import date, time
from pathlib import Path
import logging
import os

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "airline_processed.csv"
)

API_URL = os.getenv("AIRLINE_API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL.rstrip('/')}/predict"
HEALTH_URL = f"{API_URL.rstrip('/')}/health"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="BTS Airline | Flight Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_airline_data():
    """Load the processed BTS dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    required_columns = [
        "YEAR",
        "QUARTER",
        "MONTH",
        "DAY_OF_WEEK",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "DISTANCE",
        "DEP_HOUR",
        "OP_UNIQUE_CARRIER",
        "TAIL_NUM",
        "ORIGIN",
        "ORIGIN_CITY_NAME",
        "ORIGIN_STATE_ABR",
        "DEST",
        "DEST_CITY_NAME",
        "DEST_STATE_ABR",
        "TIME_OF_DAY",
        "ROUTE",
        "DISTANCE_CATEGORY",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {sorted(missing)}"
        )

    return df


@st.cache_data
def prepare_metadata(df):
    """Build selection options and route lookups from observed data."""

    clean = df.copy()

    categorical_columns = [
        "OP_UNIQUE_CARRIER",
        "TAIL_NUM",
        "ORIGIN",
        "ORIGIN_CITY_NAME",
        "ORIGIN_STATE_ABR",
        "DEST",
        "DEST_CITY_NAME",
        "DEST_STATE_ABR",
        "TIME_OF_DAY",
        "ROUTE",
        "DISTANCE_CATEGORY",
    ]

    for column in categorical_columns:
        clean[column] = clean[column].astype("string").str.strip()

    clean["DISTANCE"] = pd.to_numeric(
        clean["DISTANCE"], errors="coerce"
    )
    clean["DEP_HOUR"] = pd.to_numeric(
        clean["DEP_HOUR"], errors="coerce"
    )

    origins = (
        clean[["ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR"]]
        .dropna()
        .drop_duplicates()
        .sort_values("ORIGIN")
        .drop_duplicates("ORIGIN")
        .set_index("ORIGIN")
        .to_dict("index")
    )

    destinations = (
        clean[["DEST", "DEST_CITY_NAME", "DEST_STATE_ABR"]]
        .dropna()
        .drop_duplicates()
        .sort_values("DEST")
        .drop_duplicates("DEST")
        .set_index("DEST")
        .to_dict("index")
    )

    route_data = clean[
        [
            "ORIGIN",
            "DEST",
            "ROUTE",
            "DISTANCE",
            "DISTANCE_CATEGORY",
        ]
    ].dropna()

    route_data = route_data[
        (route_data["ORIGIN"] != "")
        & (route_data["DEST"] != "")
        & (route_data["ROUTE"] != "")
        & (route_data["DISTANCE"] >= 0)
    ]

    route_metadata = (
        route_data.groupby(["ORIGIN", "DEST"])
        .agg(
            route=("ROUTE", lambda values: values.mode().iloc[0]),
            distance=("DISTANCE", "median"),
            distance_category=(
                "DISTANCE_CATEGORY",
                lambda values: values.mode().iloc[0],
            ),
        )
        .to_dict("index")
    )

    destinations_by_origin = {}

    for origin, destination in route_metadata:
        destinations_by_origin.setdefault(origin, []).append(destination)

    destinations_by_origin = {
        origin: sorted(set(values))
        for origin, values in destinations_by_origin.items()
    }

    time_data = clean[["DEP_HOUR", "TIME_OF_DAY"]].dropna()

    time_data = time_data[
        time_data["DEP_HOUR"].between(0, 23)
        & (time_data["TIME_OF_DAY"] != "")
    ]

    time_by_hour = (
        time_data.groupby("DEP_HOUR")["TIME_OF_DAY"]
        .agg(lambda values: values.mode().iloc[0])
        .to_dict()
    )

    carriers = sorted(
        clean["OP_UNIQUE_CARRIER"].dropna().unique().tolist()
    )

    tail_numbers = sorted(
        clean["TAIL_NUM"].dropna().unique().tolist()
    )

    return {
        "origins": origins,
        "destinations": destinations,
        "destinations_by_origin": destinations_by_origin,
        "routes": route_metadata,
        "time_by_hour": time_by_hour,
        "carriers": carriers,
        "tail_numbers": tail_numbers,
        "rows": len(clean),
    }


def get_time_category(hour, lookup):
    """Return the most representative observed category for an hour."""

    if not lookup:
        raise ValueError("No time-of-day categories are available.")

    if hour in lookup:
        return lookup[hour]

    nearest_hour = min(
        lookup,
        key=lambda available_hour: abs(available_hour - hour),
    )

    return lookup[nearest_hour]


# ============================================================
# INITIALIZATION
# ============================================================

try:
    airline_df = load_airline_data()
    metadata = prepare_metadata(airline_df)

except Exception as error:
    st.error(f"Application initialization failed: {error}")
    st.stop()


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    :root {
        --background: #080e1b;
        --surface: #101a2c;
        --surface-light: #16243a;
        --border: #263750;
        --text: #f1f5f9;
        --muted: #91a4bd;
        --cyan: #36c5f0;
        --blue: #3985ff;
        --green: #39d6a0;
        --red: #ff7188;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 15% 0%,
                rgba(36, 112, 190, 0.15),
                transparent 35%
            ),
            radial-gradient(
                ellipse at 90% 15%,
                rgba(39, 196, 220, 0.08),
                transparent 30%
            ),
            var(--background);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
    }

    p, label {
        color: #cbd5e1;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.15);
        margin-bottom: 2.2rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: 0.02em;
    }

    .brand-icon {
        width: 39px;
        height: 39px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: white;
        font-size: 1.2rem;
        background: linear-gradient(135deg, #168eea, #27c5dc);
        box-shadow: 0 5px 25px rgba(39,160,235,0.25);
    }

    .top-label {
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .hero {
        padding: 1rem 0 2rem 0;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.6rem, 6vw, 4.7rem);
        font-weight: 700;
        line-height: 1.03;
        letter-spacing: -0.055em;
        color: #f8fafc;
        margin: 0.8rem 0;
    }

    .hero-title span {
        background: linear-gradient(90deg, #49c6ef, #528eff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 650px;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.8;
    }

    .hero-tag {
        display: inline-block;
        padding: 7px 12px;
        margin-top: 1.1rem;
        border-radius: 30px;
        border: 1px solid rgba(54,197,240,0.25);
        background: rgba(54,197,240,0.08);
        color: #6bd7f4;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.84rem;
        margin-bottom: 1.2rem;
    }

    .panel {
        background: linear-gradient(
            145deg,
            rgba(22,36,58,0.96),
            rgba(14,25,43,0.96)
        );
        border: 1px solid var(--border);
        border-radius: 19px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }

    .panel-label {
        font-size: 0.7rem;
        color: #69ccef;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .route-display {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        background: rgba(7,15,29,0.65);
        border: 1px solid #253a56;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    .airport-code {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .airport-city {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 3px;
    }

    .route-line {
        flex: 1;
        text-align: center;
        color: var(--cyan);
        font-size: 1.3rem;
    }

    .metric-card {
        background: rgba(7,15,29,0.58);
        border: 1px solid #263a55;
        border-radius: 13px;
        padding: 1rem;
        min-height: 82px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .metric-value {
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.45rem;
    }

    .api-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 13px;
        border-radius: 30px;
        background: rgba(57,214,160,0.09);
        border: 1px solid rgba(57,214,160,0.25);
        color: #54e1b1;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .api-pill.offline {
        background: rgba(255,113,136,0.08);
        border-color: rgba(255,113,136,0.25);
        color: #ff91a2;
    }

    .result-card {
        border-radius: 19px;
        padding: 2rem;
        margin-top: 1.3rem;
        border: 1px solid #29415e;
        background: linear-gradient(
            135deg,
            rgba(23,40,65,0.98),
            rgba(12,25,43,0.98)
        );
    }

    .result-card.delayed {
        border-color: rgba(255,113,136,0.45);
        box-shadow: 0 12px 45px rgba(255,80,110,0.08);
    }

    .result-card.undelayed {
        border-color: rgba(57,214,160,0.4);
        box-shadow: 0 12px 45px rgba(57,214,160,0.07);
    }

    .result-kicker {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    .result-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 700;
        margin: 0.7rem 0;
    }

    .result-heading.delayed {
        color: var(--red);
    }

    .result-heading.undelayed {
        color: var(--green);
    }

    .result-description {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.7;
    }

    .result-footer {
        border-top: 1px solid #30445e;
        margin-top: 1.5rem;
        padding-top: 1rem;
        color: #a7b8cc;
        font-size: 0.78rem;
        line-height: 1.9;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #0b1526;
        border-color: #30445e;
        border-radius: 10px;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {
        color: #f1f5f9;
    }

    div[data-testid="stDateInput"] input,
    div[data-testid="stTimeInput"] input {
        color: #f1f5f9;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(100deg, #168eea, #28b8df);
        color: white;
        border: none;
        border-radius: 11px;
        min-height: 3.1rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        box-shadow: 0 7px 22px rgba(22,142,234,0.2);
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        color: white;
        border: none;
        filter: brightness(1.12);
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] {
        background: rgba(7,15,29,0.58);
        border: 1px solid #263a55;
        padding: 0.9rem;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    hr {
        border-color: #263750;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.75rem;
        padding-top: 2rem;
        letter-spacing: 0.08em;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-title {
            font-size: 2.8rem;
        }

        .panel {
            padding: 1rem;
        }

        .airport-code {
            font-size: 1.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-icon">✈</div>
            BTS AIRLINE
        </div>
        <div class="top-label">
            MACHINE LEARNING PLATFORM &nbsp; / &nbsp; 01
        </div>
    </div>

    <div class="hero">
        <div class="eyebrow">AVIATION ANALYTICS · PREDICTIVE INTELLIGENCE</div>
        <div class="hero-title">
            Every flight has<br>
            a <span>story.</span>
        </div>
        <div class="hero-description">
            Explore flight details and uncover potential departure delays
            with machine learning powered by historical BTS airline data.
        </div>
        <div class="hero-tag">GRADIENT BOOSTING · ML MODEL V2</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API STATUS
# ============================================================

try:
    health_response = requests.get(HEALTH_URL, timeout=4)
    health_response.raise_for_status()

    api_connected = health_response.json().get("model_loaded", False)

except requests.RequestException:
    api_connected = False

if api_connected:
    st.markdown(
        '<div class="api-pill">● &nbsp; PREDICTION ENGINE ONLINE</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="api-pill offline">● &nbsp; API OFFLINE — START FASTAPI</div>',
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# FLIGHT PLANNER
# ============================================================

st.markdown(
    """
    <div class="section-title">Plan your flight</div>
    <div class="section-subtitle">
        Select your journey and configure the scheduled flight details.
    </div>
    """,
    unsafe_allow_html=True,
)

left_column, right_column = st.columns([1.05, 0.95], gap="large")


# ------------------------------------------------------------
# LEFT: ROUTE
# ------------------------------------------------------------

with left_column:

    st.markdown(
        """
        <div class="panel">
            <div class="panel-label">01 — JOURNEY DETAILS</div>
        """,
        unsafe_allow_html=True,
    )

    origin_options = list(metadata["origins"].keys())

    origin = st.selectbox(
        "FROM · Origin airport",
        options=origin_options,
        format_func=lambda code: (
            f"{code}  ·  "
            f"{metadata['origins'][code]['ORIGIN_CITY_NAME']}"
        ),
        key="origin_airport",
    )

    destination_options = metadata["destinations_by_origin"].get(
        origin, []
    )

    if destination_options:
        destination = st.selectbox(
            "TO · Destination airport",
            options=destination_options,
            format_func=lambda code: (
                f"{code}  ·  "
                f"{metadata['destinations'][code]['DEST_CITY_NAME']}"
            ),
            key="destination_airport",
        )
    else:
        destination = None
        st.warning("No destination routes found for this airport.")

    route_info = (
        metadata["routes"].get((origin, destination))
        if destination is not None
        else None
    )

    if route_info:
        origin_info = metadata["origins"][origin]
        destination_info = metadata["destinations"][destination]

        st.markdown(
            f"""
            <div class="route-display">
                <div>
                    <div class="airport-code">{origin}</div>
                    <div class="airport-city">
                        {origin_info['ORIGIN_CITY_NAME']}
                    </div>
                </div>
                <div class="route-line">━━━━ ✈ ━━━━</div>
                <div style="text-align:right">
                    <div class="airport-code">{destination}</div>
                    <div class="airport-city">
                        {destination_info['DEST_CITY_NAME']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Flight distance",
                f"{route_info['distance']:,.0f} miles",
            )

        with metric2:
            st.metric(
                "Distance category",
                str(route_info["distance_category"]),
            )

    else:
        st.info("Select an available route to see its details.")

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# RIGHT: FLIGHT DETAILS
# ------------------------------------------------------------

with right_column:

    st.markdown(
        """
        <div class="panel">
            <div class="panel-label">02 — FLIGHT CONFIGURATION</div>
        """,
        unsafe_allow_html=True,
    )

    flight_date = st.date_input(
        "Flight date",
        value=date.today(),
        key="flight_date",
    )

    col1, col2 = st.columns(2)

    with col1:
        departure_time = st.time_input(
            "Departure time",
            value=time(8, 0),
            step=300,
            key="departure_time",
        )

    with col2:
        arrival_time = st.time_input(
            "Arrival time",
            value=time(10, 0),
            step=300,
            key="arrival_time",
        )

    carrier = st.selectbox(
        "Operating airline",
        options=metadata["carriers"],
        key="airline_carrier",
    )

    tail_number = st.selectbox(
        "Aircraft registration",
        options=metadata["tail_numbers"],
        key="aircraft_tail",
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# AUTOMATIC FEATURES
# ============================================================

st.markdown(
    """
    <div class="section-title">Flight intelligence</div>
    <div class="section-subtitle">
        Additional model features are derived automatically from
        the selected flight information and historical dataset.
    </div>
    """,
    unsafe_allow_html=True,
)

if route_info and destination is not None:

    departure_hour = departure_time.hour

    try:
        time_of_day = get_time_category(
            departure_hour,
            metadata["time_by_hour"],
        )

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">ROUTE</div>
                    <div class="metric-value">{origin} → {destination}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">TIME CATEGORY</div>
                    <div class="metric-value">{time_of_day}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">DISTANCE</div>
                    <div class="metric-value">
                        {route_info['distance']:,.0f} mi
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">AIRLINE</div>
                    <div class="metric-value">{carrier}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Could not derive flight features: {error}")
        time_of_day = None

else:
    time_of_day = None


# ============================================================
# PREDICTION
# ============================================================

st.write("")

st.markdown(
    """
    <div class="section-title">03 — Prediction analysis</div>
    <div class="section-subtitle">
        Your flight details are sent to the FastAPI prediction service.
    </div>
    """,
    unsafe_allow_html=True,
)

predict_column, info_column = st.columns([1.2, 0.8])

with predict_column:
    predict_clicked = st.button(
        "✦  ANALYZE FLIGHT",
        use_container_width=True,
        disabled=not api_connected,
    )

with info_column:
    st.markdown(
        """
        <div style="
            padding: 0.85rem 0;
            color: #91a4bd;
            font-size: 0.8rem;
        ">
            Powered by the registered BTS Gradient Boosting model.
        </div>
        """,
        unsafe_allow_html=True,
    )


if predict_clicked:

    if not route_info or destination is None:
        st.error("Please select a valid flight route.")

    elif not time_of_day:
        st.error("Unable to determine the time-of-day category.")

    else:
        # BTS convention: Sunday=1, Monday=2, ..., Saturday=7.
        bts_day_of_week = ((flight_date.weekday() + 1) % 7) + 1

        request_data = {
            "YEAR": flight_date.year,
            "QUARTER": (flight_date.month - 1) // 3 + 1,
            "MONTH": flight_date.month,
            "DAY_OF_WEEK": bts_day_of_week,
            "CRS_DEP_TIME": (
                departure_time.hour * 100
                + departure_time.minute
            ),
            "CRS_ARR_TIME": (
                arrival_time.hour * 100
                + arrival_time.minute
            ),
            "DISTANCE": float(route_info["distance"]),
            "DEP_HOUR": departure_time.hour,
            "OP_UNIQUE_CARRIER": carrier,
            "TAIL_NUM": tail_number,
            "ORIGIN": origin,
            "ORIGIN_CITY_NAME": (
                metadata["origins"][origin]["ORIGIN_CITY_NAME"]
            ),
            "ORIGIN_STATE_ABR": (
                metadata["origins"][origin]["ORIGIN_STATE_ABR"]
            ),
            "DEST": destination,
            "DEST_CITY_NAME": (
                metadata["destinations"][destination]["DEST_CITY_NAME"]
            ),
            "DEST_STATE_ABR": (
                metadata["destinations"][destination]["DEST_STATE_ABR"]
            ),
            "TIME_OF_DAY": str(time_of_day),
            "ROUTE": str(route_info["route"]),
            "DISTANCE_CATEGORY": str(route_info["distance_category"]),
        }

        with st.spinner("Analyzing flight data..."):

            try:
                response = requests.post(
                    PREDICT_URL,
                    json=request_data,
                    timeout=30,
                )

                response.raise_for_status()
                result = response.json()

                prediction = int(result["prediction"])
                prediction_label = result["prediction_label"]

                if prediction not in (0, 1):
                    raise ValueError("Unexpected prediction class.")

                expected_label = {
                    0: "Undelayed",
                    1: "Delayed",
                }[prediction]

                if prediction_label != expected_label:
                    raise ValueError(
                        "Prediction class and label do not match."
                    )

                model_name = result["model"]
                model_version = result["model_version"]

                result_style = (
                    "delayed"
                    if prediction_label == "Delayed"
                    else "undelayed"
                )

                st.markdown(
                    f"""
                    <div class="result-card {result_style}">
                        <div class="result-kicker">
                            FLIGHT ANALYSIS COMPLETE
                        </div>
                        <div class="result-heading {result_style}">
                            {prediction_label}
                        </div>
                        <div class="result-description">
                            The model classified this flight as
                            <strong>{prediction_label.lower()}</strong>
                            based on the supplied flight characteristics.
                        </div>
                        <div class="result-footer">
                            PREDICTED CLASS: {prediction}<br>
                            MODEL: {model_name}<br>
                            MODEL VERSION: {model_version}<br>
                            ROUTE: {origin} → {destination}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Check the API server.")

            except requests.exceptions.Timeout:
                st.error("The prediction request timed out.")

            except requests.exceptions.HTTPError as error:
                st.error(
                    f"API error {error.response.status_code}: "
                    f"{error.response.text}"
                )

            except (KeyError, ValueError) as error:
                st.error(f"Invalid API response: {error}")

            except Exception:
                logger.exception("Unexpected prediction error.")
                st.error("An unexpected error occurred. Check the logs.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        BTS AIRLINE ANALYSIS &nbsp; · &nbsp;
        FLIGHT INTELLIGENCE PLATFORM &nbsp; · &nbsp;
        MACHINE LEARNING
    </div>
    """,
    unsafe_allow_html=True,
)
