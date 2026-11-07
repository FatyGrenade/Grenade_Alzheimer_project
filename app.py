
"""
POMENOVA — Nature Meets Intelligence
EthnoHack 2026 | Track 2: Natural products and Drug Discovery

Streamlit scientific research application.
Pomegranate phytochemicals × Human AChE × Machine Learning

Run:
    python -m streamlit run app.py
"""

from pathlib import Path
import base64
import html
import math

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Draw,
    Descriptors,
    rdMolDescriptors,
    rdFingerprintGenerator,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="POMENOVA | Nature Meets Intelligence",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
ML = ROOT / "ml"
ML_DATA = ML / "data"
FIGURES = ML / "figures"
REPORTS = ML / "reports"
MODEL_PATH = ML / "models" / "random_forest_v1.joblib"

GITHUB = (
    "https://github.com/"
    "FatyGrenade/Grenade_Alzheimer_project"
)

COMPOUND_FAMILIES = {
    "Punicalagin": "Ellagitannin",
    "Punicalin": "Ellagitannin",
    "Corilagin": "Ellagitannin",
    "Pedunculagin": "Ellagitannin",
    "Ellagic acid": "Phenolic compound",
    "Gallic acid": "Phenolic acid",
    "Catechin": "Flavonoid",
    "Epicatechin": "Flavonoid",
    "Quercetin": "Flavonoid",
    "Rutin": "Flavonoid",
}


# ============================================================
# 2. IMAGE UTILITIES
# ============================================================

@st.cache_data(show_spinner=False)
def image_base64(path_string, modified):
    path = Path(path_string)

    if not path.is_file():
        return ""

    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


def image_url(filename):
    path = ASSETS / filename

    if not path.is_file():
        return ""

    suffix = path.suffix.lower()

    mime = (
        "image/png"
        if suffix == ".png"
        else "image/jpeg"
    )

    encoded = image_base64(
        str(path),
        path.stat().st_mtime_ns,
    )

    return f"data:{mime};base64,{encoded}"


BACKGROUND_URL = image_url(
    "botanical_background.jpg"
)

BANNER_URL = image_url(
    "botanical_banner.jpg"
)

LOGO_URL = image_url(
    "pomenova_logo.png"
)


# ============================================================
# 3. VISUAL DESIGN — POMENOVA BOTANICAL EDITION
# ============================================================

st.markdown(
    """
<style>
@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700;800&display=swap'
);

:root {
    --berry: #92163e;
    --wine: #700c30;
    --rose: #d77988;
    --blush: #ffe9e4;
    --cream: #fff9f3;
    --green: #164b2c;
    --ink: #29382d;
    --gold: #d7a355;
}

html, body, .stApp {
    font-family: 'DM Sans', sans-serif !important;
}

.stApp {
    background-color: #fff4ee !important;
    background-image: url('BACKGROUND_IMAGE') !important;
    background-size: cover !important;
    background-position: center top !important;
    background-attachment: fixed !important;
    color: #29382d !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: rgba(255,248,244,.72) !important;
}

.block-container {
    max-width: 1460px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
}

h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #164b2c !important;
    letter-spacing: -.02em;
}

h1 { font-size: 2.5rem !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.3rem !important; }

p, li, label {
    line-height: 1.55;
}

.stMarkdown p,
.stMarkdown li,
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] p {
    color: #29382d;
}

a {
    color: #92163e;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: rgba(255,245,240,.96) !important;
    border-right: 1px solid #e9b9b3 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

[data-testid="stSidebar"] img {
    width: 100% !important;
    max-width: 315px !important;
    height: auto !important;
    object-fit: contain !important;
    margin: 0 auto !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 5px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,.75) !important;
    border: 1px solid #f0d5d0 !important;
    border-radius: 25px !important;
    padding: 7px 11px !important;
    margin: 2px 0 !important;
    transition: all .18s ease;
    box-shadow: 0 2px 5px rgba(112,12,48,.04);
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #ffe4e3 !important;
    transform: translateX(3px);
}

[data-testid="stSidebar"]
[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(
        100deg, #ae284e, #850e38
    ) !important;
    border-color: #92163e !important;
    box-shadow: 0 5px 12px rgba(112,12,48,.17);
}

[data-testid="stSidebar"]
[role="radiogroup"] label p {
    color: #671b34 !important;
    font-weight: 700 !important;
    font-size: .92rem !important;
}

[data-testid="stSidebar"]
[role="radiogroup"]
label:has(input:checked) p {
    color: white !important;
}

[data-testid="stSidebar"]
[role="radiogroup"] [data-testid="stMarkdownContainer"] {
    overflow: visible;
}

[data-testid="stSidebar"]
[role="radiogroup"] label > div:first-child {
    display: none !important;
}

.track-label {
    border: 1px solid #e8bcb6;
    background: rgba(255,255,255,.8);
    border-radius: 20px;
    padding: 10px;
    text-align: center;
    color: #70102f;
    font-size: .84rem;
    font-weight: 800;
    margin: 12px 0 18px;
}

.side-footer {
    border-top: 2px solid #bc5470;
    margin-top: 25px;
    padding-top: 20px;
    color: #3b4337;
    font-size: .85rem;
    line-height: 1.6;
}

/* BANNER */

.pom-banner {
    position: relative;
    min-height: 195px;
    border-radius: 30px;
    overflow: hidden;
    background:
        linear-gradient(110deg,#660b2c,#ad2548);
    background-image: url('BANNER_IMAGE');
    background-size: cover;
    background-position: center;
    box-shadow: 0 12px 28px rgba(112,12,48,.15);
    margin: 2px 0 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 30px 18px;
}

.pom-banner-fallback {
    background: linear-gradient(
        110deg,#6b0d30,#9e2447,#c44b65
    );
}

.pom-banner-text {
    background: rgba(88,5,36,.16);
    border-radius: 18px;
    padding: 7px 22px;
}

.pom-banner h1 {
    color: #fffaf0 !important;
    font-family: Georgia, serif !important;
    font-size: clamp(2.5rem,5vw,4.4rem) !important;
    font-weight: 800 !important;
    font-style: italic;
    margin: 0 !important;
    text-shadow: 0 2px 8px rgba(50,0,20,.3);
}

.pom-banner p {
    color: white !important;
    font-family: Georgia, serif;
    font-size: clamp(1.1rem,2.1vw,1.7rem);
    font-style: italic;
    margin: 0;
    text-shadow: 0 2px 6px rgba(50,0,20,.4);
}

/* CONTENT CARDS */

.pom-card {
    background: rgba(255,253,248,.90);
    border: 1px solid #f2d8d0;
    border-radius: 25px;
    padding: 23px 25px;
    margin-bottom: 13px;
    box-shadow:
        0 5px 15px rgba(111,53,60,.08),
        inset 0 0 0 2px rgba(255,255,255,.65);
    transition: transform .2s,box-shadow .2s;
}

.pom-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(111,53,60,.13);
}

.pom-card h3 {
    font-family: Georgia, serif !important;
    color: #8a123b !important;
    font-size: 1.25rem !important;
    margin: 0 0 8px;
}

.pom-card p {
    color: #283b2e !important;
    margin: 0;
    font-size: .96rem;
}

.pom-card-feature h3 {
    color: #164b2c !important;
    font-size: 1.7rem !important;
}

.pom-quote {
    background: linear-gradient(
        100deg,rgba(255,231,223,.92),
        rgba(255,250,244,.95)
    );
    border: 1px solid #efc8c0;
    border-radius: 23px;
    padding: 18px 26px;
    margin: 8px 0 20px;
    font-family: Georgia,serif;
    font-size: 1.3rem;
    color: #8a123b;
    font-weight: 600;
    font-style: italic;
}

.pom-section {
    background: rgba(255,249,244,.75);
    border: 1px solid #f0d8d0;
    border-radius: 25px;
    padding: 20px 24px;
    margin: 10px 0 20px;
}

.pom-section-title {
    font-family: Georgia,serif;
    color: #164b2c;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0 0 12px;
}

/* NOTES */

.pom-note {
    background: rgba(230,243,227,.94);
    border-left: 5px solid #4e805c;
    border-radius: 14px;
    padding: 15px 18px;
    margin: 15px 0;
    color: #214a31;
}

.pom-warning {
    background: rgba(255,238,218,.94);
    border-left: 5px solid #dcaa66;
    border-radius: 14px;
    padding: 15px 18px;
    margin: 15px 0;
    color: #6e4529;
}

/* TABLES */

.pom-table-wrap {
    background: #fffaf5;
    border: 1px solid #edcfc8;
    border-radius: 18px;
    overflow-x: auto;
    margin: 15px 0 24px;
    box-shadow: 0 6px 18px rgba(112,12,48,.06);
}

.pom-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .88rem;
    min-width: 580px;
}

.pom-table th {
    background: #8e2446 !important;
    color: #fffdf7 !important;
    text-align: left;
    padding: 13px;
    font-weight: 800;
}

.pom-table td {
    padding: 12px 13px;
    border-bottom: 1px solid #f0d9d0;
    color: #304133 !important;
    overflow-wrap: anywhere;
}

.pom-table tr:nth-child(odd) td {
    background: #fffdf8 !important;
}

.pom-table tr:nth-child(even) td {
    background: #ffefea !important;
}

.pom-table tr:hover td {
    background: #eaf3e5 !important;
}

/* NATIVE STREAMLIT DATAFRAMES */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: 16px;
    overflow: hidden;
}

/* METRICS */

[data-testid="stMetric"] {
    background: rgba(255,251,245,.95);
    border: 1px solid #edd2c9;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(112,12,48,.06);
}

[data-testid="stMetricLabel"] p {
    color: #36503b !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #91163d !important;
    font-weight: 800 !important;
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {
    background: #fff0e8 !important;
    border: 1px solid #efd0c8;
    border-radius: 17px;
    padding: 6px;
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #fffaf5 !important;
    border-radius: 12px !important;
    color: #782b46 !important;
    font-weight: 700;
    padding: 11px 15px !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #f7d7d9 !important;
}

.stTabs [aria-selected="true"] {
    background: #92163e !important;
}

.stTabs [aria-selected="true"] p {
    color: white !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: rgba(255,251,246,.65);
    border-radius: 16px;
    padding: 15px 10px;
    margin-top: 12px;
}

/* BUTTONS */

.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
    border-radius: 14px !important;
    font-weight: 700 !important;
    border-color: #c88999 !important;
    color: #7c1b3c !important;
    background: #fffaf7 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        100deg,#a8234d,#821137
    ) !important;
    color: white !important;
    border: none !important;
}

.stButton > button[kind="primary"] p {
    color: white !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stLinkButton > a:hover {
    border-color: #92163e !important;
    background: #ffe9e5 !important;
}

.stButton > button[kind="primary"]:hover {
    background: #70102f !important;
}

/* INPUTS AND SELECTORS */

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
textarea {
    background: #fffdf8 !important;
    color: #29382d !important;
    border-radius: 12px !important;
}

[data-baseweb="select"] *,
[data-baseweb="input"] *,
textarea {
    color: #29382d !important;
}

/* IMAGE FRAMES */

[data-testid="stImage"] img {
    border-radius: 20px;
}

.pom-footer {
    border-top: 1px solid #d9a4a6;
    text-align: center;
    padding: 20px 10px;
    margin-top: 35px;
    color: #67364a;
    font-size: .85rem;
}

@media (max-width: 750px) {
    .pom-banner {
        min-height: 145px;
        padding: 20px 10px;
    }

    .pom-banner h1 {
        font-size: 2.5rem !important;
    }

    .pom-card {
        padding: 17px;
    }

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
</style>
"""
    .replace(
        "BACKGROUND_IMAGE",
        BACKGROUND_URL or "none",
    )
    .replace(
        "BANNER_IMAGE",
        BANNER_URL or "none",
    ),
    unsafe_allow_html=True,
)


# ============================================================
# 4. REUSABLE UI COMPONENTS
# ============================================================

def safe(value):
    return html.escape(str(value), quote=True)


def banner():
    banner_path = ASSETS / "botanical_banner.jpg"

    if banner_path.exists():
        st.image(
            str(banner_path),
            use_container_width=True
        )
    else:
        st.warning(
            "Image introuvable : assets/botanical_banner.jpg"
        )




def card(title, description, feature=False):
    css = (
        "pom-card pom-card-feature"
        if feature
        else "pom-card"
    )

    st.markdown(
        f"""
        <div class="{css}">
            <h3>{safe(title)}</h3>
            <p>{safe(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(message, warning=False):
    css = (
        "pom-warning"
        if warning
        else "pom-note"
    )

    st.markdown(
        f'<div class="{css}">{safe(message)}</div>',
        unsafe_allow_html=True,
    )


def heading(title):
    st.markdown(
        f"""
        <div class="pom-section-title">
            {safe(title)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def quote(text):
    st.markdown(
        f'<div class="pom-quote">{safe(text)}</div>',
        unsafe_allow_html=True,
    )


def bright_table(frame):
    if frame is None or frame.empty:
        st.info("No data available.")
        return

    df = frame.copy()

    headers = "".join(
        f"<th>{safe(str(col).replace('_', ' ').title())}</th>"
        for col in df.columns
    )

    rows = []

    for _, record in df.iterrows():
        cells = []

        for value in record:
            if pd.isna(value):
                text = "—"
            elif isinstance(
                value, (float, np.floating)
            ):
                text = f"{value:.3f}"
            else:
                text = str(value)

            cells.append(
                f"<td>{safe(text)}</td>"
            )

        rows.append(
            "<tr>"
            + "".join(cells)
            + "</tr>"
        )

    st.markdown(
        '<div class="pom-table-wrap">'
        '<table class="pom-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table></div>",
        unsafe_allow_html=True,
    )


def show_figure(filename):
    path = FIGURES / filename

    if path.is_file():
        st.image(
            str(path),
            use_container_width=True,
        )
        return True

    return False


def display_photo():
    path = ASSETS / "pomegranate_photo.jpg"

    if path.is_file():
        st.image(
            str(path),
            use_container_width=True,
        )
    else:
        st.info(
            "Add pomegranate_photo.jpg to assets/."
        )


# ============================================================
# 5. DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path_string, modified):
    return pd.read_csv(path_string)


def load_data(paths):
    for path in paths:
        if not path.is_file():
            continue

        try:
            return read_csv(
                str(path),
                path.stat().st_mtime_ns,
            ).copy()

        except Exception as error:
            st.warning(
                f"Cannot read {path.name}: {error}"
            )

    return pd.DataFrame()


compounds = load_data([
    ML_DATA / "pomegranate_descriptors_verified.csv",
    DATA / "pomegranate_descriptors.csv",
])

predictions = load_data([
    ML_DATA / "predictions.csv",
    DATA / "predictions.csv",
])

bioactivity = load_data([
    ML_DATA / "ache_ml_ready.csv",
])

evidence = load_data([
    DATA / "evidence.csv",
])

targets = load_data([
    DATA / "targets.csv",
])

verification = load_data([
    ML_DATA / "pubchem_verification.csv",
])

scaffold_metrics = load_data([
    REPORTS / "scaffold_split_metrics.csv",
])


if (
    not compounds.empty
    and "compound_name" in compounds.columns
):
    compounds["family"] = (
        compounds["compound_name"]
        .map(COMPOUND_FAMILIES)
        .fillna("Other")
    )


# ============================================================
# 6. RDKIT AND MACHINE LEARNING
# ============================================================

@st.cache_resource(show_spinner=False)
def get_fp_generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True,
    )


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.is_file():
        return None

    # Load only the model trained and trusted by your team.
    return joblib.load(MODEL_PATH)


def get_molecule(smiles):
    if not isinstance(smiles, str):
        return None

    if not smiles.strip():
        return None

    try:
        return Chem.MolFromSmiles(
            smiles.strip()
        )
    except Exception:
        return None


def get_descriptors(mol):
    return {
        "MolWt": round(
            Descriptors.MolWt(mol), 2
        ),
        "LogP": round(
            Descriptors.MolLogP(mol), 2
        ),
        "TPSA": round(
            rdMolDescriptors.CalcTPSA(mol), 2
        ),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
    }


def molecule_image(mol, size=(500, 350)):
    return Draw.MolToImage(
        mol,
        size=size,
    )


def predict_activity(smiles):
    mol = get_molecule(smiles)

    if mol is None:
        raise ValueError(
            "Invalid SMILES. Check the molecular structure."
        )

    model = load_model()

    if model is None:
        raise FileNotFoundError(
            "Model not found: "
            "ml/models/random_forest_v1.joblib"
        )

    fp = get_fp_generator().GetFingerprint(mol)

    vector = np.zeros(
        2048,
        dtype=np.int8,
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        vector,
    )

    X = vector.reshape(1, -1)

    predicted_pic50 = float(
        model.predict(X)[0]
    )

    predicted_ic50 = (
        10 ** (9 - predicted_pic50)
    )

    result = {
        "pIC50": predicted_pic50,
        "IC50_nM": predicted_ic50,
        "molecule": mol,
        "descriptors": get_descriptors(mol),
    }

    if hasattr(model, "estimators_"):
        tree_values = np.array([
            float(tree.predict(X)[0])
            for tree in model.estimators_
        ])

        result["tree_std"] = float(
            np.std(tree_values)
        )

    return result


# ============================================================
# 7. SCIENTIFIC CHARTS
# ============================================================

def ranking_chart(frame):
    required = {
        "compound_name",
        "predicted_pIC50",
    }

    if (
        frame.empty
        or not required.issubset(frame.columns)
    ):
        st.info(
            "Prediction ranking is not available."
        )
        return

    data = frame.copy()

    data["predicted_pIC50"] = pd.to_numeric(
        data["predicted_pIC50"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["predicted_pIC50"]
    )

    if data.empty:
        st.info("No valid predictions.")
        return

    data = data.sort_values(
        "predicted_pIC50",
        ascending=False,
    )

    order = data["compound_name"].tolist()

    bars = alt.Chart(data).mark_bar(
        color="#ae3e60",
        size=25,
        cornerRadiusEnd=7,
    ).encode(
        x=alt.X(
            "predicted_pIC50:Q",
            title="Predicted pIC50",
            scale=alt.Scale(zero=True),
            axis=alt.Axis(
                labelColor="#164b2c",
                titleColor="#164b2c",
                gridColor="#ead9d1",
            ),
        ),
        y=alt.Y(
            "compound_name:N",
            sort=order,
            title=None,
            axis=alt.Axis(
                labelColor="#164b2c",
                labelFontSize=13,
                labelLimit=230,
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "compound_name:N",
                title="Compound",
            ),
            alt.Tooltip(
                "predicted_pIC50:Q",
                title="Predicted pIC50",
                format=".3f",
            ),
        ],
    )

    labels = alt.Chart(data).mark_text(
        align="left",
        baseline="middle",
        dx=7,
        fontSize=12,
        fontWeight="bold",
        color="#7b1538",
    ).encode(
        x="predicted_pIC50:Q",
        y=alt.Y(
            "compound_name:N",
            sort=order,
        ),
        text=alt.Text(
            "predicted_pIC50:Q",
            format=".2f",
        ),
    )

    chart = (
        bars + labels
    ).properties(
        height=max(
            390,
            len(data) * 42,
        ),
        background="#fffaf5",
        padding={
            "left": 10,
            "right": 50,
            "top": 10,
            "bottom": 10,
        },
    ).configure_view(
        stroke=None,
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    st.caption(
        "Higher predicted pIC50 corresponds "
        "to lower predicted IC50. "
        "These are computational estimates."
    )


def model_comparison():
    metrics = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost",
        ],
        "MAE": [
            0.5400,
            0.6851,
        ],
        "R2": [
            0.7350,
            0.6291,
        ],
    })

    bright_table(metrics)

    left, right = st.columns(2)

    with left:
        st.markdown("### Mean Absolute Error")

        if not show_figure(
            "model_comparison_mae.png"
        ):
            st.bar_chart(
                metrics.set_index("Model")[["MAE"]]
            )

    with right:
        st.markdown(
            "### Coefficient of Determination"
        )

        if not show_figure(
            "model_comparison_r2.png"
        ):
            st.bar_chart(
                metrics.set_index("Model")[["R2"]]
            )


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    if (ASSETS / "pomenova_logo.png").is_file():
        st.image(
            str(ASSETS / "pomenova_logo.png"),
            use_container_width=True,
        )
    else:
        st.markdown("# 🍎 POMENOVA")

    st.markdown(
        """
        <div class="track-label">
            ⚗️ EthnoHack 2026 · Track 2: Natural products and Drug Discovery
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "✨ 01  Discover POMENOVA",
        "🍎 02  Compound Library",
        "🔬 03  Molecular Explorer",
        "🧠 04  AI Activity Predictor",
        "📈 05  Model Validation",
        "🎯 06  Biological Target",
        "📚 07  Scientific Evidence",
        "🧪 08  Research Workflow",
        "🌱 09  Impact & Next Steps",
    ]

    page = st.radio(
        "Explore",
        pages,
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="side-footer">
            <strong>
                AI-assisted early-stage drug discovery
            </strong>
            <p>
                Natural Products × Machine Learning
            </p>
            <p>
                Scientific prototype · Not for clinical use
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 1 — DISCOVER POMENOVA
# ============================================================

if page == pages[0]:
    banner()

    left, right = st.columns(
        [1.65, 1],
        gap="large",
    )

    with left:
        card(
            "🍎 From Pomegranate to Possibility",
            "What if a familiar fruit could inspire "
            "new therapeutic hypotheses? "
            "POMENOVA combines natural-product "
            "research, cheminformatics and machine "
            "learning to explore pomegranate "
            "phytochemicals and human "
            "acetylcholinesterase (AChE) activity.",
            feature=True,
        )

        card(
            "🌿 How does POMENOVA Work?",
            "A transparent and reproducible "
            "workflow connecting botanical "
            "research, PubChem molecular structures, "
            "ChEMBL bioactivity records, RDKit "
            "fingerprints and AI-driven "
            "candidate prioritization.",
            feature=True,
        )

        quote(
            "A new perspective on discovering "
            "potential AChE inhibitors from "
            "natural products."
        )

    with right:
        display_photo()

        st.caption(
            "Punica granatum L. — the botanical "
            "inspiration for POMENOVA."
        )

    heading("Our Research at a Glance")

    a, b = st.columns(2)

    with a:
        card(
            "🌿 Natural Compounds",
            "A curated library of ten selected "
            "pomegranate-associated phytochemicals."
        )

        card(
            "🔬 Molecular Science",
            "PubChem structures, RDKit molecular "
            "descriptors and Morgan fingerprints."
        )

    with b:
        card(
            "🧠 AI-Powered Prediction",
            "Random Forest and XGBoost models "
            "evaluated using human AChE "
            "bioactivity data."
        )

        card(
            "🛡️ Scientific Validation",
            "Model comparison, scaffold-based "
            "evaluation and transparent limitations."
        )

    heading("Our Research in Numbers")

    a, b, c, d = st.columns(4)

    a.metric(
        "Selected compounds",
        len(compounds)
        if not compounds.empty
        else "—",
    )

    b.metric(
        "Curated AChE molecules",
        (
            bioactivity["canonical_smiles"].nunique()
            if (
                not bioactivity.empty
                and "canonical_smiles"
                in bioactivity.columns
            )
            else (
                len(bioactivity)
                if not bioactivity.empty
                else "—"
            )
        ),
    )

    c.metric(
        "AI predictions",
        len(predictions)
        if not predictions.empty
        else "—",
    )

    d.metric(
        "Biological target",
        "AChE",
    )

    note(
        "POMENOVA generates computational "
        "research hypotheses. It does not "
        "demonstrate that pomegranate treats "
        "or prevents Alzheimer's disease.",
        warning=True,
    )

    heading("Explore Our Predicted Candidates")

    ranking_chart(predictions)

    st.link_button(
        "Explore the research on GitHub",
        GITHUB,
    )


# ============================================================
# PAGE 2 — COMPOUND LIBRARY
# ============================================================

elif page == pages[1]:
    banner()

    heading("🍎 The Compound Garden")

    st.write(
        "Explore selected phytochemicals "
        "associated with pomegranate, their "
        "chemical families and molecular properties."
    )

    if (
        compounds.empty
        or "compound_name" not in compounds.columns
    ):
        st.error(
            "Compound dataset is missing or invalid."
        )

    else:
        left, right = st.columns(2)

        families = sorted(
            compounds["family"]
            .dropna()
            .unique()
            .tolist()
        )

        with left:
            family = st.selectbox(
                "Chemical family",
                ["All families"] + families,
            )

        with right:
            search = st.text_input(
                "Search compounds",
                placeholder="Quercetin, ellagic acid...",
            )

        filtered = compounds.copy()

        if family != "All families":
            filtered = filtered[
                filtered["family"] == family
            ]

        if search.strip():
            filtered = filtered[
                filtered["compound_name"]
                .astype(str)
                .str.contains(
                    search.strip(),
                    case=False,
                    regex=False,
                )
            ]

        heading(
            f"🌿 {len(filtered)} Compounds"
        )

        columns = [
            "compound_name",
            "family",
            "molecular_formula",
            "MolWt",
            "LogP",
            "TPSA",
            "HBD",
            "HBA",
        ]

        available = [
            col for col in columns
            if col in filtered.columns
        ]

        bright_table(
            filtered[available]
        )

        heading("Discover a Molecule")

        selected = st.selectbox(
            "Select a compound",
            compounds["compound_name"].tolist(),
        )

        row = compounds[
            compounds["compound_name"] == selected
        ].iloc[0]

        smiles = str(
            row.get("smiles", "")
        )

        mol = get_molecule(smiles)

        left, right = st.columns(2)

        with left:
            if mol is not None:
                st.image(
                    molecule_image(mol),
                    caption=selected,
                )
            else:
                st.warning(
                    "Molecular structure unavailable."
                )

        with right:
            card(
                selected,
                "Chemical family: "
                + str(
                    row.get("family", "Unknown")
                ),
            )

            cid = str(
                row.get("pubchem_cid", "")
            ).strip()

            if cid.endswith(".0"):
                cid = cid[:-2]

            if (
                cid
                and cid.lower() != "nan"
            ):
                st.link_button(
                    "Open in PubChem",
                    "https://pubchem.ncbi.nlm.nih.gov/"
                    f"compound/{cid}",
                )

            st.markdown("**Molecular SMILES**")

            st.code(
                smiles,
                language="text",
            )

        with st.expander(
            "PubChem verification records"
        ):
            if verification.empty:
                st.info(
                    "Verification records unavailable."
                )
            else:
                bright_table(verification)

        note(
            "Molecular identity and physicochemical "
            "descriptors do not prove therapeutic "
            "activity, safety or brain penetration.",
            warning=True,
        )


# ============================================================
# PAGE 3 — MOLECULAR EXPLORER
# ============================================================

elif page == pages[2]:
    banner()

    heading("🔬 Molecular Explorer")

    st.write(
        "Explore chemical structures, "
        "molecular descriptors and predicted "
        "AChE activity."
    )

    if (
        compounds.empty
        or "compound_name" not in compounds.columns
    ):
        st.error(
            "Compound library unavailable."
        )

    else:
        selected = st.selectbox(
            "Choose a phytochemical",
            compounds["compound_name"].tolist(),
        )

        row = compounds[
            compounds["compound_name"] == selected
        ].iloc[0]

        smiles = str(
            row.get("smiles", "")
        )

        mol = get_molecule(smiles)

        if mol is None:
            st.error(
                "Invalid molecular structure."
            )

        else:
            left, right = st.columns(2)

            with left:
                st.image(
                    molecule_image(
                        mol,
                        (500, 360),
                    ),
                    caption=selected,
                )

            with right:
                card(
                    f"🍃 {selected}",
                    "Molecular structure and "
                    "physicochemical profile.",
                )

                descriptors = get_descriptors(mol)

                bright_table(
                    pd.DataFrame([
                        {
                            "Descriptor": k,
                            "Value": v,
                        }
                        for k, v
                        in descriptors.items()
                    ])
                )

            heading("Predicted AChE Activity")

            if (
                not predictions.empty
                and "compound_name"
                in predictions.columns
            ):
                matching = predictions[
                    predictions["compound_name"]
                    .astype(str)
                    .str.casefold()
                    == selected.casefold()
                ]

                if not matching.empty:
                    prediction = matching.iloc[0]

                    p = pd.to_numeric(
                        prediction.get(
                            "predicted_pIC50"
                        ),
                        errors="coerce",
                    )

                    ic50 = pd.to_numeric(
                        prediction.get(
                            "predicted_IC50_nM"
                        ),
                        errors="coerce",
                    )

                    a, b = st.columns(2)

                    if pd.notna(p):
                        a.metric(
                            "Predicted pIC50",
                            f"{p:.3f}",
                        )

                    if pd.notna(ic50):
                        b.metric(
                            "Predicted IC50",
                            f"{ic50:.2f} nM",
                        )

                    if (
                        "max_tanimoto_similarity"
                        in prediction.index
                    ):
                        st.write(
                            "**Maximum training similarity:**",
                            prediction[
                                "max_tanimoto_similarity"
                            ],
                        )

            note(
                "Predicted AChE activity does "
                "not establish experimental "
                "inhibition, drug safety or "
                "clinical efficacy.",
                warning=True,
            )


# ============================================================
# PAGE 4 — AI ACTIVITY PREDICTOR
# ============================================================

elif page == pages[3]:
    banner()

    heading("🧠 Meet the AI Lab")

    st.write(
        "Predict human AChE activity from "
        "a molecular SMILES representation "
        "using the trained Random Forest model."
    )

    note(
        "SMILES → RDKit → Morgan fingerprint "
        "(radius 2, 2048 bits, chirality-aware) "
        "→ Random Forest → predicted pIC50."
    )

    mode = st.radio(
        "Choose input method",
        [
            "Select a library compound",
            "Enter custom SMILES",
        ],
        horizontal=True,
    )

    if mode == "Select a library compound":
        if (
            compounds.empty
            or "compound_name"
            not in compounds.columns
            or "smiles"
            not in compounds.columns
        ):
            smiles = ""
            st.error(
                "Compound library unavailable."
            )

        else:
            selected = st.selectbox(
                "Choose a molecule",
                compounds["compound_name"].tolist(),
            )

            smiles = str(
                compounds.loc[
                    compounds["compound_name"]
                    == selected,
                    "smiles",
                ].iloc[0]
            )

            st.code(
                smiles,
                language="text",
            )

    else:
        smiles = st.text_area(
            "Enter SMILES",
            placeholder=(
                "Enter a valid molecular SMILES"
            ),
            height=110,
        )

    if st.button(
        "🔬 Predict AChE Activity",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Analyzing molecular structure..."
            ):
                result = predict_activity(smiles)

            st.success(
                "Prediction completed successfully."
            )

            a, b = st.columns(2)

            a.metric(
                "Predicted pIC50",
                f"{result['pIC50']:.3f}",
            )

            b.metric(
                "Predicted IC50",
                f"{result['IC50_nM']:.2f} nM",
            )

            left, right = st.columns(2)

            with left:
                st.image(
                    molecule_image(
                        result["molecule"]
                    ),
                    caption="Predicted molecule",
                )

            with right:
                heading("Molecular Descriptors")

                bright_table(
                    pd.DataFrame([
                        result["descriptors"]
                    ])
                )

            if "tree_std" in result:
                st.write(
                    "**Random Forest tree dispersion:**",
                    f"{result['tree_std']:.3f}",
                )

                st.caption(
                    "Tree dispersion is not a "
                    "calibrated confidence interval."
                )

            note(
                "Computational prediction only. "
                "Experimental validation is required.",
                warning=True,
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )

    heading("Understanding the Limitations")

    a, b, c = st.columns(3)

    with a:
        card(
            "🧪 Experimental Validation",
            "Laboratory AChE inhibition assays "
            "are needed to test predictions.",
        )

    with b:
        card(
            "🧠 Brain Exposure",
            "The model does not directly "
            "predict blood-brain barrier "
            "penetration.",
        )

    with c:
        card(
            "📊 Prediction Uncertainty",
            "Predictions may be less reliable "
            "for unfamiliar molecular structures.",
        )


# ============================================================
# PAGE 5 — MODEL VALIDATION
# ============================================================

elif page == pages[4]:
    banner()

    heading("📈 Behind the Predictions")

    st.write(
        "Explore model performance and "
        "scientific limitations."
    )

    tabs = st.tabs([
        "🌸 Model Comparison",
        "🌿 Scaffold Validation",
        "🍎 Candidate Ranking",
    ])

    with tabs[0]:
        heading("Random Forest vs XGBoost")

        st.write(
            "MAE measures average absolute "
            "prediction error in pIC50 units. "
            "R² evaluates predictive fit."
        )

        model_comparison()

        note(
            "Reported initial evaluation: "
            "Random Forest MAE = 0.5400 "
            "and R² = 0.7350."
        )

        heading("Actual vs Predicted pIC50")

        if not show_figure(
            "rf_actual_vs_predicted.png"
        ):
            st.info(
                "Validation figure unavailable."
            )

    with tabs[1]:
        heading("Scaffold-Based Evaluation")

        st.write(
            "Scaffold splitting evaluates "
            "generalization to molecular "
            "frameworks separated between "
            "training and test sets."
        )

        metrics = pd.DataFrame({
            "Model": [
                "Random Forest",
                "XGBoost",
            ],
            "MAE": [
                0.5935,
                0.7013,
            ],
            "R2": [
                0.6777,
                0.5924,
            ],
        })

        bright_table(metrics)

        if not scaffold_metrics.empty:
            with st.expander(
                "Original scaffold metrics"
            ):
                bright_table(
                    scaffold_metrics
                )

        note(
            "Scaffold-based validation is a "
            "more demanding generalization test. "
            "It does not guarantee accurate "
            "predictions for every candidate.",
            warning=True,
        )

    with tabs[2]:
        heading("Candidate Ranking")

        ranking_chart(predictions)

        if not predictions.empty:
            cols = [
                "compound_name",
                "predicted_pIC50",
                "predicted_IC50_nM",
                "max_tanimoto_similarity",
                "uncertainty_std",
            ]

            available = [
                col for col in cols
                if col in predictions.columns
            ]

            bright_table(
                predictions[available]
            )

        note(
            "Predicted potency alone is not "
            "sufficient to select a drug "
            "candidate. Selectivity, safety "
            "and exposure require evaluation.",
            warning=True,
        )


# ============================================================
# PAGE 6 — BIOLOGICAL TARGET
# ============================================================

elif page == pages[5]:
    banner()

    heading("🎯 Understanding Our Target")

    st.write(
        "Human acetylcholinesterase (AChE): "
        "its biological role and relevance "
        "to Alzheimer's disease research."
    )

    a, b, c = st.columns(3)

    a.metric(
        "ChEMBL Target",
        "CHEMBL220",
    )

    b.metric(
        "Gene",
        "ACHE",
    )

    c.metric(
        "UniProt",
        "P22303",
    )

    heading("Why Acetylcholinesterase?")

    left, right = st.columns(2)

    with left:
        card(
            "🧠 Biological Function",
            "Acetylcholinesterase hydrolyzes "
            "acetylcholine and terminates "
            "cholinergic signaling.",
        )

    with right:
        card(
            "🔬 Therapeutic Relevance",
            "AChE inhibition is an established "
            "symptomatic treatment strategy "
            "in Alzheimer's disease.",
        )

    note(
        "AChE inhibition does not automatically "
        "establish neuroprotection, disease "
        "modification or prevention.",
        warning=True,
    )

    a, b = st.columns(2)

    with a:
        st.link_button(
            "Explore CHEMBL220",
            "https://www.ebi.ac.uk/chembl/"
            "target_report_card/CHEMBL220/",
        )

    with b:
        st.link_button(
            "Explore UniProt P22303",
            "https://www.uniprot.org/"
            "uniprotkb/P22303/entry",
        )

    if not targets.empty:
        heading("Curated Target Information")
        bright_table(targets)


# ============================================================
# PAGE 7 — SCIENTIFIC EVIDENCE
# ============================================================

elif page == pages[6]:
    banner()

    heading("📚 The Science Behind the Story")

    st.write(
        "Explore scientific literature "
        "and distinguish experimental "
        "observations from computational "
        "predictions."
    )

    if evidence.empty:
        st.warning(
            "Evidence file not found or empty: "
            "data/evidence.csv"
        )

    else:
        st.metric(
            "Literature records",
            len(evidence),
        )

        search = st.text_input(
            "Search the literature",
            placeholder="Compound, study, target...",
        )

        filtered = evidence.copy()

        if search.strip():
            searchable = (
                filtered.astype(str)
                .agg(" ".join, axis=1)
            )

            filtered = filtered[
                searchable.str.contains(
                    search.strip(),
                    case=False,
                    regex=False,
                )
            ]

        heading("Scientific Evidence Table")

        bright_table(filtered)

        heading("Explore Individual Studies")

        for idx, row in filtered.iterrows():
            title = str(
                row.get(
                    "compound_name",
                    f"Study {idx + 1}",
                )
            )

            with st.expander(title):
                for col in filtered.columns:
                    value = row[col]

                    if pd.notna(value):
                        st.write(
                            f"**{col.replace('_', ' ').title()}:**",
                            value,
                        )

                doi = str(
                    row.get("doi", "")
                ).strip()

                if (
                    doi
                    and doi.lower() != "nan"
                ):
                    doi = (
                        doi.replace(
                            "https://doi.org/", ""
                        )
                        .replace(
                            "http://doi.org/", ""
                        )
                    )

                    st.link_button(
                        "Open publication",
                        "https://doi.org/" + doi,
                    )

        st.download_button(
            "Download evidence table",
            evidence.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="pomenova_evidence.csv",
            mime="text/csv",
        )

    heading("Understanding Evidence Levels")

    a, b, c = st.columns(3)

    with a:
        card(
            "💻 Computational",
            "Predictions generate testable "
            "hypotheses but do not prove activity.",
        )

    with b:
        card(
            "🧫 Experimental",
            "Biochemical and cellular studies "
            "can investigate biological effects.",
        )

    with c:
        card(
            "🏥 Translational",
            "Pharmacokinetics, safety and "
            "clinical studies are necessary "
            "for therapeutic conclusions.",
        )

    note(
        "Results obtained with pomegranate "
        "extracts cannot automatically be "
        "attributed to individual compounds. "
        "Study conditions and publication "
        "identifiers must be verified.",
        warning=True,
    )


# ============================================================
# PAGE 8 — RESEARCH WORKFLOW
# ============================================================

elif page == pages[7]:
    banner()

    heading("🧪 From Nature to Intelligence")

    st.write(
        "Follow the scientific journey "
        "from botanical research to "
        "AI-assisted candidate prioritization."
    )

    workflow = [
        (
            "01 🌿 Botanical Research",
            "Identify phytochemicals associated "
            "with Punica granatum L.",
        ),
        (
            "02 🍎 PubChem Verification",
            "Verify molecular identities, "
            "SMILES and stereochemistry.",
        ),
        (
            "03 🧬 ChEMBL Extraction",
            "Retrieve human AChE bioactivity "
            "records for CHEMBL220.",
        ),
        (
            "04 🧹 Data Curation",
            "Filter measurements, handle "
            "duplicates and aggregate "
            "repeated activity records.",
        ),
        (
            "05 🔬 Molecular Fingerprints",
            "Generate RDKit Morgan fingerprints "
            "with radius 2 and 2048 bits.",
        ),
        (
            "06 🧠 Machine Learning",
            "Train and compare Random Forest "
            "and XGBoost regressors.",
        ),
        (
            "07 📈 Validation",
            "Evaluate random-split and "
            "scaffold-split performance.",
        ),
        (
            "08 🍃 Candidate Screening",
            "Estimate pIC50 values and "
            "assess chemical similarity.",
        ),
        (
            "09 🧫 Future Experiments",
            "Propose biochemical assays, "
            "selectivity and ADMET studies.",
        ),
    ]

    for i in range(
        0,
        len(workflow),
        3,
    ):
        columns = st.columns(3)

        for col, item in zip(
            columns,
            workflow[i:i + 3],
        ):
            with col:
                card(
                    item[0],
                    item[1],
                )

    heading("Technologies and Resources")

    a, b = st.columns(2)

    with a:
        card(
            "📚 Scientific Databases",
            "PubChem, ChEMBL and UniProt.",
        )

    with b:
        card(
            "💻 Computational Tools",
            "Python, RDKit, Pandas, "
            "scikit-learn, XGBoost and Streamlit.",
        )

    st.link_button(
        "Explore our ML workflow",
        GITHUB + "/tree/main/ml",
    )


# ============================================================
# PAGE 9 — IMPACT AND NEXT STEPS
# ============================================================

elif page == pages[8]:
    banner()

    heading("🌱 Beyond the Prototype")

    st.write(
        "What POMENOVA contributes today "
        "and what remains to be investigated."
    )

    heading("What We Have Built")

    a, b, c = st.columns(3)

    with a:
        card(
            "🌿 Molecular Library",
            "A curated collection of ten "
            "selected phytochemicals.",
        )

    with b:
        card(
            "🧠 AI Prediction",
            "A Random Forest model trained "
            "on curated human AChE data.",
        )

    with c:
        card(
            "💻 Interactive Platform",
            "An accessible interface for "
            "molecular exploration and "
            "scientific interpretation.",
        )

    heading("Scientific Limitations")

    note(
        "The ten candidate predictions "
        "have not been experimentally "
        "validated by our team. "
        "We have not established brain "
        "exposure, safety, in-vivo efficacy "
        "or clinical benefit.",
        warning=True,
    )

    heading("Our Future Research Roadmap")

    steps = [
        (
            "1 · Verify Compounds",
            "Confirm molecular identity, "
            "purity and availability.",
        ),
        (
            "2 · Measure AChE Inhibition",
            "Perform concentration-response "
            "enzyme inhibition assays.",
        ),
        (
            "3 · Evaluate Selectivity",
            "Investigate BChE activity and "
            "possible assay interference.",
        ),
        (
            "4 · Investigate ADMET",
            "Assess solubility, metabolism, "
            "toxicity and CNS exposure.",
        ),
        (
            "5 · Improve the Model",
            "Integrate experimental data "
            "and strengthen validation.",
        ),
    ]

    for title, description in steps:
        card(
            title,
            description,
        )

    heading("Our Vision")

    quote(
        "Connecting the intelligence of "
        "nature with the possibilities "
        "of artificial intelligence."
    )

    st.write(
        "POMENOVA combines natural-product "
        "research and machine learning "
        "through a transparent, reproducible "
        "early-stage drug discovery workflow."
    )

    st.write(
        "Our objective is to generate "
        "testable scientific hypotheses "
        "while communicating uncertainty "
        "and limitations responsibly."
    )

    st.link_button(
        "Explore POMENOVA on GitHub",
        GITHUB,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="pom-footer">
        <strong>POMENOVA</strong>
        · Nature Meets Intelligence
        <p>
            🍎 <em>Punica granatum L.</em>
            × Human AChE
            × Responsible Artificial Intelligence
        </p>
        EthnoHack 2026
        · Track: Drug Discovery
        · Not for clinical use
    </div>
    """,
    unsafe_allow_html=True,
)



