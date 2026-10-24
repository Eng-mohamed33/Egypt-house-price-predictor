"""Egypt House Price Predictor: local Gradio app.

Run the notebook first (it creates price_model.joblib), then:
    python app.py

If egypt_real_estate_listings.csv (or the Kaggle zip) is in the same folder, the app also:
  - orders compounds by number of listings (most popular first), and
  - compares your input with similar listings and warns when they are rare.
Without the CSV the app still works, just without those two extras.
"""
import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
bundle = joblib.load(HERE / "price_model.joblib")

TYPES = bundle["types"]
AREAS = bundle["areas"]
COMPOUNDS_BY_AREA = bundle["compounds_by_area"]
DEFAULT_AREA = "New Cairo City" if "New Cairo City" in AREAS else AREAS[0]

MIN_LISTINGS_IN_DROPDOWN = 3   # hide compounds with fewer listings (you can still type them)
SIMILAR_SIZE_TOLERANCE = 0.25  # "similar" = same type and compound, size within +/-25%
MIN_SIMILAR_FOR_CONFIDENCE = 8  # fewer similar listings than this -> show a warning


# ---------------------------------------------------------------- listing statistics (optional)
def load_stats():
    """Re-create the cleaned listings table from the CSV. Returns None if no data file is found."""
    candidates = [HERE / "egypt_real_estate_listings.csv", *sorted(HERE.glob("*.zip"))]
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path, usecols=["price", "description", "location", "type", "size"])
        except Exception:
            continue
        df = raw.drop_duplicates(subset=["price", "description", "location"]).copy()
        parts = df["location"].fillna("").str.split(r"\s*,\s*", regex=True)
        df["compound"] = parts.str[0]
        df["area"] = parts.str[-2].where(parts.str.len() > 1, parts.str[-1])
        df["size_sqm"] = pd.to_numeric(
            df["size"].str.extract(r"/\s*([\d,]+)\s*sqm")[0].str.replace(",", ""), errors="coerce")
        df["price"] = pd.to_numeric(df["price"].str.replace(",", "", regex=False), errors="coerce")
        df = df.dropna(subset=["price", "size_sqm", "type", "area"])
        df = df[df["size_sqm"].between(20, 2000)]
        return df[["type", "area", "compound", "size_sqm", "price"]]
    return None


STATS = load_stats()
LISTING_COUNTS = (
    STATS.groupby(["area", "compound"]).size().to_dict() if STATS is not None else {}
)


def compound_choices(area):
    """(label, value) pairs, most popular compounds first."""
    names = COMPOUNDS_BY_AREA.get(area, [])
    if not LISTING_COUNTS:
        return [(n, n) for n in names]
    ranked = sorted(names, key=lambda n: (-LISTING_COUNTS.get((area, n), 0), n))
    common = [n for n in ranked if LISTING_COUNTS.get((area, n), 0) >= MIN_LISTINGS_IN_DROPDOWN]
    shown = common or ranked
    return [(f"{n} ({LISTING_COUNTS.get((area, n), 0)})", n) for n in shown]


def similar_listings(type_, area, compound, size_sqm):
    """How many listings share this type and compound, and how many are close in size."""
    if STATS is None:
        return None
    same = STATS[(STATS["compound"] == compound) & (STATS["area"] == area) & (STATS["type"] == type_)]
    lo, hi = size_sqm * (1 - SIMILAR_SIZE_TOLERANCE), size_sqm * (1 + SIMILAR_SIZE_TOLERANCE)
    near = same[same["size_sqm"].between(lo, hi)]
    median_price = near["price"].median() if len(near) >= 3 else None
    return {"same_type": len(same), "similar": len(near), "median_price": median_price}


# ---------------------------------------------------------------- model
def predict_range(model, X, lo=25, hi=75):
    """Ask each tree separately, then take percentiles to get a range."""
    tree_preds = np.column_stack([t.predict(X.values) for t in model.estimators_])
    return (
        np.expm1(np.percentile(tree_preds, lo, axis=1)),
        np.expm1(np.median(tree_preds, axis=1)),
        np.expm1(np.percentile(tree_preds, hi, axis=1)),
    )


def build_features(type_, area, compound, size_sqm, beds, baths, has_maid, installments, description):
    d = (description or "").lower()
    row = {
        "log_size": np.log(size_sqm),
        "size_sqm": size_sqm,
        "beds": beds,
        "baths": baths,
        "has_maid": int(has_maid),
        "installments": int(installments),
        "compound_te": bundle["enc_compound"].get(compound, bundle["g_compound"]),
        "area_te": bundle["enc_area"].get(area, bundle["g_area"]),
        # an empty box means "not provided", so use the typical length instead of 0
        "desc_len": np.log1p(len(description)) if description and description.strip()
                    else float(bundle["medians"]["desc_len"]),
    }
    for name, pattern in bundle["keywords"].items():
        row[name] = int(bool(re.search(pattern, d)))
    X = pd.DataFrame([row])[bundle["num_feats"]]
    X = pd.concat([X, pd.get_dummies(pd.Series([type_]))], axis=1)
    return X.reindex(columns=bundle["columns"], fill_value=0).fillna(bundle["medians"])


def round_100k(x):
    return round(x / 100_000) * 100_000


def million(x):
    return f"{round_100k(x) / 1e6:,.1f}M"


def reference_text(type_, area, compound, size_sqm):
    info = similar_listings(type_, area, compound, size_sqm)
    if info is None:
        return ""
    pct = int(SIMILAR_SIZE_TOLERANCE * 100)
    if info["same_type"] == 0:
        return (
            f"\n\n> **Low confidence:** the data has no {type_} listings in this compound, "
            "so the estimate relies mostly on the area and general patterns."
        )
    n = info["similar"]
    text = f"\n\n**Similar listings:** {n} ({type_}, same compound, size within ±{pct}%)"
    if info["median_price"] is not None:
        text += f", median asking price {million(info['median_price'])} EGP"
    text += "."
    if n < MIN_SIMILAR_FOR_CONFIDENCE:
        count = "are no comparable listings" if n == 0 else (
            "is only 1 comparable listing" if n == 1 else f"are only {n} comparable listings")
        text += (
            f"\n\n> **Low confidence:** there {count} for this combination, so the model is "
            "less reliable here and the range above can be too narrow."
        )
    text += "\n\nUse the similar listings as a cross-check: if they differ a lot from the estimate, trust them more."
    return text


def predict(type_, area, compound, size_sqm, beds, baths, has_maid, installments, description):
    if not size_sqm or size_sqm < 20 or size_sqm > 2000:
        return "Please enter a size between 20 and 2000 sqm."
    if not compound:
        return "Please choose or type a compound."

    X = build_features(type_, area, compound, size_sqm, beds, baths, has_maid, installments, description)
    low, mid, high = (v[0] for v in predict_range(bundle["model"], X))

    note = ""
    if compound not in bundle["enc_compound"].index:
        note = (
            "\n\n> This compound is not in the training data, so the estimate relies on the "
            "area and the unit's features only. Treat it with extra caution."
        )
    return (
        f"## Estimate: about {million(mid)} EGP\n"
        f"**Likely range:** {million(low)} to {million(high)} EGP\n\n"
        f"The range comes from the spread of the model's 300 trees (25th to 75th percentile). "
        f"It should contain the true asking price roughly half of the time. "
        f"This is an estimate based on listing prices, not an official valuation."
        f"{reference_text(type_, area, compound, size_sqm)}"
        f"{note}"
    )


# ---------------------------------------------------------------- UI
def update_compounds(area):
    choices = compound_choices(area)
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


with gr.Blocks(title="Egypt House Price Predictor") as demo:
    gr.Markdown(
        "# Egypt House Price Predictor\n"
        "A Random Forest model trained on ~19,000 PropertyFinder listings. "
        "Pick the property details and get an estimated asking price with a likely range."
    )
    with gr.Row():
        with gr.Column():
            type_in = gr.Dropdown(TYPES, value="Apartment", label="Property type")
            area_in = gr.Dropdown(AREAS, value=DEFAULT_AREA, label="Area")
            default_choices = compound_choices(DEFAULT_AREA)
            default_values = [v for _, v in default_choices]
            compound_in = gr.Dropdown(
                default_choices,
                value="Mivida" if "Mivida" in default_values else (default_values[0] if default_values else None),
                label="Compound (number of listings in brackets)",
                allow_custom_value=True,
            )
            size_in = gr.Number(value=150, label="Size (sqm)", precision=0)
            with gr.Row():
                beds_in = gr.Slider(0, 8, value=3, step=1, label="Bedrooms (0 = studio)")
                baths_in = gr.Slider(1, 8, value=2, step=1, label="Bathrooms")
            with gr.Row():
                maid_in = gr.Checkbox(label="Maid's room")
                inst_in = gr.Checkbox(label="Installments", value=True)
            desc_in = gr.Textbox(
                label="Listing description (optional)",
                placeholder="e.g. Fully finished, garden view, 8 years installments",
                lines=3,
            )
            btn = gr.Button("Predict price", variant="primary")
        with gr.Column():
            out = gr.Markdown("Fill in the details and press **Predict price**.")

    area_in.change(update_compounds, area_in, compound_in)
    inputs = [type_in, area_in, compound_in, size_in, beds_in, baths_in, maid_in, inst_in, desc_in]
    btn.click(predict, inputs, out)

if __name__ == "__main__":
    demo.launch()
