"""
Airbnb Pricing & Upgrade-ROI Dashboard
MSCI 436 Project

Run with: streamlit run app.py
Requires: pricing_model.joblib in the same folder (already trained --
see airbnb_idss_model.py to retrain on fresh data).
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Austin Airbnb Upgrade Advisor", layout="centered")

# ---------------------------------------------------------------------------
# LOAD THE REAL TRAINED MODEL
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("pricing_model.joblib")

model = load_model()

UPGRADE_FLAGS = {
    "Add hot tub / pool":   "hot_tub_pool",
    "Add luxury amenities": "luxury_amenities",
    "Add self check-in":    "self_check_in",
    "Make pet-friendly":    "pet_friendly",
    "Add parking":          "parking",
}
UPGRADE_COST = {
    "Add hot tub / pool":   9000,
    "Add luxury amenities": 3000,
    "Add self check-in":     250,
    "Make pet-friendly":     150,
    "Add parking":            800,
}

NEIGHBOURHOODS = ["78704", "78702", "78701", "78745", "78741", "78703",
                   "78705", "78744", "78723", "78751", "78758", "78734"]
PROPERTY_TYPES = ["Entire home", "Entire rental unit", "Private room in home",
                   "Entire condo", "Entire guesthouse", "Room in hotel",
                   "Entire townhouse", "Entire guest suite"]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

# ---------------------------------------------------------------------------
# UI - LISTING PROFILE
# ---------------------------------------------------------------------------
st.title("Austin Airbnb upgrade advisor")
st.caption("Enter your listing details, then see which upgrade pays back fastest.")

col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", 0, 10, 2)
    beds = st.number_input("Beds", 0, 10, 3)
    review_score = st.slider("Review score", 3.0, 5.0, 4.8, 0.1)
    neighbourhood = st.selectbox("Neighbourhood (zip)", NEIGHBOURHOODS)
with col2:
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.5, 0.5)
    accommodates = st.number_input("Sleeps", 1, 16, 4)
    number_of_reviews = st.number_input("Number of reviews", 0, 500, 20)
    property_type = st.selectbox("Property type", PROPERTY_TYPES)

room_type = st.selectbox("Room type", ROOM_TYPES)

col3, col4 = st.columns(2)
with col3:
    host_is_superhost = st.checkbox("Superhost")
with col4:
    instant_bookable = st.checkbox("Instant bookable")

st.subheader("Amenities you already have")
amenity_cols = st.columns(5)
current_amenities = {}
for col, (label, key) in zip(amenity_cols, UPGRADE_FLAGS.items()):
    with col:
        current_amenities[key] = st.checkbox(label.replace("Add ", "").replace("Make ", ""), key=f"has_{key}")

occupancy = st.slider("Assumed occupancy for revenue estimate", 10, 90, 50, 5) / 100

# ---------------------------------------------------------------------------
# BUILD FEATURE ROW MATCHING THE TRAINED MODEL'S EXPECTED COLUMNS
# ---------------------------------------------------------------------------
def build_row(amenities_state):
    return pd.DataFrame([{
        "bedrooms": bedrooms, "bathrooms_clean": bathrooms, "beds": beds,
        "accommodates": accommodates, "review_scores_rating": review_score,
        "number_of_reviews": number_of_reviews,
        "pet_friendly": amenities_state["pet_friendly"],
        "self_check_in": amenities_state["self_check_in"],
        "parking": amenities_state["parking"],
        "hot_tub_pool": amenities_state["hot_tub_pool"],
        "luxury_amenities": amenities_state["luxury_amenities"],
        "host_is_superhost": int(host_is_superhost),
        "instant_bookable": int(instant_bookable),
        "room_type": room_type, "property_type": property_type,
        "neighbourhood_cleansed": neighbourhood,
    }])

# ---------------------------------------------------------------------------
# PREDICTED PRICE
# ---------------------------------------------------------------------------
base_row = build_row(current_amenities)
base_price = model.predict(base_row)[0]

st.markdown("---")
st.metric("Predicted nightly price", f"${base_price:,.2f}")

# ---------------------------------------------------------------------------
# UPGRADE ROI RANKING - the core decision-support output
# ---------------------------------------------------------------------------
st.subheader("Recommended upgrades, ranked by payback")

results = []
for label, flag_col in UPGRADE_FLAGS.items():
    if current_amenities[flag_col] == 1:
        continue  # already has this amenity
    upgraded_amenities = dict(current_amenities)
    upgraded_amenities[flag_col] = 1
    upgraded_row = build_row(upgraded_amenities)
    upgraded_price = model.predict(upgraded_row)[0]

    nightly_lift = upgraded_price - base_price
    annual_revenue_lift = nightly_lift * 365 * occupancy
    cost = UPGRADE_COST[label]
    payback = cost / annual_revenue_lift if annual_revenue_lift > 0 else float("inf")

    results.append({
        "Upgrade": label,
        "$/night": round(nightly_lift, 2),
        "Est. annual revenue lift": round(annual_revenue_lift, 2),
        "Cost": cost,
        "Payback (years)": round(payback, 2) if payback != float("inf") else None,
    })

if results:
    roi_df = pd.DataFrame(results).sort_values(
        "Payback (years)", na_position="last"
    ).reset_index(drop=True)
    st.dataframe(roi_df, use_container_width=True, hide_index=True)

    negative = roi_df[roi_df["Payback (years)"].isna()]
    if not negative.empty:
        st.caption(
            "Upgrades with no listed payback show a negative price association in "
            "this model -- likely because budget listings tend to advertise these "
            "amenities more than luxury ones (confounding), not because the amenity "
            "itself lowers price. Treat as correlational, not causal."
        )
else:
    st.info("You already have every upgrade this model tracks.")

st.markdown("---")
st.caption(
    "Model: Linear Regression trained on 10,184 Austin listings "
    "(Inside Airbnb). Test set RMSE $203.28, R\u00b2 0.527."
)
