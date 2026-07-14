"""
Quick sanity test for pricing_model.joblib

Run with: python test_model.py

This does NOT re-measure accuracy (that's already reported by
airbnb_idss_model.py as Test RMSE / R^2 on the held-out test set).
This just confirms the saved model loads correctly and produces
sane, differentiated predictions -- a basic "does the code work" check.
"""

import joblib
import pandas as pd

model = joblib.load("pricing_model.joblib")

# A handful of very different listing profiles, so we can eyeball whether
# the model responds sensibly (bigger/nicer listing -> higher price, etc.)
test_listings = pd.DataFrame([
    {  # small budget room
        "bedrooms": 1, "bathrooms_clean": 1, "beds": 1, "accommodates": 2,
        "review_scores_rating": 4.5, "number_of_reviews": 10,
        "pet_friendly": 0, "self_check_in": 0, "parking": 0,
        "hot_tub_pool": 0, "luxury_amenities": 0,
        "host_is_superhost": 0, "instant_bookable": 0,
        "room_type": "Private room", "property_type": "Private room in home",
        "neighbourhood_cleansed": "78741",
    },
    {  # mid-size family home
        "bedrooms": 3, "bathrooms_clean": 2, "beds": 4, "accommodates": 6,
        "review_scores_rating": 4.8, "number_of_reviews": 50,
        "pet_friendly": 1, "self_check_in": 1, "parking": 1,
        "hot_tub_pool": 0, "luxury_amenities": 0,
        "host_is_superhost": 1, "instant_bookable": 1,
        "room_type": "Entire home/apt", "property_type": "Entire home",
        "neighbourhood_cleansed": "78704",
    },
    {  # luxury listing with pool
        "bedrooms": 5, "bathrooms_clean": 4, "beds": 6, "accommodates": 10,
        "review_scores_rating": 4.95, "number_of_reviews": 100,
        "pet_friendly": 0, "self_check_in": 0, "parking": 0,
        "hot_tub_pool": 1, "luxury_amenities": 1,
        "host_is_superhost": 1, "instant_bookable": 1,
        "room_type": "Entire home/apt", "property_type": "Entire villa",
        "neighbourhood_cleansed": "78703",
    },
])

predictions = model.predict(test_listings)

print("Sanity check results:")
for i, price in enumerate(predictions):
    print(f"  Listing {i+1}: ${price:.2f}/night")

# Basic checks -- these should always hold for a working pricing model
assert all(p > 0 for p in predictions), "FAILED: model predicted a negative price"
assert predictions[2] > predictions[0], "FAILED: luxury listing priced lower than budget room"

print("\nAll checks passed -- model loads correctly and predictions are sane.")
