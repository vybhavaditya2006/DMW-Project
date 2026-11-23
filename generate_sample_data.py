"""
Generate a realistic sample Bengaluru_House_Data.csv for development/testing.
Run this only if you don't already have the dataset from Kaggle.

Usage:
    python generate_sample_data.py
"""

import random
import csv
import os

random.seed(42)

# ── Reference lists ──────────────────────────────────────────────────────────
AREA_TYPES = [
    "Super built-up  Area",
    "Built-up  Area",
    "Plot  Area",
    "Carpet  Area",
]
AREA_WEIGHTS = [0.45, 0.35, 0.10, 0.10]

LOCATIONS = [
    "Whitefield", "Sarjapur Road", "Electronic City", "Bannerghatta Road",
    "Thanisandra", "Yelahanka", "Hebbal", "KR Puram", "Marathahalli",
    "Raja Rajeshwari Nagar", "Peenya", "Bellandur", "JP Nagar",
    "Banashankari", "HSR Layout", "Koramangala", "Indira Nagar",
    "Jayanagar", "BTM Layout", "Bommanahalli", "Sahakara Nagar",
    "Yeshwanthpur", "Hennur Road", "Hoodi", "Uttarahalli",
    "Kanakapura Road", "Old Madras Road", "Rachenahalli", "Harlur",
    "Kasavanhalli", "Domlur", "Rajaji Nagar", "Frazer Town",
    "Brookefield", "Varthur", "Akshaya Nagar", "Chandapura",
    "Hormavu", "Nagarbhavi", "Kalyan Nagar",
]

SOCIETIES = [
    "Cleo County", "Prestige Shantiniketan", "Sobha Dream Garden",
    "Brigade Meadows", "Mantri Serenity", "Purva Venezia",
    "Salarpuria Sattva", "Embassy Pristine", "RMZ Galleria",
    "Godrej Woodsman", None, None, None, None,  # ~30 % missing
]

AVAILABILITY = [
    "Ready To Move",
    "19-Dec", "18-Jun", "18-Sep", "19-Mar", "18-Nov", "18-Apr",
    "19-Jan", "18-Aug", "19-Feb", "18-Jul",
]
AVAIL_WEIGHTS = [0.60] + [0.04] * 10


def random_total_sqft(bhk: int) -> str:
    """Return a total_sqft string — occasionally a range."""
    base = bhk * random.randint(300, 700)
    if random.random() < 0.08:  # 8 % are ranges
        lo = base
        hi = base + random.randint(50, 300)
        return f"{lo} - {hi}"
    return str(base)


def generate_rows(n: int = 13_320) -> list[dict]:
    rows = []
    for _ in range(n):
        bhk = random.choices([1, 2, 3, 4, 5, 6, 7, 8],
                             weights=[0.08, 0.30, 0.35, 0.15, 0.06, 0.03, 0.02, 0.01])[0]
        bath = max(1, bhk + random.choice([-1, 0, 0, 0, 1]))
        balcony = random.choices([0, 1, 2, 3], weights=[0.10, 0.40, 0.35, 0.15])[0]

        sqft_str = random_total_sqft(bhk)
        # derive numeric sqft for price calc
        if " - " in sqft_str:
            lo, hi = sqft_str.split(" - ")
            sqft_num = (float(lo) + float(hi)) / 2
        else:
            sqft_num = float(sqft_str)

        location = random.choice(LOCATIONS)

        # Price model  (lakhs)
        price_per_sqft = random.gauss(5500, 2000)  # ₹ per sqft
        price_per_sqft = max(2000, price_per_sqft)
        location_factor = 1 + (LOCATIONS.index(location) % 10) * 0.04
        price = round((sqft_num * price_per_sqft * location_factor) / 1e5, 2)
        price = max(8.0, price)

        size_str = f"{bhk} BHK" if random.random() < 0.85 else f"{bhk} Bedroom"

        society = random.choice(SOCIETIES)

        row = {
            "area_type": random.choices(AREA_TYPES, weights=AREA_WEIGHTS)[0],
            "availability": random.choices(AVAILABILITY, weights=AVAIL_WEIGHTS)[0],
            "location": location,
            "size": size_str if random.random() < 0.98 else "",  # 2 % missing
            "society": society if society else "",
            "total_sqft": sqft_str if random.random() < 0.97 else "",  # 3 % missing
            "bath": str(bath) if random.random() < 0.97 else "",
            "balcony": str(balcony) if random.random() < 0.95 else "",
            "price": str(price) if random.random() < 0.99 else "",
        }
        rows.append(row)
    return rows


def main():
    out_path = os.path.join(os.path.dirname(__file__), "Bengaluru_House_Data.csv")
    rows = generate_rows()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["area_type", "availability", "location", "size",
                         "society", "total_sqft", "bath", "balcony", "price"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅  Generated {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
