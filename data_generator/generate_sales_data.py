"""
Synthetic B2B specialty-foods sales data with embedded purchase-pattern
structure, so the recommendation engine has real signal to find.

Customers belong to one of four buyer personas (steakhouse, grocery retail,
sushi & seafood, deli & charcuterie), each with a preferred SKU pool.
Orders draw ~80% from the persona pool and ~20% from the full catalog —
which is exactly the structure collaborative filtering should recover,
and what the holdout evaluation in evaluation/ verifies it recovers.

Synthetic only: no real customers, reps, suppliers, or prices.
Fixed seed for reproducibility.

Usage:
    python data_generator/generate_sales_data.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = ["Alberta", "BC Interior", "BC Lower Mainland", "Ontario"]
REPS = ["A. Fontaine", "B. Okafor", "C. Delgado", "D. Tremblay",
        "E. Novak", "F. Osei", "G. Lindqvist", "H. Moreau"]

# (sku, protein, description, unit_cost, unit_price)
CATALOG = []
_sku_id = 0
def _add(protein, items):
    global _sku_id
    for desc, cost, price in items:
        _sku_id += 1
        CATALOG.append((f"SKU-{_sku_id:03d}", protein, desc, cost, price))

_add("Beef", [("Striploin AAA", 18.5, 26.0), ("Ribeye AAA", 21.0, 30.5),
              ("Tenderloin", 28.0, 41.0), ("Brisket", 7.2, 10.8),
              ("Ground Chuck", 5.1, 7.6), ("Short Ribs", 9.8, 14.5),
              ("Flank Steak", 11.0, 16.2), ("Beef Bones", 2.1, 3.6)])
_add("Pork", [("Berkshire Loin", 6.8, 10.4), ("Belly Skin-On", 5.9, 9.2),
              ("Shoulder Butt", 4.2, 6.5), ("Back Ribs", 6.1, 9.5),
              ("Prosciutto Leg", 12.5, 19.0), ("Chorizo Sausage", 6.4, 10.1)])
_add("Poultry", [("Chicken Breast", 5.6, 8.4), ("Chicken Thigh BL", 4.9, 7.3),
                 ("Whole Duck", 8.7, 13.5), ("Duck Breast", 14.2, 21.5),
                 ("Turkey Breast Roast", 7.8, 11.9), ("Quail 4-Pack", 9.5, 15.0)])
_add("Seafood", [("Atlantic Salmon Fillet", 13.5, 20.5), ("Sablefish Fillet", 19.8, 29.5),
                 ("Ahi Tuna Saku", 22.0, 33.5), ("Hamachi Loin", 24.5, 37.0),
                 ("Scallops U10", 26.0, 39.5), ("Spot Prawns", 18.9, 28.5),
                 ("Uni Tray", 38.0, 58.0), ("Octopus Tentacle", 16.4, 25.0)])
_add("Deli", [("Genoa Salami", 8.9, 13.8), ("Smoked Speck", 11.2, 17.4),
              ("Duck Rillette", 9.9, 15.6), ("Pate de Campagne", 7.4, 11.8),
              ("Coppa Sliced", 12.8, 19.9), ("Bresaola", 15.5, 24.0)])
_add("Lamb", [("Lamb Rack Frenched", 24.0, 35.5), ("Lamb Leg BRT", 11.5, 17.4),
              ("Lamb Shank", 8.4, 12.9), ("Ground Lamb", 7.9, 12.0)])

PERSONAS = {
    "steakhouse":   {"Beef": 0.55, "Lamb": 0.18, "Pork": 0.12, "Poultry": 0.08, "Seafood": 0.05, "Deli": 0.02},
    "grocery":      {"Beef": 0.25, "Pork": 0.22, "Poultry": 0.28, "Deli": 0.12, "Lamb": 0.07, "Seafood": 0.06},
    "sushi":        {"Seafood": 0.72, "Poultry": 0.10, "Beef": 0.10, "Pork": 0.04, "Deli": 0.02, "Lamb": 0.02},
    "charcuterie":  {"Deli": 0.48, "Pork": 0.24, "Poultry": 0.10, "Beef": 0.10, "Lamb": 0.05, "Seafood": 0.03},
}
PERSONA_NAMES = {
    "steakhouse": ["Grill", "Chophouse", "Steakhouse", "Tavern", "Prime"],
    "grocery": ["Market", "Grocer", "Foods", "Fresh Mart", "Pantry"],
    "sushi": ["Sushi", "Izakaya", "Kaiseki", "Ocean Bar", "Omakase"],
    "charcuterie": ["Deli", "Salumeria", "Provisions", "Charcuterie", "Cellar"],
}
NAME_STEMS = ["Aurora", "Granville", "Cedar", "Harbour", "Summit", "Willow",
              "Copper", "Juniper", "Lakeside", "Meridian", "Nordic", "Orchard",
              "Pacific", "Quarry", "Riverbend", "Sterling", "Timber", "Union",
              "Vista", "Whistler", "Yardley", "Zenith", "Bluffs", "Canyon",
              "Dockside", "Elmwood", "Foothill", "Grove", "Highline", "Inlet"]

N_CUSTOMERS = 120
N_ORDERS = 2600
START = date(2025, 7, 1)
DAYS = 365


def main():
    by_protein = {}
    for sku in CATALOG:
        by_protein.setdefault(sku[1], []).append(sku)

    customers = []
    for i in range(1, N_CUSTOMERS + 1):
        persona = random.choices(list(PERSONAS), weights=[0.3, 0.3, 0.2, 0.2])[0]
        stem = random.choice(NAME_STEMS)
        suffix = random.choice(PERSONA_NAMES[persona])
        customers.append({
            "customer_id": f"CUST-{i:03d}",
            "customer_name": f"{stem} {suffix} {i:03d}",
            "persona": persona,
            "region": random.choice(REGIONS),
            "rep": random.choice(REPS),
            # customer size multiplier drives order frequency & basket size
            "size": random.lognormvariate(0, 0.6),
        })

    lines = []
    order_id = 0
    weights = [c["size"] for c in customers]
    for _ in range(N_ORDERS):
        order_id += 1
        c = random.choices(customers, weights=weights)[0]
        order_date = START + timedelta(days=random.randint(0, DAYS - 1))
        persona_mix = PERSONAS[c["persona"]]
        n_lines = max(1, int(random.gauss(4 + 2 * c["size"], 2)))
        chosen = set()
        for _ in range(n_lines):
            # 80% persona-driven, 20% exploration across the catalog
            if random.random() < 0.8:
                protein = random.choices(list(persona_mix), weights=persona_mix.values())[0]
                sku = random.choice(by_protein[protein])
            else:
                sku = random.choice(CATALOG)
            if sku[0] in chosen:
                continue
            chosen.add(sku[0])
            qty = max(1, int(random.lognormvariate(2.2, 0.7)))
            price = round(sku[4] * random.uniform(0.96, 1.04), 2)
            lines.append({
                "order_id": f"ORD-{order_id:05d}",
                "order_date": order_date.isoformat(),
                "customer_id": c["customer_id"],
                "customer_name": c["customer_name"],
                "region": c["region"],
                "rep": c["rep"],
                "sku": sku[0],
                "protein": sku[1],
                "description": sku[2],
                "quantity_lb": qty,
                "unit_price": price,
                "revenue": round(qty * price, 2),
                "cost": round(qty * sku[3], 2),
            })

    with open(OUT / "sales_lines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=lines[0].keys())
        w.writeheader()
        w.writerows(lines)

    with open(OUT / "customers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["customer_id", "customer_name", "persona",
                                          "region", "rep"])
        w.writeheader()
        w.writerows([{k: c[k] for k in ("customer_id", "customer_name", "persona",
                                        "region", "rep")} for c in customers])

    with open(OUT / "catalog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sku", "protein", "description", "unit_cost", "unit_price"])
        w.writerows(CATALOG)

    print(f"wrote {len(lines)} order lines, {len(customers)} customers, "
          f"{len(CATALOG)} SKUs -> {OUT}")


if __name__ == "__main__":
    main()
