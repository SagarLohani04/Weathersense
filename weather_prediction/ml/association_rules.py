"""
ml/association_rules.py
WeatherSense AI - Weather Pattern Mining (Association Rules)

Discretizes continuous weather variables into categorical bins
(e.g. "High Humidity", "Low Pressure") and applies the Apriori
algorithm to discover frequent itemsets and association rules
(support / confidence / lift) linking weather-condition combinations
to outcomes such as rainfall or fog.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def bin_value(value, low_thresh, high_thresh, label):
    if pd.isna(value):
        return None
    if value < low_thresh:
        return f"Low {label}"
    if value > high_thresh:
        return f"High {label}"
    return f"Moderate {label}"


def build_transactions(df):
    df = df.copy()
    items = []
    for _, row in df.iterrows():
        basket = []
        basket.append(bin_value(row["humidity"], 40, 70, "Humidity"))
        basket.append(bin_value(row["pressure"], 1008, 1016, "Pressure"))
        basket.append(bin_value(row["cloud_cover"], 30, 70, "Cloud Cover"))
        basket.append(bin_value(row["temperature"], 20, 33, "Temperature"))
        basket.append(bin_value(row["wind_speed"], 8, 25, "Wind Speed"))
        basket.append(bin_value(row.get("visibility", 8), 4, 8, "Visibility"))
        if row.get("rainfall", 0) > 20:
            basket.append("High Rainfall")
        if row.get("heatwave_indicator", 0) == 1:
            basket.append("Heatwave")
        if row.get("fog_indicator", 0) == 1:
            basket.append("Fog Event")
        basket = [b for b in basket if b]
        items.append(basket)
    return items


def run_association_mining(df, min_support=0.03, min_confidence=0.4, sample_n=8000, seed=42):
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed)

    transactions = build_transactions(df)
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_array, columns=te.columns_)

    freq_items = apriori(trans_df, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence, num_itemsets=len(trans_df))
    rules = rules.sort_values("lift", ascending=False)

    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))

    # Keep the most interesting, human-readable rules
    rules = rules[rules["consequents"].isin(
        ["High Rainfall", "Fog Event", "Heatwave"] +
        [c for c in rules["consequents"].unique() if "High Rainfall" in c or "Fog" in c or "Heatwave" in c]
    )]

    out = rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/weather_features.csv")
    rules = run_association_mining(df)
    print(f"Discovered {len(rules)} association rules")
    print(rules.head(15).to_string(index=False))
    rules.to_csv("data/association_rules.csv", index=False)
    print("Saved data/association_rules.csv")
