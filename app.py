"""
Flask web app for the Poker Preflop GTO Study Tool.
Uses pre-computed probabilities from BEST_GTO__optA_46pct_accuracy/gto_results.csv.
"""

import os
import sys
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

RESULTS_CSV = os.path.join("models", "BEST_GTO__optA_46pct_accuracy", "gto_results.csv")

results_df = pd.read_csv(RESULTS_CSV)
print(f"Loaded {len(results_df)} pre-computed results from {RESULTS_CSV}")

RANK_NAMES = {
    14: "A", 13: "K", 12: "Q", 11: "J", 10: "T",
    9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2",
}
SCENARIO_LABELS = {
    "SB_RFI":     "SB Open (First to Act)",
    "BB_vs_RFI":  "BB vs SB Open",
    "SB_vs_3bet": "SB vs BB 3-Bet",
}


def hand_name(r1, r2, suited):
    n1, n2 = RANK_NAMES[r1], RANK_NAMES[r2]
    if r1 == r2:
        return f"{n1}{n2}"
    return f"{n1}{n2}s" if suited else f"{n1}{n2}o"


def lookup(r1, r2, suited, scenario):
    """Look up pre-computed agent probs + GTO action from the results CSV."""
    hi, lo = max(r1, r2), min(r1, r2)
    is_suited = suited and (hi != lo)
    row = results_df[
        (results_df["rank1"]    == hi) &
        (results_df["rank2"]    == lo) &
        (results_df["suited"]   == int(is_suited)) &
        (results_df["scenario"] == scenario)
    ]
    if row.empty:
        return None
    r = row.iloc[0]
    total = r["fold_prob"] + r["call_prob"] + r["raise_prob"]
    return {
        "hand":       hand_name(hi, lo, is_suited),
        "gto_action": r["agent_action"],
        "probs": {
            "fold":  round(float(r["fold_prob"])  / total * 100, 1),
            "call":  round(float(r["call_prob"])  / total * 100, 1),
            "raise": round(float(r["raise_prob"]) / total * 100, 1),
        },
        "scenario": SCENARIO_LABELS[scenario],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error("Server error: %s", e)
    return jsonify({"error": "Internal server error"}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    data     = request.json
    r1       = int(data["rank1"])
    r2       = int(data["rank2"])
    suited   = bool(data["suited"])
    scenario = data["scenario"]

    if r1 not in range(2, 15) or r2 not in range(2, 15):
        return jsonify({"error": "Invalid rank: must be 2–14"}), 400
    if scenario not in SCENARIO_LABELS:
        return jsonify({"error": f"Invalid scenario. Valid options: {list(SCENARIO_LABELS.keys())}"}), 400

    logger.info("predict rank1=%d rank2=%d suited=%s scenario=%s", r1, r2, suited, scenario)
    result = lookup(r1, r2, suited, scenario)
    if result is None:
        return jsonify({"error": "Hand not found"}), 404
    return jsonify(result)


@app.route("/api/quiz", methods=["GET"])
def quiz():
    scenario = request.args.get("scenario")
    pool = results_df[results_df["scenario"] == scenario] if scenario in SCENARIO_LABELS else results_df
    row = pool.sample(1).iloc[0]
    r1, r2   = int(row["rank1"]), int(row["rank2"])
    suited   = bool(int(row["suited"]))
    scenario = row["scenario"]
    return jsonify({
        "rank1":          r1,
        "rank2":          r2,
        "suited":         suited,
        "scenario":       scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "hand":           row["hand"],
        "gto_action":     row["agent_action"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
