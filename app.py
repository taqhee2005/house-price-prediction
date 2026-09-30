from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "house_price_model.pkl"
RESULTS_PATH = BASE / "models" / "model_results.json"
DATA_PATH = BASE / "dataset" / "house_prices.csv"

app = Flask(
    __name__,
    template_folder=str(BASE / "templates"),
    static_folder=str(BASE / "static")
)

# Load model
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

# Load evaluation metadata
model_meta = {}
if RESULTS_PATH.exists():
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            model_meta = json.load(f)
    except Exception as e:
        print(f"Error loading model results: {e}")

# Load dataset for market analytics
df_market = pd.DataFrame()
if DATA_PATH.exists():
    try:
        df_market = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        area = float(data.get("area", 1650))
        bedrooms = int(data.get("bedrooms", 3))
        bathrooms = int(data.get("bathrooms", 2))
        stories = int(data.get("stories", 2))
        parking = int(data.get("parking", 1))
        location = str(data.get("location", "Pune"))
        furnishing = str(data.get("furnishing", "Semi-Furnished"))

        input_df = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "location": location,
            "furnishing": furnishing
        }])

        pred_lakh = float(model.predict(input_df)[0])
        pred_rupees = pred_lakh * 100000

        # Formatting
        if pred_lakh >= 100:
            formatted_crore_lakh = f"₹ {pred_lakh / 100:.2f} Cr"
        else:
            formatted_crore_lakh = f"₹ {pred_lakh:.2f} Lakh"

        formatted_exact = f"₹ {pred_rupees:,.0f}"
        price_per_sqft = round(pred_rupees / max(area, 1))

        # Confidence bounds using model RMSE
        best_algo = model_meta.get("best_model", "Gradient Boosting")
        rmse = model_meta.get("results", {}).get(best_algo, {}).get("RMSE_lakh", 14.07)
        low_bound = max(10.0, pred_lakh - rmse)
        high_bound = pred_lakh + rmse

        low_str = f"₹ {low_bound / 100:.2f} Cr" if low_bound >= 100 else f"₹ {low_bound:.1f} L"
        high_str = f"₹ {high_bound / 100:.2f} Cr" if high_bound >= 100 else f"₹ {high_bound:.1f} L"

        # Tier classification
        if pred_lakh >= 400:
            tier = "Ultra-Luxury Estate"
            tier_class = "badge-amber"
        elif pred_lakh >= 200:
            tier = "Premium Residential"
            tier_class = "badge-emerald"
        else:
            tier = "Mid-Market Residential"
            tier_class = "badge-primary"

        return jsonify({
            "success": True,
            "price_lakh": round(pred_lakh, 2),
            "price_rupees": round(pred_rupees),
            "formatted_crore_lakh": formatted_crore_lakh,
            "formatted_exact": formatted_exact,
            "price_per_sqft": price_per_sqft,
            "confidence_low_str": low_str,
            "confidence_high_str": high_str,
            "tier": tier,
            "tier_class": tier_class
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/market-data", methods=["GET"])
def market_data():
    if df_market.empty:
        return jsonify({"success": False, "error": "No market data available"}), 404

    try:
        # City averages
        city_grp = df_market.groupby("location")["price_lakh"].mean().reset_index()
        city_grp = city_grp.sort_values(by="price_lakh", ascending=False)
        city_averages = [
            {"location": row["location"], "avg_price_lakh": round(row["price_lakh"], 2)}
            for _, row in city_grp.iterrows()
        ]

        # Sample points for scatter plot
        sample = df_market.sample(min(120, len(df_market)), random_state=42)
        sample_points = [
            {
                "area": int(row["area"]),
                "price_lakh": round(float(row["price_lakh"]), 2),
                "location": row["location"],
                "bedrooms": int(row["bedrooms"])
            }
            for _, row in sample.iterrows()
        ]

        return jsonify({
            "success": True,
            "city_averages": city_averages,
            "sample_points": sample_points
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "success": True,
        "metadata": model_meta
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "records_count": len(df_market)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
