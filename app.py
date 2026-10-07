from flask import Flask, render_template, request, jsonify
from predict import analyze_traffic

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()

        result = analyze_traffic(
            data["vehicle_count"],
            data["avg_speed_kmh"],
            data["occupancy"] / 100.0,
            data["avg_wait_time_sec"]
        )

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
