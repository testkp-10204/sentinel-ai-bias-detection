from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sentinel_utils import analyze_text
import sqlite3
from datetime import datetime
from check_db import insert_analysis
import csv
from flask import Response



app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/app.js")
def js():
    return send_file("app.js")


@app.route("/styles.css")
def css():
    return send_file("styles.css")


@app.route("/analyze")
def analyze():

    text = request.args.get("text")

    result = analyze_text(text)

    # save to database
    insert_analysis(text, result)

    result["text"] = text
    result["id"] = 1
    result["created_at"] = datetime.now().isoformat()

    return jsonify(result)



@app.route("/analyze/compare", methods=["POST"])
def compare():

    data = request.json
    text_a = data["text_a"]
    text_b = data["text_b"]

    result_a = analyze_text(text_a)
    result_b = analyze_text(text_b)

    comparison = {
        "more_positive": "A" if result_a["sentiment_score"] > result_b["sentiment_score"] else "B",
        "more_toxic": "A" if result_a["toxicity_score"] > result_b["toxicity_score"] else "B",
        "more_biased": "A" if result_a["bias_score"] > result_b["bias_score"] else "B",
        "risk_score_diff": result_a["risk_score"] - result_b["risk_score"]
    }

    return jsonify({
        "text_a": {**result_a, "text": text_a},
        "text_b": {**result_b, "text": text_b},
        "comparison": comparison
    })
@app.route("/stats")
def stats():

    conn = sqlite3.connect("instance/sentinel_audit.db")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM history")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM history WHERE toxicity='toxic'")
    toxic_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM history WHERE sentiment='POSITIVE'")
    positive_count = cur.fetchone()[0]

    conn.close()

    return jsonify({
        "total": total,
        "positive_count": positive_count,
        "negative_count": total - positive_count,
        "toxic_count": toxic_count,
        "avg_sentiment_score": 0.5,
        "avg_toxicity_score": 0.5
    })

@app.route("/history")
def history():

    limit = int(request.args.get("limit", 15))
    offset = int(request.args.get("offset", 0))

    conn = sqlite3.connect("instance/sentinel_audit.db")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM history")
    total = cur.fetchone()[0]

    cur.execute(
        "SELECT id,text,toxicity,sentiment,overall_risk,identity_mention,created_at FROM history ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    )

    rows = cur.fetchall()

    conn.close()

    items = []

    for r in rows:
        items.append({
            "id": r[0],
            "text": r[1],
            "toxicity": r[2],
            "sentiment": r[3],
            "overall_risk": r[4],
            "identity_mention": bool(r[5]),
            "created_at": r[6]
        })

    return jsonify({
        "total": total,
        "items": items
    })

@app.route("/analyze/batch", methods=["POST"])
def batch():

    data = request.json
    texts = data["texts"]

    results = []

    for t in texts:
        r = analyze_text(t)
        r["text"] = t
        results.append(r)

    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "positive_count": 0,
            "negative_count": 0,
            "toxic_count": sum(1 for r in results if r["toxicity"] == "toxic"),
            "safe_count": sum(1 for r in results if r["toxicity"] == "safe"),
            "avg_sentiment_score": 0.5,
            "avg_toxicity_score": 0.5,
            "avg_bias_score": 0.5
        }
    })
@app.route("/history/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):

    conn = sqlite3.connect("instance/sentinel_audit.db")
    cur = conn.cursor()

    cur.execute("DELETE FROM history WHERE id=?", (item_id,))
    conn.commit()
    conn.close()

    return jsonify({"status": "deleted"})

@app.route("/export")
def export_csv():

    conn = sqlite3.connect("instance/sentinel_audit.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM history")
    rows = cur.fetchall()

    conn.close()

    def generate():
        data = csv.writer(open("temp.csv", "w", newline=""))
    
    output = "id,text,toxicity,sentiment,bias,risk,date\n"

    for r in rows:
        output += f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[7]}\n"

    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sentinel_history.csv"}
    )



if __name__ == "__main__":
    app.run(port=8000, debug=True)