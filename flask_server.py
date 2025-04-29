from flask import Flask, request, jsonify
import sqlite3
import numpy as np
from numpy.linalg import norm

app = Flask(__name__)

def load_embeddings():
    conn = sqlite3.connect("jurisprudentie.db")
    cursor = conn.cursor()
    cursor.execute("SELECT ecli, titel, samenvatting, link, embedding FROM uitspraken")
    uitspraken = []
    for row in cursor.fetchall():
        ecli, titel, samenvatting, link, embedding_blob = row
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        uitspraken.append((ecli, titel, samenvatting, link, embedding))
    conn.close()
    return uitspraken

def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))

uitspraken = load_embeddings()

@app.route("/suggesties", methods=["POST"])
def suggesties():
    data = request.json
    input_vec = data.get("embedding")
    if not input_vec:
        return jsonify({"error": "embedding ontbreekt"}), 400

    input_vec = np.array(input_vec, dtype=np.float32)
    scores = []
    for ecli, titel, samenvatting, link, vec in uitspraken:
        score = cosine_similarity(input_vec, vec)
        scores.append((score, ecli, titel, samenvatting, link))

    top5 = sorted(scores, reverse=True)[:5]
    resultaten = [{
        "ecli": ecli,
        "titel": titel,
        "samenvatting": samenvatting,
        "link": link,
        "score": round(score, 4)
    } for score, ecli, titel, samenvatting, link in top5]

    return jsonify(resultaten)

@app.route("/")
def health():
    return "Render-versie jurisprudentiezoeker draait!", 200

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

