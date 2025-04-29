from flask import Flask, request, jsonify
import requests
import sqlite3
import numpy as np
from numpy.linalg import norm
from bert_loader import load_legalbert_embedding_pipeline
import os

app = Flask(__name__)

# Configuratie
API_INDEX_URL    = "https://api.rechtspraak.nl/v1/index"
API_DOCUMENT_URL = "https://api.rechtspraak.nl/v1/document"
RECHTSGEBIED_URI = "http://psi.rechtspraak.nl/rechtsgebied#bestuursrecht_omgevingsrecht"

# 🧠 Load embedding model bij opstart
enbed_pipeline = load_legalbert_embedding_pipeline()

# 📦 Load alle embeddings uit SQLite
def load_embeddings():
    conn = sqlite3.connect("jurisprudentie.db")
    cursor = conn.cursor()
    cursor.execute("SELECT ecli, titel, samenvatting, link, embedding FROM uitspraken")
    data = []
    for ecli, titel, samenv, link, blob in cursor.fetchall():
        vec = np.frombuffer(blob, dtype=np.float32)
        data.append((ecli, titel, samenv, link, vec))
    conn.close()
    return data

uitspraken = load_embeddings()

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# 🔎 Endpoint 1 – Zoek jurisprudentie (Open Data API)
@app.route("/jurisprudentie/zoek", methods=["POST"])
def zoek_jurisprudentie():
    data = request.json or {}
    zoekterm = data.get("zoekterm", "").strip()
    if not zoekterm:
        return jsonify({"error": "Zoekterm is verplicht"}), 400

    payload = {
        "criteria": {
            "rechtsgebied": [RECHTSGEBIED_URI],
            "searchterm": zoekterm,
            "maxRecords": 10
        }
    }
    idx_resp = requests.post(API_INDEX_URL, json=payload)
    if idx_resp.status_code != 200:
        return jsonify({"error": "Index-opvraag mislukt", "details": idx_resp.text}), 500

    results = []
    for item in idx_resp.json().get("results", []):
        ecli = item.get("ecli")
        doc_resp = requests.get(f"{API_DOCUMENT_URL}?ecli={ecli}")
        if doc_resp.status_code != 200:
            continue
        doc = doc_resp.json()
        gebieden = [g.get("uri") for g in doc.get("inRechtsgebied", [])]
        if RECHTSGEBIED_URI not in gebieden:
            continue
        results.append({
            "ecli": doc.get("ecli"),
            "titel": doc.get("title"),
            "datum": doc.get("decisionDate"),
            "link": doc.get("documentUrl"),
            "samenvatting": (doc.get("berichttekst") or doc.get("summary", ""))[:200] + "..."
        })

    return jsonify(results), 200

# 📊 Endpoint 2 – Analyseer ECLI en genereer embedding
@app.route("/jurisprudentie/analyse", methods=["POST"])
def analyseer_ecli():
    data = request.json or {}
    ecli = data.get("ecli")
    if not ecli:
        return jsonify({"error": "ECLI is verplicht"}), 400

    doc_resp = requests.get(f"{API_DOCUMENT_URL}?ecli={ecli}")
    if doc_resp.status_code != 200:
        return jsonify({"error": "Document niet gevonden", "details": doc_resp.text}), 404
    doc = doc_resp.json()

    tekst = doc.get("berichttekst", "")
    if not tekst:
        return jsonify({"error": "Geen uitspraaktekst gevonden"}), 404

    tekst_kort = tekst[:2000]
    embedding = enbed_pipeline(tekst_kort)

    return jsonify({
        "ecli": ecli,
        "titel": doc.get("title"),
        "samenvatting": tekst[:1000] + "...",
        "embedding": embedding
    }), 200

# 🤖 Endpoint 3 – Embedding-based suggesties uit eigen DB
@app.route("/suggesties", methods=["POST"])
def suggesties():
    data = request.json or {}
    input_vec = data.get("embedding")
    if not input_vec:
        return jsonify({"error": "embedding ontbreekt"}), 400

    q = np.array(input_vec, dtype=np.float32)
    scored = [
        (cosine_similarity(q, vec), ecli, titel, samenv, link)
        for ecli, titel, samenv, link, vec in uitspraken
    ]
    top5 = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
    results = [
        {"ecli": e, "titel": t, "samenvatting": s, "link": l, "score": round(score, 4)}
        for score, e, t, s, l in top5
    ]
    return jsonify(results), 200

# ✅ Health-check
@app.route("/", methods=["GET"])
def health_check():
    return "Jurisprudentie-proxy draait!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

