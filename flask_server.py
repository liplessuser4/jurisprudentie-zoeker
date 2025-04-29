from flask import Flask, request, jsonify
import sqlite3
import numpy as np
from numpy.linalg import norm
import os
from bert_loader import load_legalbert_embedding_pipeline

app = Flask(__name__)

# Configuratie
API_INDEX_URL    = "https://api.rechtspraak.nl/v1/index"
API_DOCUMENT_URL = "https://api.rechtspraak.nl/v1/document"
RECHTSGEBIED_URI = "http://psi.rechtspraak.nl/rechtsgebied#bestuursrecht_omgevingsrecht"
DB_PATH          = "jurisprudentie.db"

# Laad embeddings eenmaal bij opstart
rather_load_embeddings = load_legalbert_embedding_pipeline

def load_embeddings(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT ecli, titel, samenvatting, link, embedding FROM uitspraken")
    data = [(ecli, titel, samenv, link, np.frombuffer(blob, dtype=np.float32))
            for ecli, titel, samenv, link, blob in cursor.fetchall()]
    conn.close()
    return data

uitspraken = load_embeddings()

# Bereken cosine similarity
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# Fallback: haal één relevante ECLI met samenvatting via Open Data API
def fallback_analysis(searchterm=None):
    import requests
    criteria = {"rechtsgebied": [RECHTSGEBIED_URI]}
    if searchterm:
        criteria["searchterm"] = searchterm
    payload = {"criteria": criteria, "maxRecords": 1}
    idx_resp = requests.post(API_INDEX_URL, json=payload)
    if idx_resp.status_code != 200 or not idx_resp.json().get("results"):
        return None
    ecli = idx_resp.json()["results"][0]["ecli"]
    doc_resp = requests.get(f"{API_DOCUMENT_URL}?ecli={ecli}")
    if doc_resp.status_code != 200:
        return None
    doc = doc_resp.json()
    return {
        "ecli": doc.get("ecli"),
        "titel": doc.get("title"),
        "samenvatting": (doc.get("berichttekst", "") or doc.get("summary", ""))[:300] + "...",
        "link": doc.get("documentUrl")
    }

# Endpoint: suggesties met fallback
@app.route('/suggesties', methods=['POST'])
def suggesties_route():
    data = request.json or {}
    embedding = data.get("embedding")
    searchterm = data.get("searchterm")
    if not embedding:
        return jsonify({"error": "embedding ontbreekt"}), 400

    q = np.array(embedding, dtype=np.float32)
    scored = [(cosine_similarity(q, vec), ecli, titel, samenv, link)
              for ecli, titel, samenv, link, vec in uitspraken]
    top5 = sorted(scored, key=lambda x: x[0], reverse=True)[:5]

    # Trigger fallback als geen goede match
    if not top5 or top5[0][0] < 0.1:
        fallback = fallback_analysis(searchterm)
        if fallback:
            return jsonify({"fallback": True, **fallback}), 200
        return jsonify({"fallback": True, "error": "Geen data gevonden"}), 404

    results = [{"ecli": e, "titel": t, "samenvatting": s, "link": l, "score": round(score,4)}
               for score, e, t, s, l in top5]
    return jsonify({"fallback": False, "results": results}), 200

@app.route('/', methods=['GET'])
def health_check():
    return "Jurisprudentie-zoeker draait!", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"⚙️ Starting server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)

