from flask import Flask, request, jsonify
import sqlite3
import numpy as np
from numpy.linalg import norm
import os
from bert_loader import load_legalbert_embedding_pipeline

app = Flask(__name__)

# Configuratie voor update
API_INDEX_URL    = "https://api.rechtspraak.nl/v1/index"
API_DOCUMENT_URL = "https://api.rechtspraak.nl/v1/document"
# URI voor bestuursrecht/omgevingsrecht als string
RECHTSGEBIED_URI = "http://psi.rechtspraak.nl/rechtsgebied#bestuursrecht_omgevingsrecht"

# 🛠️ Functie om database bij te werken met nieuwe uitspraken
def update_db(db_path="jurisprudentie.db"):
    # lokale imports om geheugen web-service te sparen
    import requests
    # laad embedder
    embedder = load_legalbert_embedding_pipeline()
    # connectie maken
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # maak tabel indien nodig
    cur.execute("""
    CREATE TABLE IF NOT EXISTS uitspraken (
        ecli TEXT PRIMARY KEY,
        titel TEXT,
        samenvatting TEXT,
        link TEXT,
        embedding BLOB
    )""")
    # haal nieuwe ECLI's op
    payload = {"criteria": {"rechtsgebied": [RECHTSGEBIED_URI], "maxRecords": 100}}
    idx_resp = requests.post(API_INDEX_URL, json=payload)
    if idx_resp.status_code != 200:
        print(f"Index-opvraag mislukt: {idx_resp.text}")
        return
    eclis = [item["ecli"] for item in idx_resp.json().get("results", [])]
    # verwerk elke ECLI
    for ecli in eclis:
        doc_resp = requests.get(f"{API_DOCUMENT_URL}?ecli={ecli}")
        if doc_resp.status_code != 200:
            continue
        doc = doc_resp.json()
        # filter rechtsgebied
        gebieden = [g.get("uri") for g in doc.get("inRechtsgebied", [])]
        if RECHTSGEBIED_URI not in gebieden:
            continue
        # bereid data
        titel = doc.get("title", "")
        samenv = doc.get("berichttekst", "")[:2000]
        link = doc.get("documentUrl", "")
        vec = embedder(samenv)
        blob = np.array(vec, dtype=np.float32).tobytes()
        # insert or replace
        cur.execute(
            "INSERT OR REPLACE INTO uitspraken (ecli, titel, samenvatting, link, embedding) VALUES (?,?,?,?,?)",
            (ecli, titel, samenv, link, blob)
        )
    conn.commit()
    conn.close()
    print(f"🗃️ Database bijgewerkt met {len(eclis)} uitspraken.")

# 🛑 Database-update wordt alleen lokaal/CI uitgevoerd via --update-db flag

# 📦 Load alle embeddings uit SQLite bij opstart
def load_embeddings(db_path="jurisprudentie.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT ecli, titel, samenvatting, link, embedding FROM uitspraken")
    data = []
    for ecli, titel, samenv, link, blob in cursor.fetchall():
        vec = np.frombuffer(blob, dtype=np.float32)
        data.append((ecli, titel, samenv, link, vec))
    conn.close()
    return data

uitspraken = load_embeddings()

# Bereken cosine similarity tussen twee vectors
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# 🤖 Endpoint – Embedding-based suggesties uit eigen DB
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

# Registreer endpoint en health-check
@app.route('/suggesties', methods=['POST'])
def suggesties_route():
    return suggesties()

@app.route('/', methods=['GET'])
def health_check():
    return "Jurisprudentie-zoeker draait!", 200

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--update-db', action='store_true', help='Voer database update uit')
    args = parser.parse_args()

    if args.update_db:
        update_db()
        exit(0)

    port = int(os.environ.get("PORT", 8080))
    print(f"⚙️ Starting server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

