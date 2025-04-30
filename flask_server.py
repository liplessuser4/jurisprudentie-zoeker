# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import sqlite3
import numpy as np
from numpy.linalg import norm
import os
import requests

# Importeer de functie uit bert_loader.py
from bert_loader import load_legalbert_embedding_pipeline

app = Flask(__name__)

# --- Configuratie ---
API_INDEX_URL    = "https://api.rechtspraak.nl/v1/index"
API_DOCUMENT_URL = "https://api.rechtspraak.nl/v1/document"
RECHTSGEBIED_URI = "http://psi.rechtspraak.nl/rechtsgebied#bestuursrecht_omgevingsrecht"
DB_PATH          = "jurisprudentie.db"
FALLBACK_MAX_RECORDS = 5 # <<< Nieuw: Aantal resultaten voor fallback check
SIMILARITY_THRESHOLD = 0.6 # Drempelwaarde voor primaire zoekresultaten

# --- Initialisatie (bij opstarten server) ---

print("🚀 Initialiseren Legal BERT embedding model...")
try:
    get_embedding_function = load_legalbert_embedding_pipeline()
    print("✅ Legal BERT model geladen en embedding functie klaar.")
except Exception as e:
    print(f"❌ Fout bij laden Legal BERT model: {e}")
    get_embedding_function = None

def load_embeddings(db_path=DB_PATH):
    """Laadt ECLI, metadata en pre-computed embeddings uit de database."""
    # ... (code ongewijzigd, inclusief error handling) ...
    if not os.path.exists(db_path):
        print(f"❌ Fout: Databasebestand niet gevonden op {db_path}")
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ecli, titel, samenvatting, link, embedding FROM uitspraken")
        data = [(ecli, titel, samenv, link, np.frombuffer(blob, dtype=np.float32))
                for ecli, titel, samenv, link, blob in cursor.fetchall()]
        conn.close()
        print(f"✅ {len(data)} uitspraken met embeddings geladen uit {db_path}.")
        return data
    except Exception as e:
        print(f"❌ Fout bij laden embeddings uit database: {e}")
        return []

uitspraken = load_embeddings()

# --- Hulpfuncties ---

def cosine_similarity(a, b):
    """Berekent de cosine similarity tussen twee numpy arrays."""
    # ... (code ongewijzigd, inclusief zero norm check) ...
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# --- Fallback Functie (Aangepast) ---
def get_best_fallback_result(searchterm=None, query_embedding_vector=None):
    """
    Haalt meerdere fallback kandidaten op via Open Data API en selecteert
    de meest semantisch relevante op basis van de query embedding.
    """
    if not searchterm or query_embedding_vector is None or get_embedding_function is None:
        print("ℹ️ Fallback: Ontbrekende searchterm, query embedding of embedding functie.")
        return None # Noodzakelijke input ontbreekt

    print(f"⏳ Start fallback zoekopdracht voor '{searchterm}' met {FALLBACK_MAX_RECORDS} records...")
    criteria = {"rechtsgebied": [RECHTSGEBIED_URI], "searchterm": searchterm}
    # Vraag meerdere records op
    payload = {"criteria": criteria, "maxRecords": FALLBACK_MAX_RECORDS, "return": ["ecli"]} # Vraag alleen ECLI's op eerst
    candidate_results = []

    try:
        # Stap 1: Haal ECLI's op van potentiële kandidaten
        idx_resp = requests.post(API_INDEX_URL, json=payload, timeout=15)
        idx_resp.raise_for_status()
        idx_data = idx_resp.json()
        eclis = [res["ecli"] for res in idx_data.get("results", []) if "ecli" in res]

        if not eclis:
            print(f"ℹ️ Fallback: Geen index resultaten gevonden voor '{searchterm}'")
            return None
        
        print(f"ℹ️ Fallback: {len(eclis)} kandidaat ECLI's gevonden. Details ophalen...")

        # Stap 2: Haal details en genereer embeddings voor elke kandidaat
        for ecli in eclis:
            try:
                doc_resp = requests.get(f"{API_DOCUMENT_URL}?ecli={ecli}", timeout=10)
                doc_resp.raise_for_status()
                doc = doc_resp.json()

                # Pak samenvatting of (deel van) berichttekst
                summary_text = doc.get("summary", "") or doc.get("berichttekst", "")
                if not summary_text:
                    print(f"⚠️ Fallback: Geen samenvatting/tekst gevonden voor {ecli}")
                    continue # Sla deze kandidaat over

                # Genereer embedding voor de samenvatting/tekst van de kandidaat
                fallback_embedding_list = get_embedding_function(summary_text[:1024]) # Beperk lengte voor embedding
                fallback_vector = np.array(fallback_embedding_list, dtype=np.float32)

                # Bereken similariteit met de originele query embedding
                semantic_score = cosine_similarity(query_embedding_vector, fallback_vector)

                candidate_results.append({
                    "ecli": doc.get("ecli"),
                    "titel": doc.get("title"),
                    "samenvatting": summary_text[:500] + "...", # Neem eerste 500 tekens voor weergave
                    "link": doc.get("documentUrl"),
                    "semantic_score": semantic_score # Bewaar de semantische score
                })
                print(f"  - Kandidaat {ecli} verwerkt, score: {semantic_score:.4f}")

            except requests.exceptions.RequestException as e_doc:
                print(f"❌ Fout bij ophalen document {ecli}: {e_doc}")
            except Exception as e_emb:
                print(f"❌ Fout bij verwerken/embedden {ecli}: {e_emb}")

    except requests.exceptions.RequestException as e_idx:
        print(f"❌ Fout tijdens fallback index API call: {e_idx}")
        return None
    except Exception as e_general:
        print(f"❌ Onverwachte fout tijdens fallback: {e_general}")
        return None

    # Stap 3: Selecteer de beste kandidaat op basis van hoogste semantische score
    if not candidate_results:
        print("ℹ️ Fallback: Geen geldige kandidaten over na detail ophalen/verwerking.")
        return None

    best_result = max(candidate_results, key=lambda x: x["semantic_score"])
    print(f"✅ Beste fallback resultaat geselecteerd: {best_result['ecli']} (Score: {best_result['semantic_score']:.4f})")
    
    # Verwijder de interne score voor het eindresultaat
    del best_result["semantic_score"]
    return best_result


# --- API Endpoint (Aangepast voor nieuwe fallback) ---
@app.route('/suggesties', methods=['POST']) # Of /zoek
def suggesties_route():
    if get_embedding_function is None or not uitspraken:
         return jsonify({"error": "Service is tijdelijk niet volledig beschikbaar (model/data probleem)"}), 503

    data = request.json or {}
    query_text = data.get("query_text")
    if not query_text:
        return jsonify({"error": "query_text ontbreekt"}), 400

    # Genereer embedding voor de query
    try:
        print(f"⏳ Genereren embedding voor: \"{query_text[:100]}...\"")
        embedding_list = get_embedding_function(query_text)
        q_vector = np.array(embedding_list, dtype=np.float32)
        print("✅ Embedding gegenereerd.")
    except Exception as e:
         print(f"❌ Fout bij genereren embedding: {e}")
         return jsonify({"error": "Kon de zoekvraag niet verwerken (embedding error)"}), 500

    # Primaire zoektocht via embeddings in DB
    try:
        scored = [(cosine_similarity(q_vector, vec), ecli, titel, samenv, link)
                  for ecli, titel, samenv, link, vec in uitspraken]
        top5 = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
    except Exception as e:
        print(f"❌ Fout bij berekenen similariteit: {e}")
        return jsonify({"error": "Kon de zoekvraag niet verwerken (similarity error)"}), 500

    # Bepaal of fallback nodig is
    use_fallback = not top5 or top5[0][0] < SIMILARITY_THRESHOLD

    if use_fallback:
        print(f"⚠️ Primaire zoektocht onvoldoende (hoogste score: {top5[0][0] if top5 else 'N/A'} < {SIMILARITY_THRESHOLD}). Start verbeterde fallback...")
        # Roep de nieuwe fallback functie aan, geef de query embedding mee!
        fallback_result = get_best_fallback_result(searchterm=query_text, query_embedding_vector=q_vector)

        if fallback_result:
            # We hebben nu 1 beste fallback resultaat na re-ranking
            return jsonify({"fallback": True, **fallback_result}), 200
        else:
            print("❌ Fallback heeft geen resultaten opgeleverd.")
            return jsonify({"fallback": True, "message": "Geen relevante jurisprudentie gevonden, ook niet via fallback."}), 404
    else:
        # Geef normale resultaten terug (top 5 uit primaire zoektocht)
        print(f"✅ Top {len(top5)} resultaten gevonden via embeddings (DB).")
        results = [{"ecli": e, "titel": t, "samenvatting": s, "link": l, "score": round(score, 4)}
                   for score, e, t, s, l in top5]
        return jsonify({"fallback": False, "results": results}), 200


# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    status_model = "OK" if get_embedding_function else "ERROR"
    status_db = f"{len(uitspraken)} records" if uitspraken else "ERROR"
    return f"Jurisprudentie-zoeker (v3 - text input, semantic fallback) draait!\nModel: {status_model}, DB: {status_db}", 200

# Server starten
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"⚙️ Starting server on 0.0.0.0:{port}")
    # Gebruik debug=False voor productie
    app.run(host='0.0.0.0', port=port, debug=True)
