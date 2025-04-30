# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import sqlite3
import numpy as np
from numpy.linalg import norm
import os
import requests
import threading # Toegevoegd voor eventuele toekomstige locking, nu niet strikt nodig maar kan handig zijn

# Importeer de functie uit bert_loader.py
from bert_loader import load_legalbert_embedding_pipeline # [cite: 2]

app = Flask(__name__)

# --- Configuratie ---
API_INDEX_URL    = "https://api.rechtspraak.nl/v1/index" # [cite: 1]
API_DOCUMENT_URL = "https://api.rechtspraak.nl/v1/document" # [cite: 1]
RECHTSGEBIED_URI = "http://psi.rechtspraak.nl/rechtsgebied#bestuursrecht_omgevingsrecht" # [cite: 1]
DB_PATH          = "jurisprudentie.db" # [cite: 1]
FALLBACK_MAX_RECORDS = 5 # Aantal resultaten voor fallback check
SIMILARITY_THRESHOLD = 0.6 # Drempelwaarde voor primaire zoekresultaten

# --- Initialisatie (bij opstarten server) ---

print("🚀 Initialiseren Legal BERT embedding model...")
try:
    # load_legalbert_embedding_pipeline() retourneert de 'embed' functie [cite: 2]
    get_embedding_function = load_legalbert_embedding_pipeline() # [cite: 2]
    print("✅ Legal BERT model geladen en embedding functie klaar.")
except Exception as e:
    print(f"❌ Fout bij laden Legal BERT model: {e}")
    get_embedding_function = None # Markeer dat laden mislukt is

def load_embeddings(db_path=DB_PATH): # [cite: 1]
    """Laadt ECLI, metadata en pre-computed embeddings uit de database."""
    if not os.path.exists(db_path):
        print(f"❌ Fout: Databasebestand niet gevonden op {db_path}")
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Haalt data op uit jurisprudentie.db [cite: 1]
        cursor.execute("SELECT ecli, titel, samenvatting, link, embedding FROM uitspraken")
        data = [(ecli, titel, samenv, link, np.frombuffer(blob, dtype=np.float32))
                for ecli, titel, samenv, link, blob in cursor.fetchall()]
        conn.close()
        print(f"✅ {len(data)} uitspraken met embeddings geladen uit {db_path}.")
        return data
    except Exception as e:
        print(f"❌ Fout bij laden embeddings uit database: {e}")
        return []

# Laadt de database embeddings [cite: 1]
uitspraken = load_embeddings() # [cite: 1]

# --- Hulpfuncties ---

def cosine_similarity(a, b): # [cite: 1]
    """Berekent de cosine similarity tussen twee numpy arrays."""
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0 # Geen similariteit als een vector nul is
    return float(np.dot(a, b) / (norm_a * norm_b))

# --- Fallback Functie (Aangepast met Semantische Re-ranking) ---
def get_best_fallback_result(searchterm=None, query_embedding_vector=None):
    """
    Haalt meerdere fallback kandidaten op via Open Data API en selecteert
    de meest semantisch relevante op basis van de query embedding.
    """
    # Controleer of alle benodigde input aanwezig is
    if not searchterm or query_embedding_vector is None or get_embedding_function is None:
        print("ℹ️ Fallback: Ontbrekende searchterm, query embedding of embedding functie.")
        return None

    print(f"⏳ Start fallback zoekopdracht voor '{searchterm}' met max {FALLBACK_MAX_RECORDS} records...")
    # Criteria voor de API index call [cite: 1]
    criteria = {"rechtsgebied": [RECHTSGEBIED_URI], "searchterm": searchterm}
    # Vraag meerdere records op, alleen ECLI's eerst
    payload = {"criteria": criteria, "maxRecords": FALLBACK_MAX_RECORDS, "return": ["ecli"]}
    candidate_results = []

    try:
        # Stap 1: Haal ECLI's op van potentiële kandidaten via index API [cite: 1]
        idx_resp = requests.post(API_INDEX_URL, json=payload, timeout=15)
        idx_resp.raise_for_status() # Check voor HTTP errors
        idx_data = idx_resp.json()
        eclis = [res["ecli"] for res in idx_data.get("results", []) if "ecli" in res]

        if not eclis:
            print(f"ℹ️ Fallback: Geen index resultaten gevonden voor '{searchterm}'")
            return None

        print(f"ℹ️ Fallback: {len(eclis)} kandidaat ECLI's gevonden. Details ophalen...")

        # Stap 2: Haal details en genereer embeddings voor elke kandidaat
        for ecli in eclis:
            try:
                # Haal document details op via document API [cite: 1]
                doc_resp = requests.get(f"{API_DOCUMENT_URL}?ecli={ecli}", timeout=10)
                doc_resp.raise_for_status()
                doc = doc_resp.json()

                # Pak samenvatting of (deel van) berichttekst [cite: 1]
                summary_text = doc.get("summary", "") or doc.get("berichttekst", "")
                if not summary_text:
                    print(f"⚠️ Fallback: Geen samenvatting/tekst gevonden voor {ecli}")
                    continue # Sla deze kandidaat over

                # Genereer embedding voor de samenvatting/tekst van de kandidaat [cite: 2]
                # Beperk lengte voor embedding; voorkomt problemen met te lange teksten
                fallback_embedding_list = get_embedding_function(summary_text[:1024])
                fallback_vector = np.array(fallback_embedding_list, dtype=np.float32)

                # Bereken similariteit met de originele query embedding [cite: 1]
                semantic_score = cosine_similarity(query_embedding_vector, fallback_vector)

                candidate_results.append({
                    "ecli": doc.get("ecli"),
                    "titel": doc.get("title"),
                    # Neem eerste 500 tekens voor weergave [cite: 1]
                    "samenvatting": summary_text[:500] + "...",
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

    # Zoek het resultaat met de hoogste semantische score
    best_result = max(candidate_results, key=lambda x: x["semantic_score"])
    print(f"✅ Beste fallback resultaat geselecteerd (semantisch): {best_result['ecli']} (Score: {best_result['semantic_score']:.4f})")

    # Verwijder de interne score voor het eindresultaat dat teruggestuurd wordt
    del best_result["semantic_score"]
    return best_result


# --- API Endpoint (Aangepast voor text input en nieuwe fallback) ---
@app.route('/suggesties', methods=['POST']) # Je kunt dit hernoemen naar /zoek als je wilt
def suggesties_route():
    # Check of model en database correct geladen zijn
    if get_embedding_function is None or not uitspraken:
         return jsonify({"error": "Service is tijdelijk niet volledig beschikbaar (model/data probleem)"}), 503

    data = request.json or {}
    query_text = data.get("query_text") # Lees query_text ipv embedding
    if not query_text:
        return jsonify({"error": "query_text ontbreekt in request body"}), 400

    # Genereer embedding voor de query HIER in de server [cite: 2]
    try:
        print(f"⏳ Genereren embedding voor: \"{query_text[:100]}...\"")
        embedding_list = get_embedding_function(query_text)
        q_vector = np.array(embedding_list, dtype=np.float32)
        print("✅ Embedding gegenereerd.")
    except Exception as e:
         print(f"❌ Fout bij genereren embedding: {e}")
         return jsonify({"error": "Kon de zoekvraag niet verwerken (embedding error)"}), 500

    # Primaire zoektocht via embeddings in DB [cite: 1]
    try:
        # Vergelijk query embedding met database embeddings [cite: 1]
        scored = [(cosine_similarity(q_vector, vec), ecli, titel, samenv, link)
                  for ecli, titel, samenv, link, vec in uitspraken]
        # Sorteer en neem top 5 [cite: 1]
        top5 = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
    except Exception as e:
        print(f"❌ Fout bij berekenen similariteit: {e}")
        return jsonify({"error": "Kon de zoekvraag niet verwerken (similarity error)"}), 500

    # Bepaal of fallback nodig is gebaseerd op drempelwaarde
    use_fallback = not top5 or top5[0][0] < SIMILARITY_THRESHOLD

    if use_fallback:
        print(f"⚠️ Primaire zoektocht onvoldoende (hoogste score: {top5[0][0] if top5 else 'N/A'} < {SIMILARITY_THRESHOLD}). Start verbeterde fallback...")
        # Roep de nieuwe fallback functie aan, geef de query embedding mee!
        fallback_result = get_best_fallback_result(searchterm=query_text, query_embedding_vector=q_vector)

        if fallback_result:
            # We hebben nu 1 beste fallback resultaat na re-ranking
            print("✅ Fallback resultaat gevonden via semantische re-ranking.")
            return jsonify({"fallback": True, **fallback_result}), 200
        else:
            print("❌ Fallback heeft ook geen resultaten opgeleverd.")
            return jsonify({"fallback": True, "message": "Geen relevante jurisprudentie gevonden, ook niet via fallback."}), 404 # Not Found
    else:
        # Geef normale resultaten terug (top 5 uit primaire zoektocht)
        print(f"✅ Top {len(top5)} resultaten gevonden via embeddings (DB).")
        results = [{"ecli": e, "titel": t, "samenvatting": s, "link": l, "score": round(score, 4)}
                   for score, e, t, s, l in top5]
        return jsonify({"fallback": False, "results": results}), 200


# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    # Geef status van model en database aan
    status_model = "OK" if get_embedding_function else "ERROR (Niet geladen)"
    status_db = f"{len(uitspraken)} records geladen" if uitspraken else "ERROR (Niet geladen / Leeg)"
    return f"Jurisprudentie-zoeker (v3 - text input, semantic fallback) draait!\nModel: {status_model}, DB: {status_db}", 200

# Server starten
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"⚙️ Starting server on 0.0.0.0:{port}")
    # Zet debug=False voor productieomgevingen!
    app.run(host='0.0.0.0', port=port, debug=True) # debug=True is handig tijdens ontwikkelen
