from flask import Flask, request, jsonify
import sqlite3
import numpy as np
from numpy.linalg import norm
import os
import requests # Importeer requests hier bovenaan

# Importeer de functie uit bert_loader.py
from bert_loader import load_legalbert_embedding_pipeline #

app = Flask(__name__)

# --- Configuratie ---
API_INDEX_URL    = "https://api.rechtspraak.nl/v1/index"
API_DOCUMENT_URL = "https://api.rechtspraak.nl/v1/document"
RECHTSGEBIED_URI = "http://psi.rechtspraak.nl/rechtsgebied#bestuursrecht_omgevingsrecht"
DB_PATH          = "jurisprudentie.db"

# --- Initialisatie (bij opstarten server) ---

# Laad de embedding functie EENMAAL bij opstart
print("🚀 Initialiseren Legal BERT embedding model...")
try:
    # load_legalbert_embedding_pipeline() retourneert de 'embed' functie
    get_embedding_function = load_legalbert_embedding_pipeline() #
    print("✅ Legal BERT model geladen en embedding functie klaar.")
except Exception as e:
    print(f"❌ Fout bij laden Legal BERT model: {e}")
    # Optioneel: stop de server als het model niet laadt
    # raise RuntimeError("Kon Legal BERT model niet laden.") from e
    get_embedding_function = None # Zorg dat we weten dat het mislukt is

# Laad de jurisprudentie embeddings uit de database
def load_embeddings(db_path=DB_PATH):
    """Laadt ECLI, metadata en pre-computed embeddings uit de database."""
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
    # Voeg check toe voor zero norm om deling door nul te voorkomen
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0 # Geen similariteit als een vector nul is
    return float(np.dot(a, b) / (norm_a * norm_b))

def fallback_analysis(searchterm=None):
    """Haalt één relevante ECLI met samenvatting via Open Data API."""
    if not searchterm:
        return None # Geen zoekterm, geen fallback mogelijk
    criteria = {"rechtsgebied": [RECHTSGEBIED_URI], "searchterm": searchterm}
    payload = {"criteria": criteria, "maxRecords": 1}
    try:
        idx_resp = requests.post(API_INDEX_URL, json=payload, timeout=10) # Timeout toegevoegd
        idx_resp.raise_for_status() # Check voor HTTP errors
        idx_data = idx_resp.json()
        if not idx_data.get("results"):
            print(f"ℹ️ Fallback: Geen index resultaten voor '{searchterm}'")
            return None

        ecli = idx_data["results"][0]["ecli"]
        doc_resp = requests.get(f"{API_DOCUMENT_URL}?ecli={ecli}", timeout=10) # Timeout toegevoegd
        doc_resp.raise_for_status() # Check voor HTTP errors
        doc = doc_resp.json()

        # Haal samenvatting of berichttekst op en trim
        summary = (doc.get("summary", "") or doc.get("berichttekst", ""))[:500] + "..." # Iets langer gemaakt

        return {
            "ecli": doc.get("ecli"),
            "titel": doc.get("title"),
            "samenvatting": summary,
            "link": doc.get("documentUrl")
        }
    except requests.exceptions.RequestException as e:
        print(f"❌ Fout tijdens fallback API call: {e}")
        return None
    except Exception as e:
        print(f"❌ Onverwachte fout tijdens fallback: {e}")
        return None


# --- API Endpoint ---

# Endpoint aangepast om query_text te accepteren ipv embedding
# Eventueel hernoemen naar /zoek voor duidelijkheid? Ik laat het nu /suggesties.
@app.route('/suggesties', methods=['POST'])
def suggesties_route():
    # Check of het embedding model correct geladen is
    if get_embedding_function is None:
         return jsonify({"error": "Embedding model is niet beschikbaar"}), 503 # Service Unavailable

    # Check of er überhaupt embeddings geladen zijn uit de DB
    if not uitspraken:
         return jsonify({"error": "Jurisprudentie data is niet beschikbaar"}), 503 # Service Unavailable

    data = request.json or {}
    query_text = data.get("query_text") # <<< WIJZIGING: Lees query_text ipv embedding

    if not query_text:
        return jsonify({"error": "query_text ontbreekt in request body"}), 400 # Duidelijkere foutmelding

    # --- Genereer embedding HIER ---
    try:
        print(f"⏳ Genereren embedding voor: \"{query_text[:100]}...\"")
        embedding_list = get_embedding_function(query_text) # Roep de functie aan
        q_vector = np.array(embedding_list, dtype=np.float32) # Converteer naar numpy array
        print("✅ Embedding gegenereerd.")
    except Exception as e:
         print(f"❌ Fout bij genereren embedding voor query: {e}")
         # Geef een interne server error terug, niet de details lekken
         return jsonify({"error": "Kon de zoekvraag niet verwerken (embedding error)"}), 500
    # --- Einde embedding generatie ---

    # Bereken scores (deze logica blijft grotendeels hetzelfde)
    try:
        scored = [(cosine_similarity(q_vector, vec), ecli, titel, samenv, link)
                  for ecli, titel, samenv, link, vec in uitspraken]
        # Sorteer en neem top 5
        top5 = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
    except Exception as e:
        print(f"❌ Fout bij berekenen similariteit: {e}")
        return jsonify({"error": "Kon de zoekvraag niet verwerken (similarity error)"}), 500


    # Fallback logica (gebruik query_text als searchterm)
    # Stel een drempelwaarde in voor wanneer de match goed genoeg is
    SIMILARITY_THRESHOLD = 0.6 # Voorbeeld, pas deze waarde aan naar wens!
    use_fallback = not top5 or top5[0][0] < SIMILARITY_THRESHOLD

    if use_fallback:
        print(f"⚠️ Geen goede match gevonden (hoogste score: {top5[0][0] if top5 else 'N/A'} < {SIMILARITY_THRESHOLD}), fallback proberen...")
        fallback_result = fallback_analysis(query_text) # <<< WIJZIGING: Geef query_text mee
        if fallback_result:
            print("✅ Fallback resultaat gevonden.")
            return jsonify({"fallback": True, **fallback_result}), 200
        else:
            print("❌ Fallback heeft ook geen resultaten opgeleverd.")
            # Stuur nu een duidelijke 'niets gevonden' response ipv een error
            return jsonify({"fallback": True, "message": "Geen relevante jurisprudentie gevonden, ook niet via fallback."}), 404 # Not Found

    # Geef normale resultaten terug
    print(f"✅ Top {len(top5)} resultaten gevonden via embeddings.")
    results = [{"ecli": e, "titel": t, "samenvatting": s, "link": l, "score": round(score, 4)}
               for score, e, t, s, l in top5]
    return jsonify({"fallback": False, "results": results}), 200

# Health check endpoint (onveranderd)
@app.route('/', methods=['GET'])
def health_check():
    return "Jurisprudentie-zoeker (v2 - text input) draait!", 200

# Server starten (onveranderd)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    # Zorg dat de app niet in debug mode draait in productie!
    # app.run(host='0.0.0.0', port=port, debug=False) # Voor productie
    print(f"⚙️ Starting server on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True) # Debug=True is handig tijdens ontwikkelen
