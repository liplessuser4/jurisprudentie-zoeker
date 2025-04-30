import requests
import json
import os # Toegevoegd voor file path

# Zorg ervoor dat bert_loader.py in hetzelfde pad staat of via PYTHONPATH bereikbaar is
from bert_loader import load_legalbert_embedding_pipeline #

# --- Configuratie ---
# Pas aan als je server elders draait of op een andere poort
FLASK_SERVER_URL = "http://127.0.0.1:8080/suggesties"

# --- Initialisatie (doe dit één keer bij opstarten client applicatie) ---
print("🚀 Initialiseren Legal BERT embedding model...")
# Laad de functie die embeddings kan genereren
try:
    # load_legalbert_embedding_pipeline() retourneert de 'embed' functie
    get_embedding_function = load_legalbert_embedding_pipeline()
    print("✅ Legal BERT model geladen en embedding functie klaar.")

except ImportError:
    print(f"❌ Fout: Kon bert_loader niet importeren. Zorg dat bert_loader.py bestaat in {os.getcwd()} of in je PYTHONPATH.")
    exit()
except Exception as e:
    print(f"❌ Fout bij laden Legal BERT model: {e}")
    # Toon meer details bij Hugging Face / network errors
    if "offline" in str(e).lower() or "connection" in str(e).lower():
         print("ℹ️ Tip: Controleer je internetverbinding. Het model moet mogelijk gedownload worden.")
    exit()


# --- Functie om suggesties op te halen ---
def get_jurisprudentie_suggesties(query_text: str):
    """
    Vraagt suggesties op bij de Flask server voor een gegeven query.

    Args:
        query_text: De zoekvraag van de gebruiker (string).

    Returns:
        Een dictionary met de resultaten van de server, of None bij een fout.
    """
    print(f"\n🔍 Zoekvraag: \"{query_text}\"")

    # 1. Genereer de embedding voor de zoekvraag met JOUW Legal BERT functie
    try:
        print("⏳ Genereren van Legal BERT embedding via bert_loader...")
        # Roep de verkregen 'embed' functie aan
        query_embedding_list = get_embedding_function(query_text)
        # De functie retourneert al een list[float], dus geen conversie nodig
        print(f"✅ Embedding gegenereerd (vector lengte: {len(query_embedding_list)}).")

    except Exception as e:
        print(f"❌ Fout tijdens genereren embedding: {e}")
        return None

    # 2. Roep de Flask API aan met de embedding
    # De payload verwacht een 'embedding' key
    payload = {
        "embedding": query_embedding_list,
        "searchterm": query_text # Stuur de originele term mee voor fallback
    }
    headers = {'Content-Type': 'application/json'}

    print(f"📡 Aanroepen Flask server op {FLASK_SERVER_URL}...")
    try:
        response = requests.post(FLASK_SERVER_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Genereert een error bij status codes 4xx/5xx
        print(f"✅ Server response status: {response.status_code}")
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Fout bij aanroepen Flask server: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"   Server response: {e.response.status_code} - {e.response.text}")
        return None
    except json.JSONDecodeError:
        print("❌ Fout: Kon JSON response van server niet parsen.")
        print("Raw response:", response.text)
        return None


# --- Voorbeeld Gebruik ---
if __name__ == "__main__":
    # Zorg dat je Flask server (flask_server (2).py) draait!

    # Voorbeeld zoekvraag
    gebruikers_vraag = "Wat zijn de regels omtrent geluidsoverlast door horeca?"

    # Haal suggesties op
    resultaten = get_jurisprudentie_suggesties(gebruikers_vraag)

    # Verwerk de resultaten
    if resultaten:
        print("\n--- Resultaten van Server ---")
        # Check of de server een fallback resultaat heeft gestuurd
        if resultaten.get("fallback"):
            print("⚠️ Fallback resultaat gebruikt:")
            if "error" in resultaten:
                print(f"   Fout: {resultaten['error']}")
            else:
                print(f"   ECLI: {resultaten.get('ecli', 'N/A')}")
                print(f"   Titel: {resultaten.get('titel', 'N/A')}")
                print(f"   Samenvatting: {resultaten.get('samenvatting', 'N/A')}")
                print(f"   Link: {resultaten.get('link', 'N/A')}")
        # Verwerk de normale resultaten
        elif "results" in resultaten:
            print("✅ Top suggesties gevonden:")
            for i, res in enumerate(resultaten.get("results", [])):
                print(f"  {i+1}. ECLI: {res['ecli']} (Score: {res['score']})")
                print(f"     Titel: {res['titel']}")
                # print(f"     Samenvatting: {res['samenvatting'][:100]}...") # Optioneel
                print(f"     Link: {res['link']}")
        else:
             print("❓ Onbekende response structuur:", resultaten)

    else:
        print("\n❌ Kon geen suggesties ophalen.")
