# Jurisprudentiezoeker - Render Deployment

## Deploy instructies

1. Maak een nieuwe GitHub repository aan (bijvoorbeeld `jurisprudentie-zoeker`)
2. Upload de inhoud van deze map (`render_app/`) naar je repository
3. Voeg je `jurisprudentie.db` toe in de `render_app/` map
4. Ga naar [https://dashboard.render.com](https://dashboard.render.com)
5. Maak een nieuwe **Web Service** aan
   - Selecteer jouw GitHub repository
   - Kies `render_app` als root map
   - Port: 8080
6. Klik Deploy en klaar! 🚀

Je `/suggesties` endpoint is daarna beschikbaar om je Custom GPT te gebruiken.
