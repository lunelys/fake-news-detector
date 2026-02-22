import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

IDENTIFIER = os.getenv("BSKY_IDENTIFIER")
PASSWORD = os.getenv("BSKY_PASSWORD")
URL = "https://bsky.social/xrpc/com.atproto.server.createSession"


def login(identifier: str, password: str, timeout: int = 10):
    if not identifier or not password:
        print("Erreur: identifiant ou mot de passe manquant dans .env")
        return False

    payload = {"identifier": identifier, "password": password}
    try:
        r = requests.post(URL, json=payload, timeout=timeout)
        print("HTTP status:", r.status_code)
        if r.status_code != 200:
            print("Erreur connexion :", r.text)
            return False

        data = r.json()
        access = data.get("accessJwt")
        refresh = data.get("refreshJwt")
        if access:
            with open("token.json", "w", encoding="utf-8") as f:
                json.dump({"accessJwt": access, "refreshJwt": refresh}, f, ensure_ascii=False, indent=2)
            print("Login OK — tokens sauvegardés dans token.json")
            return True
        else:
            print("Login échoué — aucun accessJwt reçu")
            return False

    except requests.exceptions.Timeout:
        print("Erreur : timeout")
    except requests.exceptions.ConnectionError:
        print("Erreur : échec connexion réseau")
    except Exception as e:
        print("Erreur inattendue :", e)

    return False


if __name__ == "__main__":
    login(IDENTIFIER, PASSWORD)