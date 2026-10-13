"""Print only the OAuth scope names granted to the control repo's Google token (never the token itself)."""

import json
import os
import urllib.parse
import urllib.request


def main() -> None:
    try:
        client = json.loads(os.environ.get("YTC_GOOGLE_CLIENT") or "{}")
        saved = json.loads(os.environ.get("YTC_GOOGLE_TOKEN") or "{}")
    except ValueError:
        print("token JSON unreadable")
        return
    print("token fields:", sorted(saved))
    if isinstance(saved.get("scopes"), list):
        print("saved scopes:", sorted(saved["scopes"]))
    app = client.get("installed") or client.get("web") or {}
    try:
        data = urllib.parse.urlencode({"client_id": app["client_id"], "client_secret": app["client_secret"],
                                       "refresh_token": saved["refresh_token"], "grant_type": "refresh_token"}).encode()
        with urllib.request.urlopen(urllib.request.Request("https://oauth2.googleapis.com/token", data=data), timeout=25) as r:
            access = json.loads(r.read()).get("access_token")
        query = urllib.parse.urlencode({"access_token": access})
        with urllib.request.urlopen(f"https://oauth2.googleapis.com/tokeninfo?{query}", timeout=25) as r:
            info = json.loads(r.read())
        print("live scopes:", sorted(str(info.get("scope", "")).split()))
    except Exception as err:  # noqa: BLE001 - report the failure kind only, never a value
        print("live scope check failed:", type(err).__name__)


if __name__ == "__main__":
    main()
