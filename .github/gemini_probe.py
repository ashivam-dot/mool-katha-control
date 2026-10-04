"""Print which Gemini models the QA key can call now: status and quota names only, never the key."""

import json
import os
import urllib.error
import urllib.request

KEY = os.environ["QA_GEMINI_API_KEY"]
BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
          "gemini-3-flash-preview", "gemini-flash-latest", "gemini-3.5-flash-lite",
          "gemini-flash-lite-latest", "gemini-3.8-pro", "gemini-3-pro-preview", "gemini-2.5-flash",
          "gemini-2.5-pro"]


def call(request):
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except ValueError:
            return exc.code, {}


status, listing = call(urllib.request.Request(f"{BASE}?pageSize=200", headers={"x-goog-api-key": KEY}))
names = sorted(m["name"].split("/", 1)[1] for m in listing.get("models", [])
               if "generateContent" in m.get("supportedGenerationMethods", []) and "gemini" in m["name"])
print("listed", status, names)
body = json.dumps({"contents": [{"role": "user", "parts": [{"text": "Reply with OK."}]}],
                   "generationConfig": {"maxOutputTokens": 8}}).encode()
for model in MODELS:
    request = urllib.request.Request(f"{BASE}/{model}:generateContent", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
    status, answer = call(request)
    error = answer.get("error", {}) if isinstance(answer, dict) else {}
    quotas = [v.get("quotaId") for d in error.get("details", []) for v in d.get("violations", []) if isinstance(v, dict)]
    delays = [d.get("retryDelay") for d in error.get("details", []) if "retryDelay" in d]
    print(model, status, error.get("status", ""), quotas, delays, (error.get("message") or "")[:160])
