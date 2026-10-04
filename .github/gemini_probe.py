"""Print which Gemini models the QA key can call with audio now: status and quota names only, never the key."""

import base64
import io
import json
import math
import os
import struct
import urllib.error
import urllib.request
import wave

KEY = os.environ["QA_GEMINI_API_KEY"]
BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-flash-lite-latest",
          "gemini-3.1-flash-lite", "gemini-3.8-flash"]


def call(request):
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except ValueError:
            return exc.code, {}


buffer = io.BytesIO()
with wave.open(buffer, "wb") as out:
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(16000)
    out.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / 16000)))
                             for i in range(16000 * 20)))
audio = base64.b64encode(buffer.getvalue()).decode()
for kind, parts in (("text", [{"text": "Reply with OK."}]),
                    ("audio", [{"text": "Describe this audio in three words."},
                               {"inlineData": {"mimeType": "audio/wav", "data": audio}}])):
    body = json.dumps({"contents": [{"role": "user", "parts": parts}],
                       "generationConfig": {"maxOutputTokens": 16}}).encode()
    for model in MODELS:
        request = urllib.request.Request(f"{BASE}/{model}:generateContent", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
        status, answer = call(request)
        error = answer.get("error", {}) if isinstance(answer, dict) else {}
        quotas = [(v.get("quotaId"), v.get("quotaValue")) for d in error.get("details", [])
                  for v in d.get("violations", []) if isinstance(v, dict)]
        print(kind, model, status, error.get("status", ""), quotas)
