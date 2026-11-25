from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

from scam_ai import predict_text


class handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 200_000:
                self._respond(413, {"error": "Мәтін көлемі тым үлкен."})
                return
            payload = json.loads(self.rfile.read(length))
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip():
                self._respond(400, {"error": "Талдау үшін диалог мәтінін енгізіңіз."})
                return
            if len(text) > 20_000:
                self._respond(413, {"error": "Мәтін 20 000 таңбадан аспауы керек."})
                return
            self._respond(200, predict_text(text))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._respond(400, {"error": "Сұрау пішімі дұрыс емес."})
        except FileNotFoundError:
            self._respond(503, {"error": "Модель файлы табылмады. Жоба жарияланымын тексеріңіз."})
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
        except Exception:
            self._respond(500, {"error": "Талдау кезінде ішкі қате пайда болды."})

    def log_message(self, format: str, *args: object) -> None:
        # Avoid writing request details to function logs.
        return