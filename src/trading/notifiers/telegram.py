"""
Telegram notifier using stdlib urllib only — no requests/httpx dependency.

Handles Telegram's 4096-character message limit by splitting long messages
into chunks at newline boundaries.
"""

import json
import logging
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_CHARS = 4096
_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """
    Sends plain-text messages to a Telegram chat.

    Args:
        token:   Telegram Bot API token (from BotFather).
        chat_id: Target chat or user ID.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, token: str, chat_id: str, timeout: int = 15) -> None:
        self._url = _API_BASE.format(token=token)
        self._chat_id = chat_id
        self._timeout = timeout

    def send(self, text: str) -> None:
        """Send a single message. Raises if len(text) > 4096 or on HTTP error."""
        if len(text) > _TELEGRAM_MAX_CHARS:
            raise ValueError(
                f"Message length {len(text)} exceeds Telegram limit {_TELEGRAM_MAX_CHARS}. "
                "Use send_chunked() instead."
            )
        self._post(text)

    def send_chunked(self, text: str) -> None:
        """
        Split text into ≤4096-char chunks at newline boundaries and send each.

        Each chunk is sent as a separate message. A single line longer than
        4096 chars is truncated with a trailing ellipsis.
        """
        chunks = _split_into_chunks(text, _TELEGRAM_MAX_CHARS)
        for i, chunk in enumerate(chunks, 1):
            logger.info(
                "Sending Telegram chunk %d/%d (%d chars)", i, len(chunks), len(chunk)
            )
            self._post(chunk)

    def _post(self, text: str) -> None:
        payload = urllib.parse.urlencode(
            {
                "chat_id": self._chat_id,
                "text": text,
                # No parse_mode — OHLCV tables contain chars that break Markdown
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self._url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read()
                result = json.loads(body)
                if not result.get("ok"):
                    raise RuntimeError(f"Telegram API returned ok=false: {result}")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Telegram HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Telegram connection error: {exc.reason}") from exc


def _split_into_chunks(text: str, max_len: int) -> list[str]:
    """
    Split text into chunks of at most max_len characters,
    breaking only at newline boundaries.
    """
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if len(line) > max_len:
            line = line[: max_len - 4] + "...\n"

        if current_len + len(line) > max_len:
            if current:
                chunks.append("".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)

    if current:
        chunks.append("".join(current))

    return chunks
