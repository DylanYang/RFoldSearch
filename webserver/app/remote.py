"""Databases that another RFoldSearch server holds.

RFOLDSEARCH_REMOTE_DBS="key=http://host:port|key2=..." makes this server offer
database ``key`` although it does not hold it: a search of it is forwarded, as
the original upload, to the server at that address, and the remote hits are
stored in this server's own result record. The alignment and structure of a
remote hit are fetched from the remote record on demand. A database whose
server cannot be reached is simply not offered.
"""
from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.request
from typing import Optional

REMOTE: dict = {}
for _item in os.environ.get("RFOLDSEARCH_REMOTE_DBS", "").split("|"):
    _key, _sep, _url = _item.strip().partition("=")
    if _sep and _key.strip() and _url.strip():
        REMOTE[_key.strip()] = _url.strip().rstrip("/")

TIMEOUT = float(os.environ.get("RFOLDSEARCH_REMOTE_TIMEOUT", "600"))
_INFO_TTL = 30.0
_info_cache: dict = {}


class Offline(RuntimeError):
    """The remote server did not answer."""


def _open(req, timeout):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:      # the remote answered with an error
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        raise RemoteError(exc.code, str(detail)) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise Offline(f"{type(exc).__name__}: {exc}") from exc


class RemoteError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def info(key: str) -> Optional[dict]:
    """The remote server's description of database ``key``, or None when it is not served."""
    now = time.time()
    hit = _info_cache.get(key)
    if hit and now - hit[0] < _INFO_TTL:
        return hit[1]
    value = None
    try:
        with _open(urllib.request.Request(REMOTE[key] + "/api/health"), 5) as resp:
            health = json.loads(resp.read())
        if health.get("ready"):
            value = next((d for d in health["databases"] if d["key"] == key), None)
    except (Offline, RemoteError, ValueError):
        value = None
    _info_cache[key] = (now, value)
    return value


def search(key: str, data: bytes, name: str, top_k: int, top_n: int,
           min_diag: int, rerank: str, tie_seed: Optional[int]) -> dict:
    """Forward one search; returns the remote result for ``key`` plus the remote job id."""
    boundary = "----rfs" + secrets.token_hex(8)
    fields = {"databases": key, "top_k": top_k, "top_n": top_n,
              "min_diag": min_diag, "rerank": rerank}
    if tie_seed is not None:
        fields["tie_seed"] = tie_seed
    body = b""
    for k, v in fields.items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n').encode()
    safe = name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="structure"; '
             f'filename="{safe}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(REMOTE[key] + "/api/search", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with _open(req, TIMEOUT) as resp:
        job = json.loads(resp.read())
    res = job["results"][key]
    res["remote_job"] = job["id"]
    return res


def fetch(key: str, remote_job: str, rank: int, what: str) -> bytes:
    """Alignment (JSON) or structure (PDB text) of a hit of a remote result."""
    url = f"{REMOTE[key]}/api/result/{remote_job}/{key}/{rank}/{what}"
    with _open(urllib.request.Request(url), 120) as resp:
        return resp.read()
