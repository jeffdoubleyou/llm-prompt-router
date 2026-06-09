from __future__ import annotations

import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.core.config import settings

logger = logging.getLogger(__name__)

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is not None:
        return _fernet
    raw_key = settings.encryption_key
    if not raw_key:
        raise RuntimeError("ENCRYPTION_KEY is not set — cannot encrypt keys")
    key_bytes = raw_key.encode() if isinstance(raw_key, str) else raw_key
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b"llm-router-salt", iterations=600_000)
    derived = base64.urlsafe_b64encode(kdf.derive(key_bytes))
    _fernet = Fernet(derived)
    return _fernet


def encrypt_api_key(plaintext: str) -> str:
    f = _get_fernet()
    return f.encrypt(plaintext.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    f = _get_fernet()
    return f.decrypt(ciphertext.encode()).decode()
