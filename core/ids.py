"""ULID generation — local stand-in for the shared ``tabi_core.ids`` package.

The production monorepo ships a small ``tabi-core`` package with ULID helpers
used for correlation IDs across services. For this self-contained public repo we
inline a minimal, dependency-light generator with the same call surface
(``generate_ulid() -> str``).
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_ulid() -> str:
    """Return a 26-char Crockford-base32 ULID (48-bit time + 80-bit randomness).

    Not cryptographically rigorous — a readable, monotonic-ish identifier good
    enough for correlation/turn IDs in the showcase.
    """
    ts = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ts << 80) | rand
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


__all__ = ["generate_ulid"]
