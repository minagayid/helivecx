"""helivecx/indexed.py

Indexed-byte encoding for HelivecX.

Each byte carries a type flag in its least-significant bit:
    bit 0 = 0  →  NUMERIC  : bits 1-7 encode a value 0-127 (7-bit unsigned int)
    bit 0 = 1  →  ALPHA    : bits 1-7 encode a letter index 0-127 (A-Z = 0-25,
                              a-z = 26-51, digits = 52-61, special = 62-127)

This doubles the semantic density per byte: instead of a flat 0-255 range,
each byte is a tagged union that carries both *what kind* of value it holds
and *the value itself* in a single octet — similar to how base pairs in DNA
encode both structure and function simultaneously.

Encoding layout per byte:
    [b7 b6 b5 b4 b3 b2 b1 b0]
     |--- value (7 bits) ---| type flag (1 bit)

Typical usage:
    from helivecx.indexed import encode_indexed, decode_indexed
    data = encode_indexed(values, types)   # → bytes
    values, types = decode_indexed(data)   # → (list[int], list[str])
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Alphabet maps – index 0-127 maps to meaningful letter codes
# ---------------------------------------------------------------------------


def _build_alpha_table() -> list[str]:
    """Build the 128-entry alpha lookup table programmatically.

    Layout:
        0-25   A-Z
        26-51  a-z
        52-61  0-9
        62-93  common programming / structural symbols
        94-127 remaining printable ASCII (ascending codepoint)
    """
    table: list[str] = []
    table += [chr(c) for c in range(ord("A"), ord("Z") + 1)]    # 0-25
    table += [chr(c) for c in range(ord("a"), ord("z") + 1)]    # 26-51
    table += [chr(c) for c in range(ord("0"), ord("9") + 1)]    # 52-61
    symbols = "_-+*/|@#&%$!?<>[](){}=~^`:,;.'\" "
    table += list(symbols)                                        # 62-93
    # Fill remaining slots with printable ASCII not yet used
    used = set(table)
    c = 33  # '!'
    while len(table) < 128:
        ch = chr(c)
        if ch not in used and ch.isprintable():
            table.append(ch)
            used.add(ch)
        c += 1
    return table


_ALPHA_TABLE: list[str] = _build_alpha_table()
# Ensure exactly 128 entries
assert len(_ALPHA_TABLE) == 128, f"Alpha table has {len(_ALPHA_TABLE)} entries, need 128"

_ALPHA_LOOKUP: dict[str, int] = {ch: i for i, ch in enumerate(_ALPHA_TABLE)}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TYPE_NUMERIC = 0  # bit 0 = 0
TYPE_ALPHA = 1    # bit 0 = 1
NUMERIC_MAX = 127  # 7-bit max


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_numeric(value: int) -> int:
    """Encode a numeric value (0-127) into an indexed byte.

    Bit layout: [value_bits:7 | 0]
    """
    if not 0 <= value <= NUMERIC_MAX:
        raise ValueError(f"Numeric value {value} out of range [0, {NUMERIC_MAX}]")
    return (value << 1) | TYPE_NUMERIC


def decode_numeric(byte: int) -> int:
    """Decode an indexed byte back to a numeric value.

    Returns the 7-bit value (0-127).
    """
    return byte >> 1


def encode_alpha(char: str) -> int:
    """Encode a letter/character into an indexed byte.

    Bit layout: [alpha_index:7 | 1]
    """
    if char not in _ALPHA_LOOKUP:
        # Map unknown characters to index 0 ('A') as fallback
        idx = 0
    else:
        idx = _ALPHA_LOOKUP[char]
    return (idx << 1) | TYPE_ALPHA


def decode_alpha(byte: int) -> str:
    """Decode an indexed byte back to an alphabetic character."""
    idx = byte >> 1
    return _ALPHA_TABLE[idx]


def byte_type(byte: int) -> str:
    """Return 'numeric' or 'alpha' for a given indexed byte."""
    return "numeric" if (byte & 1) == TYPE_NUMERIC else "alpha"


def encode_indexed(values: list[int], types: list[str]) -> bytes:
    """Encode paired (value, type) lists into indexed bytes.

    Parameters
    ----------
    values : list[int]
        Numeric values (0-127) or alpha character ordinal indices.
    types : list[str]
        Either 'numeric' or 'alpha' for each position.

    Returns
    -------
    bytes
        Indexed-byte sequence.
    """
    result = bytearray()
    for val, typ in zip(values, types):
        if typ == "numeric":
            result.append(encode_numeric(val))
        elif typ == "alpha":
            if isinstance(val, int) and 0 <= val < 128:
                result.append(encode_alpha(_ALPHA_TABLE[val]))
            else:
                result.append(encode_alpha(str(val)))
        else:
            raise ValueError(f"Unknown type: {typ!r}")
    return bytes(result)


def decode_indexed(data: bytes) -> tuple[list[int], list[str]]:
    """Decode indexed bytes back to (values, types).

    Returns
    -------
    values : list[int]
        Numeric values (0-127) or alpha indices (0-127).
    types : list[str]
        'numeric' or 'alpha' for each position.
    """
    values: list[int] = []
    types: list[str] = []
    for b in data:
        if (b & 1) == TYPE_NUMERIC:
            values.append(decode_numeric(b))
            types.append("numeric")
        else:
            values.append(b >> 1)  # alpha index
            types.append("alpha")
    return values, types


# ---------------------------------------------------------------------------
# Vector-level helpers – integrate with HelivecX quantization
# ---------------------------------------------------------------------------

def quantize_to_indexed(
    arr: "np.ndarray",
    *,
    num_channels: int = 3,
    alpha_channels: int = 0,
) -> tuple[bytes, dict]:
    """Quantize a float array into indexed bytes.

    The array is split into two zones:
      - First *num_channels* columns → NUMERIC (7-bit quantization 0-127)
      - Remaining *alpha_channels* columns → ALPHA (character-encoded)

    This lets the helix carry both numeric coordinates AND symbolic
    annotations (base-pair labels, structural markers, etc.) in a
    single packed byte stream.

    Parameters
    ----------
    arr : np.ndarray, shape (rows, cols)
        Float array to quantize. Per-column min/max normalization to [0, 127].
    num_channels : int
        Number of leading columns encoded as numeric.
    alpha_channels : int
        Number of trailing columns encoded as alpha.

    Returns
    -------
    encoded: bytes
        Indexed-byte stream (rows * cols bytes).
    meta : dict
        Metadata for dequantization (per-column min/scale, channel map).
    """
    import numpy as np

    rows, cols = arr.shape
    total_channels = num_channels + alpha_channels
    if cols != total_channels:
        raise ValueError(
            f"Array has {cols} columns but num_channels={num_channels} "
            f"+ alpha_channels={alpha_channels} = {total_channels}"
        )

    # Per-column normalization to [0, 127] for 7-bit encoding
    col_min = arr.min(axis=0)
    col_max = arr.max(axis=0)
    col_scale = col_max - col_min
    col_scale[col_scale == 0] = 1.0

    normalized = (arr - col_min) / col_scale * 127.0
    normalized = np.clip(normalized, 0, 127).astype(np.uint8)

    result = bytearray()
    for row in range(rows):
        for col in range(cols):
            val = int(normalized[row, col])
            if col < num_channels:
                result.append(encode_numeric(val))
            else:
                # Alpha channels: map val to a character code
                result.append(encode_alpha(_ALPHA_TABLE[val]))

    meta = {
        "num_channels": num_channels,
        "alpha_channels": alpha_channels,
        "col_min": col_min.tolist(),
        "col_scale": col_scale.tolist(),
    }
    return bytes(result), meta


def dequantize_from_indexed(
    data: bytes,
    rows: int,
    cols: int,
    meta: dict,
) -> "np.ndarray":
    """Dequantize indexed bytes back to a float array.

    Parameters
    ----------
    data : bytes
        Indexed-byte stream from :func:`quantize_to_indexed`.
    rows, cols : int
        Array shape.
    meta : dict
        Metadata from :func:`quantize_to_indexed`.

    Returns
    -------
    np.ndarray, shape (rows, cols)
        Reconstructed float array.
    """
    import numpy as np

    num_channels = meta["num_channels"]
    col_min = np.array(meta["col_min"])
    col_scale = np.array(meta["col_scale"])

    arr = np.zeros((rows, cols), dtype=np.float64)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            byte = data[idx]
            idx += 1
            if col < num_channels:
                val = decode_numeric(byte)
            else:
                # Alpha channels decode to index, we store the index as the value
                val = byte >> 1
            arr[row, col] = val / 127.0 * col_scale[col] + col_min[col]

    return arr


# ---------------------------------------------------------------------------
# Convenience: represent any byte as (type, decoded_value, decoded_char)
# ---------------------------------------------------------------------------

def inspect_byte(byte: int) -> dict:
    """Inspect an indexed byte, returning its type and decoded value."""
    typ = byte_type(byte)
    if typ == "numeric":
        return {
            "raw": byte,
            "type": "numeric",
            "value": decode_numeric(byte),
            "char": None,
        }
    else:
        idx = byte >> 1
        return {
            "raw": byte,
            "type": "alpha",
            "value": idx,
            "char": decode_alpha(byte),
        }
