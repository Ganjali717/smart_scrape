"""Typed format checks used by the `format` constraint and candidate generation."""

from __future__ import annotations

import re

_CURRENCY = r"(?:[$£€¥₼₹₽]|usd|eur|gbp|czk|kč|azn|rs\.?)"
_AMOUNT = r"\d{1,3}(?:[ ,. ]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
PRICE_RE = re.compile(rf"^\s*(?:{_CURRENCY}\s*(?:{_AMOUNT})|(?:{_AMOUNT})\s*{_CURRENCY})\s*$", re.I)
PRICE_LOOSE_RE = re.compile(rf"(?:{_CURRENCY}\s*(?:{_AMOUNT})|(?:{_AMOUNT})\s*{_CURRENCY})", re.I)
NUMBER_RE = re.compile(r"^\s*[-+]?\d[\d,.\s]*\s*$")
DATE_RE = re.compile(
    r"(\d{1,4}[./-]\d{1,2}[./-]\d{1,4})|"
    r"((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(,\s*\d{2,4})?)|"
    r"(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4})|"
    r"(\b(19|20)\d{2}\b)", re.I)
URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.I)


def is_price(text: str, strict: bool = False) -> bool:
    t = text.strip()
    if len(t) > 40:
        return False
    return bool(PRICE_RE.match(t)) if strict else bool(PRICE_LOOSE_RE.search(t)) and any(c.isdigit() for c in t)


def is_number(text: str) -> bool:
    return bool(NUMBER_RE.match(text)) and len(text.strip()) <= 30


def is_date(text: str) -> bool:
    return bool(DATE_RE.search(text)) and len(text.strip()) <= 60


def is_text_value(text: str) -> bool:
    t = text.strip()
    return 1 < len(t) <= 300


def matches_type(text: str, ftype: str) -> bool:
    if ftype == "price":
        return is_price(text)
    if ftype == "number":
        return is_number(text)
    if ftype == "date":
        return is_date(text)
    if ftype == "url":
        return bool(URL_RE.match(text.strip()))
    return is_text_value(text)
