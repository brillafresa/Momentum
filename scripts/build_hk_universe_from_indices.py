# -*- coding: utf-8 -*-
"""
Regenerate ``hongkong_universe.csv`` from index constituent sources (LIVE).

Purpose
-------
Build the FREE-mode Hong Kong universe as the deduplicated union of:

- HSI (Hang Seng Index) — public CSV mirror
- HSCEI (Hang Seng China Enterprises Index) — Hang Seng factsheet PDF
- HSTECH (Hang Seng TECH Index) — Hang Seng factsheet PDF

Output columns: ``Symbol,Name`` with Yahoo-compatible ``####.HK`` symbols.

Usage (from repo root)
----------------------
    python scripts/build_hk_universe_from_indices.py

Requires network access to yfiua.github.io and hsi.com.hk. Not imported by
``app.py`` or ``run_scan_batch.py`` — manual maintenance only.
"""

from __future__ import annotations

import io
import re
from collections import OrderedDict

import pandas as pd
import requests
from pypdf import PdfReader


HSI_CSV_URL = "https://yfiua.github.io/index-constituents/constituents-hsi.csv"
HSCEI_PDF_URL = "https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/factsheets/hsceie.pdf"
HSTECH_PDF_URL = "https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/factsheets/hsteche.pdf"

INDUSTRY_SUFFIXES = [
    "Properties & Construction",
    "Consumer Discretionary",
    "Information Technology",
    "Telecommunications",
    "Consumer Staples",
    "Conglomerates",
    "Industrials",
    "Financials",
    "Healthcare",
    "Materials",
    "Utilities",
    "Energy",
]

SHARE_TYPE_SUFFIXES = [
    "Other HK-listed Mainland Co.",
    "HK Ordinary",
    "H Share",
    "Red Chip",
]


def _normalize_symbol(code: str) -> str:
    return f"{int(code):04d}.HK"


def load_hsi_from_csv() -> list[tuple[str, str]]:
    df = pd.read_csv(HSI_CSV_URL)
    if "Symbol" not in df.columns or "Name" not in df.columns:
        raise RuntimeError("HSI CSV schema mismatch")
    out = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        name = str(row["Name"]).strip()
        if re.match(r"^\d{4,5}\.HK$", sym):
            out.append((sym, name if name else sym))
    if not out:
        raise RuntimeError("failed to load HSI constituents from CSV")
    return out


def load_from_factsheet_pdf(url: str) -> list[tuple[str, str]]:
    data = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).content
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    rows: list[tuple[str, str]] = []

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        m = re.match(r"^(\d{3,5})\s+([A-Z0-9]{10,12})\s+(.+?)\s+([0-9]+\.[0-9]{2})$", line)
        if not m:
            continue
        payload = m.group(3).strip()
        for sfx in SHARE_TYPE_SUFFIXES:
            if payload.endswith(sfx):
                payload = payload[: -len(sfx)].strip()
                break
        for sfx in INDUSTRY_SUFFIXES:
            if payload.endswith(sfx):
                payload = payload[: -len(sfx)].strip()
                break

        symbol = _normalize_symbol(m.group(1))
        name = payload if payload else symbol
        rows.append((symbol, name))

    if not rows:
        raise RuntimeError(f"failed to parse factsheet: {url}")
    return rows


def main() -> None:
    merged: OrderedDict[str, str] = OrderedDict()

    for symbol, name in load_hsi_from_csv():
        merged[symbol] = name
    for symbol, name in load_from_factsheet_pdf(HSCEI_PDF_URL):
        merged.setdefault(symbol, name)
    for symbol, name in load_from_factsheet_pdf(HSTECH_PDF_URL):
        merged.setdefault(symbol, name)

    out = pd.DataFrame(list(merged.items()), columns=["Symbol", "Name"])
    out = out[out["Symbol"].str.match(r"^\d{4,5}\.HK$", na=False)]
    out = out.drop_duplicates("Symbol", keep="first").sort_values("Symbol").reset_index(drop=True)
    out.to_csv("hongkong_universe.csv", index=False, encoding="utf-8")

    print(f"rows={len(out)}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

