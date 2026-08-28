"""Template prompt & builder untuk anotasi sentimen (Phase 3)."""
from __future__ import annotations

import json

VALID_LABELS = ("negatif", "netral", "positif")

SYSTEM_PROMPT = (
    "Anda adalah ahli anotasi sentimen untuk tweet bahasa Indonesia tentang banjir di Sumatra. "
    "Klasifikasikan sentimen setiap tweet ke salah satu label: 'negatif', 'netral', atau 'positif'."
)

BATCH_PROMPT = """Anda akan menerima {n} tweet dalam format JSON (satu objek per baris).

Aturan klasifikasi:
- "negatif"   : keluhan, kritik, ketakutan, sedih, marah, sarkasme bernada negatif.
- "netral"    : informasi faktual, laporan, peringatan tanpa emosi kuat.
- "positif"   : apresiasi, dukungan, doa, harapan, bantuan, semangat.
- confidence  : bilangan bulat 0-100 seberapa yakin Anda.
- reason      : satu kalimat singkat dalam bahasa Indonesia menjelaskan alasan.

Tweet:
{items}

Balas HANYA dengan satu objek JSON dengan kunci "annotations" yang berisi array, misalnya:
{{"annotations": [
  {{"id": 1, "label": "negatif", "confidence": 95, "reason": "Keluhan terhadap penanganan banjir."}}
]}}
"""


def build_batch_items(rows) -> str:
    """rows: iterable of dict(id, text_with_emoticon) -> string JSON per baris."""
    return "\n".join(
        json.dumps({"id": int(r["id"]), "text": str(r["text_with_emoticon"])}, ensure_ascii=False)
        for r in rows
    )


def build_batch_prompt(rows) -> tuple[str, str]:
    """Kembalikan (system_prompt, user_prompt)."""
    items = build_batch_items(rows)
    user = BATCH_PROMPT.format(n=len(rows), items=items)
    return SYSTEM_PROMPT, user


def validate_label(label) -> bool:
    return isinstance(label, str) and label.strip().lower() in VALID_LABELS
