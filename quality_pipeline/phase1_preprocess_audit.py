"""Phase 1 — Preprocessing Audit (PRD).

Menghasilkan:
- reports/critical_stopword_report.csv   (deliverable PRD)
- reports/preprocessing_audit.md         (ringkasan audit LSTM, IndoBERT, stopword)
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from quality_pipeline import config as C
from quality_pipeline.utils import load_csv, safe_str, write_csv

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(s) -> list[str]:
    return _TOKEN_RE.findall(str(s).lower())


def stopword_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Audit stopword: pastikan kata sinyal sentimen tidak hilang dari clean_text_lstm.

    count_output  = kemunculan token persis
    count_stem    = kemunculan token yang merupakan stem kata (mis. harapan -> harap)
    """
    rows: list[dict] = []
    for w in C.CRITICAL_STOPWORDS:
        in_count = df["text_with_emoticon"].map(lambda t: tokenize(t).count(w)).sum()
        out_tokens = [tok for t in df["clean_text_lstm"] for tok in tokenize(t)]

        def _is_stem(tok: str) -> bool:
            return tok == w or w.startswith(tok) or tok.startswith(w)

        stem_count = sum(1 for tok in out_tokens if _is_stem(tok))
        exact_count = out_tokens.count(w)
        if exact_count >= in_count:
            status = "OK"
        elif stem_count >= in_count:
            status = "TER-STEMMED"
        elif stem_count == 0:
            status = "TERHAPUS"
        else:
            status = "MENURUN"
        rows.append({
            "kata": w,
            "count_input": int(in_count),
            "count_output": int(exact_count),
            "count_stem": int(stem_count),
            "status": status,
        })
    return pd.DataFrame(rows)


def lstm_negation_audit(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """Cek negasi yang hilang setelah preprocessing LSTM."""
    pat = r"\b(?:" + "|".join(C.NEGATION_WORDS) + r")\b"
    sub = df[df["text_with_emoticon"].str.contains(pat, case=False, na=False)].head(n)
    rows: list[dict] = []
    for i, r in sub.iterrows():
        in_tok = set(tokenize(r["text_with_emoticon"]))
        out_tok = set(tokenize(r["clean_text_lstm"]))
        lost = [w for w in C.NEGATION_WORDS if w in in_tok and w not in out_tok]
        if lost:
            rows.append({
                "id": i,
                "jenis": "negasi_hilang",
                "kata": ",".join(lost),
                "input": safe_str(r["text_with_emoticon"])[:250],
                "output": safe_str(r["clean_text_lstm"])[:250],
            })
    return pd.DataFrame(rows)


def lstm_stemming_audit(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """Deteksi stemming agresif: kata input panjang -> output sangat pendek."""
    sub = df.sample(n, random_state=C.RANDOM_STATE)
    rows: list[dict] = []
    for i, r in sub.iterrows():
        in_tok = tokenize(r["text_with_emoticon"])
        out_tok = tokenize(r["clean_text_lstm"])
        for out_w in set(out_tok):
            if len(out_w) < 3:
                for in_w in in_tok:
                    if len(in_w) >= 7 and in_w.startswith(out_w):
                        rows.append({
                            "id": i,
                            "jenis": "stemming_agresif",
                            "kata_input": in_w,
                            "kata_output": out_w,
                            "input": safe_str(r["text_with_emoticon"])[:250],
                            "output": safe_str(r["clean_text_lstm"])[:250],
                        })
                        break
    return pd.DataFrame(rows)


def bert_audit(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """Audit IndoBERT: hashtag, emoji, tanda baca, dan noise yang lolos."""
    from Preprocessing.emoji_dict import emoji_dict

    sub = df.sample(n, random_state=C.RANDOM_STATE)
    rows: list[dict] = []
    for i, r in sub.iterrows():
        inp = safe_str(r["text_with_emoticon"])
        out = safe_str(r["text_bert"]).lower()

        hashtags = re.findall(r"#(\w+)", inp)
        lost_h = [h for h in hashtags if h.lower() not in out]
        if lost_h:
            rows.append({
                "id": i, "jenis": "hashtag_hilang", "detail": ",".join(lost_h),
                "input": inp[:250], "output": safe_str(r["text_bert"])[:250],
            })

        for emo, word in emoji_dict.items():
            if emo in inp and word.strip() not in out:
                rows.append({
                    "id": i, "jenis": "emoji_hilang", "detail": f"{emo} -> {word.strip()}",
                    "input": inp[:250], "output": safe_str(r["text_bert"])[:250],
                })
                break

        if "?" in inp and "?" not in safe_str(r["text_bert"]):
            rows.append({
                "id": i, "jenis": "tanda_baca_hilang", "detail": "?",
                "input": inp[:250], "output": safe_str(r["text_bert"])[:250],
            })
        if "!" in inp and "!" not in safe_str(r["text_bert"]):
            rows.append({
                "id": i, "jenis": "tanda_baca_hilang", "detail": "!",
                "input": inp[:250], "output": safe_str(r["text_bert"])[:250],
            })

        for name, pattern in C.NOISE_PATTERNS.items():
            if pattern.search(inp) and not pattern.search(safe_str(r["text_bert"])):
                rows.append({
                    "id": i, "jenis": "noise_terbersihkan_bert", "detail": name,
                    "input": inp[:250], "output": safe_str(r["text_bert"])[:250],
                })
                break
    return pd.DataFrame(rows)


def run(dry_run: bool = False) -> None:
    print("Phase 1 — Preprocessing Audit")
    df = load_csv(C.PREPROCESSED_CSV)
    print(f"  Memuat {len(df)} baris dari {C.PREPROCESSED_CSV.name}")

    sw = stopword_audit(df)
    write_csv(sw, C.REPORTS_DIR / "critical_stopword_report.csv")
    print("  Stopword audit -> reports/critical_stopword_report.csv")
    terhapus = sw[~sw["status"].isin(("OK", "TER-STEMMED"))]
    if len(terhapus):
        print(f"  [PERHATIAN] {len(terhapus)} kata kritis turun/hilang: {list(terhapus['kata'])}")
    else:
        print("  Semua kata kritis tetap dipertahankan (keep_words bekerja).")

    neg = lstm_negation_audit(df)
    neg.to_csv(C.REPORTS_DIR / "lstm_negation_audit.csv", index=False, encoding="utf-8")
    stem = lstm_stemming_audit(df)
    stem.to_csv(C.REPORTS_DIR / "lstm_stemming_audit.csv", index=False, encoding="utf-8")
    bert = bert_audit(df)
    bert.to_csv(C.REPORTS_DIR / "bert_audit.csv", index=False, encoding="utf-8")

    lines = [
        "# Laporan Preprocessing Audit",
        "",
        f"**Sumber:** `Data/{C.PREPROCESSED_CSV.name}` · **Jumlah baris:** {len(df)}",
        "",
        "## Audit Stopword (kata sinyal sentimen)",
        "",
        "| Kata | Count input | Count output | Count stem | Status |",
        "|---|---|---|---|---|",
    ]
    for _, r in sw.iterrows():
        lines.append(f"| {r['kata']} | {r['count_input']} | {r['count_output']} | {r['count_stem']} | {r['status']} |")
    lines += [
        "",
        "## Audit LSTM",
        "",
        f"- Kasus negasi hilang: **{len(neg)}** (detail: `reports/lstm_negation_audit.csv`)",
        f"- Potensi stemming agresif: **{len(stem)}** (detail: `reports/lstm_stemming_audit.csv`)",
        "",
        "## Audit IndoBERT",
        "",
        f"- Temuan: **{len(bert)}** (detail: `reports/bert_audit.csv`)",
        "",
        "Catatan: `text_bert` (preprocess_bert) tidak menghapus noise seperti "
        "'tampilkan lebih banyak', 'rb', dan timestamp — ini dikoreksi pada `data_banjir_v2.csv` "
        "lewat cleaning rules Task 0.4.",
        "",
    ]
    (C.REPORTS_DIR / "preprocessing_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("  Ringkasan -> reports/preprocessing_audit.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 — Preprocessing Audit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
