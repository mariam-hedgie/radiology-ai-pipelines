import argparse, json
from pathlib import Path

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, required=True, help="outputs/preds.jsonl from generate.py")
    ap.add_argument("--pred_key", type=str, default="gen")
    ap.add_argument("--ref_key", type=str, default="target")
    ap.add_argument("--lower", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.preds)
    assert len(rows) > 0, f"No rows found in {args.preds}"

    preds = []
    refs = []

    for r in rows:
        pred = (r.get(args.pred_key) or "").strip()
        ref = (r.get(args.ref_key) or "").strip()
        if args.lower:
            pred = pred.lower()
            ref = ref.lower()
        preds.append(pred)
        refs.append(ref)

    # ---- BLEU-4 (corpus) ----
    bleu = BLEU(effective_order=True)  # handles short sequences better
    bleu_score = bleu.corpus_score(preds, [refs]).score  # 0-100

    # ---- ROUGE-L (average F1) ----
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_f1 = 0.0
    n = 0
    for p, t in zip(preds, refs):
        s = scorer.score(t, p)["rougeL"].fmeasure
        rougeL_f1 += s
        n += 1
    rougeL_f1 /= max(1, n)

    print(f"File: {args.preds}")
    print(f"N: {len(rows)}")
    print(f"BLEU-4: {bleu_score:.2f}")
    print(f"ROUGE-L (F1): {rougeL_f1:.4f}")


if __name__ == "__main__":
    main()