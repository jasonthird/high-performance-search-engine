"""Validate a CoreML shape against FP32, including masks and row permutations.

Usage: python ane_validate.py MODEL --batch 8 --seq 64 --root /path/to/repo
Works with .mlpackage and .mlmodelc; never modifies installed models.
"""
import argparse
import json
from pathlib import Path
import time

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compute-units", choices=["CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU", "ALL"], default="CPU_AND_NE")
    args = parser.parse_args()
    if args.batch < 1 or args.seq < 2:
        parser.error("batch must be positive and seq must be at least 2")

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/CodeRankEmbed")
    reference = AutoModel.from_pretrained(
        "nomic-ai/CodeRankEmbed", trust_remote_code=True, safe_serialization=True
    ).eval()
    cls = ct.models.CompiledMLModel if args.model.suffix == ".mlmodelc" else ct.models.MLModel
    start = time.perf_counter()
    model = cls(str(args.model), compute_units=getattr(ct.ComputeUnit, args.compute_units))
    load_ms = (time.perf_counter() - start) * 1000
    files = sorted((args.root / "src").rglob("*.rs"))
    if not files:
        parser.error("root/src must contain Rust source files")
    texts = [file.read_text()[:8000] for file in files]
    minimum_reference = 1.0
    minimum_permutation = 1.0
    times = []
    cases = []

    def predict(ids, mask):
        start = time.perf_counter()
        output = model.predict({"input_ids": ids, "attention_mask": mask})["embedding"]
        times.append((time.perf_counter() - start) * 1000)
        return np.asarray(output, dtype=np.float32)

    def normalized(output):
        if not np.isfinite(output).all():
            raise ValueError("non-finite embedding")
        norm = np.linalg.norm(output, axis=1, keepdims=True)
        if not ((norm > 0.9) & (norm < 1.1)).all():
            raise ValueError("non-unit embedding")
        return output / norm

    # Full, mixed-length, and partial batches; padding-row outputs are ignored.
    for case in range(9):
        real_rows = args.batch if case < 6 else (max(1, args.batch - 1), max(1, args.batch // 2), 1)[case - 6]
        ids = np.zeros((args.batch, args.seq), dtype=np.int32)
        mask = np.zeros_like(ids)
        for row in range(real_rows):
            cap = args.seq if case % 3 == 0 else min(args.seq, 4 + (row * 13 + case * 7) % args.seq)
            tokens = tokenizer(texts[(case * args.batch + row) % len(texts)],
                               truncation=True, max_length=cap)["input_ids"]
            ids[row, :len(tokens)] = tokens
            mask[row, :len(tokens)] = 1
        with torch.inference_mode():
            hidden = reference(input_ids=torch.from_numpy(ids).long(),
                               attention_mask=torch.from_numpy(mask).long())[0]
            ref = torch.nn.functional.normalize(hidden[:real_rows, 0, :], dim=-1).numpy()
        result = normalized(predict(ids, mask)[:real_rows])
        permuted = normalized(predict(ids[::-1].copy(), mask[::-1].copy())[::-1][:real_rows])
        ref_cosine = float((normalized(ref) * result).sum(axis=1).min())
        permutation_cosine = float((result * permuted).sum(axis=1).min())
        cases.append(dict(case=case, rows=real_rows, reference=ref_cosine, permutation=permutation_cosine))
        minimum_reference = min(minimum_reference, ref_cosine)
        minimum_permutation = min(minimum_permutation, permutation_cosine)
    report = dict(model=str(args.model), load_ms=load_ms,
                  prediction_median_ms=float(np.median(times[1:])),
                  min_reference_cosine=minimum_reference,
                  min_permutation_cosine=minimum_permutation,
                  compute_units=args.compute_units, cases=cases)
    print(json.dumps(report, indent=2))
    if min(minimum_reference, minimum_permutation) <= 0.995:
        raise SystemExit("CoreML shape failed fidelity validation")


if __name__ == "__main__":
    main()
