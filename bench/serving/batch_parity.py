# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
"""Greedy parity of batched decoding: every prompt is first run alone, then all prompts are
sent at once (so the server decodes them in one batch); each request's token ids must match
its single-stream run.

    python bench/serving/batch_parity.py --base-url http://localhost:8000 \
        --prompts bench/serving/prompts_set_a.json bench/serving/prompts_set_b.json --max-tokens 256
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def ids_of(base, model, prompt, max_tokens, speculative):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "speculative": speculative, "return_token_ids": True}
    r = post(base, "/v1/completions", body)
    c = r["choices"][0]
    return c.get("token_ids") or c["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="model")
    ap.add_argument("--prompts", nargs="+", required=True, help="JSON lists of user messages")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--speculative", action="store_true", help="single-stream reference with the draft")
    a = ap.parse_args()
    msgs = [m for f in a.prompts for m in json.load(open(f))]
    ids = [post(a.base_url, "/tokenize", {"messages": [{"role": "user", "content": m}]})["tokens"] for m in msgs]
    ref = [ids_of(a.base_url, a.model, x, a.max_tokens, a.speculative) for x in ids]
    with ThreadPoolExecutor(len(ids)) as ex:
        got = list(ex.map(lambda x: ids_of(a.base_url, a.model, x, a.max_tokens, False), ids))
    bad = 0
    for i, (r, g) in enumerate(zip(ref, got)):
        if r != g:
            bad += 1
            n = next((j for j, (p, q) in enumerate(zip(r, g)) if p != q), min(len(r), len(g)))
            print(f"prompt {i:2d}: MISMATCH at token {n} of {len(r)}")
    print(f"{len(ids) - bad} / {len(ids)} prompts match their single-stream output ({len(ids)} at once)")


if __name__ == "__main__":
    main()
