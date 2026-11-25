# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
"""Table of a `bench/baseline_servers.sh` run: output tok/s and TTFT p50 / p99 per engine and
concurrency.

    python bench/serving/summarize.py ~/astral-work/results/<stamp>-servers-inkling [more dirs...]
"""

import glob
import json
import os
import re
import sys


def main():
    rows = {}
    for d in sys.argv[1:]:
        for f in glob.glob(os.path.join(d, "*-c*.json")):
            m = re.match(r"(.+)-c(\d+)\.json$", os.path.basename(f))
            r = json.load(open(f))
            rows[(m.group(1), int(m.group(2)))] = (r["output_throughput"], r["median_ttft_ms"], r["p99_ttft_ms"])
    engines = sorted({e for e, _ in rows})
    levels = sorted({c for _, c in rows})
    print("| Concurrency | " + " | ".join(f"{e} tok/s | {e} TTFT p50 / p99 (ms)" for e in engines) + " |")
    print("|---" * (1 + 2 * len(engines)) + "|")
    for c in levels:
        cells = []
        for e in engines:
            v = rows.get((e, c))
            cells += [f"{v[0]:.1f}", f"{v[1]:.0f} / {v[2]:.0f}"] if v else ["—", "—"]
        print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
