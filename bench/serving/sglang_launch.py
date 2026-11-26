# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
"""`python3 -m sglang.launch_server` with the Inkling2 config classes registered with
transformers first. Some builds of the SGLang image (arm64) read the checkpoint config before
their own registration runs. The registration also runs in the scheduler subprocesses, which
import this file; only the main process launches the server."""
import runpy
import sys

from sglang.srt.configs import inkling2 as _inkling2
from transformers import AutoConfig

for _name in ("Inkling2LoopMMConfig", "Inkling2MMConfig", "Inkling2MTPConfig"):
    _cls = getattr(_inkling2, _name, None)
    if _cls is not None:
        try:
            AutoConfig.register(_cls.model_type, _cls, exist_ok=True)
        except TypeError:
            AutoConfig.register(_cls.model_type, _cls)

if __name__ == "__main__":
    sys.argv = ["sglang.launch_server"] + sys.argv[1:]
    runpy.run_module("sglang.launch_server", run_name="__main__")
