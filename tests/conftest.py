"""Let torch and xgboost share one pytest process.

On macOS each ships its own OpenMP runtime. With both running in one process, torch first makes
xgboost segfault and xgboost first makes multi-threaded torch deadlock. The one safe combination is
xgboost loaded first and torch on a single thread. The make stages never mix the two in a process
(docs/findings.md, Day 6); this file is only for the test suite. Harmless on Linux.
"""

import xgboost  # noqa: F401  (must load before torch)
import torch

torch.set_num_threads(1)
