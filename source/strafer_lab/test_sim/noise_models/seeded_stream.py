# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Each statistical case's own random stream.

A case that draws from torch's global generators draws from whatever state the cases before it left, so one seeded
case fixes every later case's sample, and at alpha 0.05 each such case is a fixed 1-in-20 draw on that stream. Each
statistical case here runs on its own seeded stream, which is restored after it, and is tested at alpha 1e-6: a
correct model does not fail by chance, and no case's draws depend on another's. The seeding is explicit per test;
nothing here is a fixture.
"""

import functools

import torch

CASE_ALPHA = 1e-6
CASE_CONFIDENCE = 1.0 - CASE_ALPHA


def seeded_stream(seed: int):
    """Run the decorated test on torch's CPU and CUDA generators seeded at ``seed``, and restore them afterwards."""

    def wrap(test):
        @functools.wraps(test)
        def run(*args, **kwargs):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                return test(*args, **kwargs)

        return run

    return wrap
