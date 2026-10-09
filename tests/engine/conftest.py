"""Shared fixtures for the engine tests."""

import pytest

from ox_zero.game import Solver


@pytest.fixture(scope="session")
def solver() -> Solver:
    """One exact solver for the whole session, so its cache is shared.

    The solver is the only ground truth the search can be checked against
    (docs/design/engineering.md, "Testing"). Solving 4x4 positions from
    scratch costs up to half a second; sharing the transposition table makes
    every fixture after the first nearly free.
    """
    return Solver()
