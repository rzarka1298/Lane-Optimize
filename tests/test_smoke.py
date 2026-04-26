"""Sanity-check that the package imports — guards CI green from day one."""

import laneiq


def test_version() -> None:
    assert laneiq.__version__ == "0.1.0"
