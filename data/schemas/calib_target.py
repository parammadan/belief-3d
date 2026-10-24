"""Fiducial calibration board: geometry known to both sim and inference.

A flat white board mounted at a known fixture pose on the table carries a
4 x 3 grid of magenta discs. Disc centres in the world frame are known.
That knowledge is what a real cell's calibration fixture provides.
"""
from __future__ import annotations

import numpy as np

BOARD_CENTRE_W = np.array([0.08, 0.0, 0.40])     # on the table top
BOARD_HALF = (0.17, 0.13, 0.002)                  # m
DISC_RADIUS = 0.014
DISC_HALF_H = 0.0004                               # thin: an oblique view sees ~no side wall
GRID = (4, 3)
SPACING = 0.08
DISC_RGBA = (0.85, 0.10, 0.85, 1.0)               # magenta: no catalog object has this hue


def disc_centres_W() -> np.ndarray:
    """(12, 3) top-surface centres of the discs, world frame, row-major."""
    nx, ny = GRID
    xs = (np.arange(nx) - (nx - 1) / 2) * SPACING
    ys = (np.arange(ny) - (ny - 1) / 2) * SPACING
    top = BOARD_CENTRE_W[2] + 2 * BOARD_HALF[2] + 2 * DISC_HALF_H
    return np.array([[BOARD_CENTRE_W[0] + x, BOARD_CENTRE_W[1] + y, top] for y in ys for x in xs])
