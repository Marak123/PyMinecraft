"""PyMinecraft launcher.

Usage:
    py launcher.py                    # play
    py launcher.py --frames 240      # auto-close after N frames (testing)
    py launcher.py --frames 240 --screenshot out.png
"""

from __future__ import annotations

import argparse
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description="PyMinecraft — voxel sandbox")
    parser.add_argument("--frames", type=int, default=None,
                        help="close automatically after N frames (for testing)")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="save a screenshot before auto-closing")
    args = parser.parse_args()

    try:
        from game.game import Game

        Game(max_frames=args.frames, screenshot_path=args.screenshot).run()
        return 0
    except Exception:  # noqa: BLE001 - top-level crash guard
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
