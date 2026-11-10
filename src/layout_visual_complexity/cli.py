"""The layout-visual-complexity command line."""

import argparse

import layout_visual_complexity


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="layout-visual-complexity", description=layout_visual_complexity.__doc__)
    p.add_argument("--version", action="version", version=layout_visual_complexity.__version__)
    return p


def main(argv: list[str] | None = None) -> int:
    parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
