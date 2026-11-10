"""The package's logger. Modules log through it; the command line sets it up once."""

import logging

logger = logging.getLogger("layout_visual_complexity")


def setup(verbose: bool = False) -> None:
    """Send the package's messages to standard error: warnings and up, or everything when verbose."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
