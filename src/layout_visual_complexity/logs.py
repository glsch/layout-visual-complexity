"""The package's logger. Modules log through it; the command line sets it up once."""

import logging

logger = logging.getLogger("layout_visual_complexity")


def setup(level: int = logging.INFO) -> None:
    """Send the package's messages at `level` and up to standard error. Calling it again only changes the level."""
    if not any(getattr(handler, "_lvc", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        handler._lvc = True
        logger.addHandler(handler)
    logger.setLevel(level)
