import pytest

import layout_visual_complexity
from layout_visual_complexity import cli


def test_the_command_line_shows_its_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert layout_visual_complexity.__version__ in capsys.readouterr().out
