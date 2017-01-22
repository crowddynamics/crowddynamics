import pytest
from click.testing import CliRunner

from crowddynamics.cli import run, gui


@pytest.mark.skip
def test_run():
    assert True


@pytest.mark.skip
def test_gui():
    runner = CliRunner()
    result = runner.invoke(gui)
    assert result.exit_code == 0
    assert result.output == ''
