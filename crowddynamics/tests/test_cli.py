import pytest
from click.testing import CliRunner


@pytest.mark.skip
def test_run():
    runner = CliRunner()
    result = runner.invoke()  # TODO: add function to test
    assert result.exit_code == 0
    assert result.output == ''
