import os

import pytest
from click.testing import CliRunner

from crowddynamics.cli import list_of_simulations


@pytest.mark.parametrize('dirpath', ['.', 'name', os.path.join('folder', 'path')])
def test_startproject(tmpdir, dirpath):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result2 = runner.invoke(list_of_simulations, [])
        assert result2.exit_code == 0
