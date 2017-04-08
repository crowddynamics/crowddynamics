import os

import pytest
from click.testing import CliRunner

from crowddynamics.cli import startproject, new, list_of_simulations, \
    CROWDDYNAMICS_CFG, ENVIRONMENT_YML


@pytest.mark.parametrize('dirpath', ['.', 'name', os.path.join('folder', 'path')])
def test_startproject(tmpdir, dirpath):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(startproject, [dirpath])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(dirpath, CROWDDYNAMICS_CFG))
        assert os.path.exists(os.path.join(dirpath, ENVIRONMENT_YML))

        os.chdir(dirpath)
        result2 = runner.invoke(new, ['simu'])
        assert result2.exit_code == 0

        result2 = runner.invoke(list_of_simulations, [])
        assert result2.exit_code == 0
