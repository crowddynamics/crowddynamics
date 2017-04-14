call activate %CONDA_ENV%

@echo on
set PYTHONFAULTHANDLER=1

@rem `--capture=sys` avoids clobbering faulthandler tracebacks on crash
set PYTEST=py.test --capture=sys

%PYTEST%