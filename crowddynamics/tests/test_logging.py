import logging

from crowddynamics.logging import setup_logging, user_info, log_with

setup_logging()
user_info()
logger = logging.getLogger(__name__)


@log_with()
def function(a, b, c='c', d='d'):
    return True


@log_with(logger)
def function2(a, b, c='c', d='d'):
    return True


class Foo:
    @log_with(logger)
    def method(self, a, b, c='c', d='d'):
        return True


def test_log_with():
    assert function('1', '2', c='3', d='4')
    assert function('a', 'b', d='d')
    assert function2('a', 'b', d='d')
    assert Foo().method('1', '2', c='3', d='4')
    assert Foo().method('a', 'b', d='d')
