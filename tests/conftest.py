import time

import numpy as np
import pytest


class FileTimer:
    def __init__(self) -> None:
        self.timers: dict[str, float] = {}
        self.current_file: str | None = None


file_tracker = FileTimer()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    file_path = str(item.fspath)

    # If this is the first test from this file
    if file_path != file_tracker.current_file:
        file_tracker.current_file = file_path
        if file_path not in file_tracker.timers:
            file_tracker.timers[file_path] = time.time()

    yield

    # If this is the last test in the file
    if nextitem is None or str(nextitem.fspath) != file_path:
        duration = time.time() - file_tracker.timers[file_path]
        reporter = item.config.pluginmanager.get_plugin("terminalreporter")
        if reporter and hasattr(reporter, "_tw"):
            reporter._tw.write(f" [{duration:.4f}s]", cyan=True)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    """Conditionally skip slow tests unless '--runslow' is specified"""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture
def rng():
    return np.random.default_rng()
