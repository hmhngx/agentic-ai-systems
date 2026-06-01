import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires OPENROUTER_API_KEY")
    config.addinivalue_line("markers", "slow: takes more than 10 seconds")
