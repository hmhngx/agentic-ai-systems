import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires Qdrant + API keys + Docling models",
    )
    config.addinivalue_line("markers", "slow: takes more than 10 seconds")
