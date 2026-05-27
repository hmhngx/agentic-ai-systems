import sys
import pathlib

# Add the module root to sys.path so 'from src.X import Y' works
sys.path.insert(0, str(pathlib.Path(__file__).parent))
