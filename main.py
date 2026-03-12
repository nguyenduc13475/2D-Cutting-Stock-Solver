import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.ui.app import CSP2DApp

def main():
    """
    Entry point of the 2D Cutting Stock Solver Application.
    """
    app = CSP2DApp()
    app.run()

if __name__ == "__main__":
    main()