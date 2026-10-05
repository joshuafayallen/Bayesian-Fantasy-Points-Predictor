#!/usr/bin/env python
"""Run the src-dev pipeline: ``uv run python src-dev/ff.py --help`` from the project root."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore', category=FutureWarning)

from ffpred.cli import main  # noqa: E402

if __name__ == '__main__':
    main()
