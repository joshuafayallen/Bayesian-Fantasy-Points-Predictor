"""
Run tests/diagnostics_report.py's comparison for all three architectures
in one shot (whichever of ff_ar_new.nc / ff_hs_new.nc / ff_team_new.nc
exist next to their ground truth in model-nc/, from
tests/refit_all_and_compare.py).

Usage:
    python tests/compare_all_diagnostics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from diagnostics_report import compare

MODELS = ['ar', 'hs', 'team']

if __name__ == '__main__':
    results = {}
    for name in MODELS:
        old_path = f'model-nc/ff_{name}.nc'
        new_path = f'model-nc/ff_{name}_new.nc'
        if not os.path.exists(new_path):
            print(f"skipping {name}: {new_path} not found (run tests/refit_all_and_compare.py {name} first)\n")
            continue
        print(f"##### {name} #####")
        results[name] = compare(old_path, new_path)
        print()

    if results:
        print("summary:", {k: ('HEALTHY' if v else 'CHECK NEEDED') for k, v in results.items()})
        sys.exit(0 if all(results.values()) else 1)
