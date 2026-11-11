#!/usr/bin/env python3
"""User-run compute-node schema inspection; never reads assay dataset values."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ftd_prediction.h5ad_io import inspect_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true', help='Only for invented local fixtures')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    if not args.synthetic and not os.environ.get('LSB_JOBID', '').isdigit():
        parser.error('Run this probe inside an LSF compute job')
    import h5py
    print(json.dumps({'probe': 'h5ad_metadata_only', 'python': platform.python_version(),
                      'h5py': h5py.__version__, 'input_count': len(args.files)}))
    failed = False
    for index, path in enumerate(args.files, 1):
        try:
            result = inspect_schema(path)
            print(json.dumps({'input_number': index, 'schema': result}, indent=2))
        except Exception as error:
            # No exception text: a reader can include private identifiers/values.
            print(json.dumps({'input_number': index, 'status': 'failed',
                              'error_type': type(error).__name__}))
            failed = True
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
