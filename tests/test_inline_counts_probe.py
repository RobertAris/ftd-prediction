"""Bounded diagnostic checks on invented matrices, never a count-provenance claim."""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(x) for x in ('anndata', 'h5py', 'numpy', 'scipy'))
SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/inline_h5ad_counts_probe.py'


@unittest.skipUnless(AVAILABLE, 'Optional H5AD environment required')
class InlineCountsProbeTests(unittest.TestCase):
    def fixture(self, path, dense=False):
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        counts = np.full((4, 30001), 7, dtype='float32')
        a = ad.AnnData(X=counts if dense else csr_matrix(counts),
                       obs=pd.DataFrame(index=[f'invented_cell_{i}' for i in range(4)]))
        a.raw = a.copy()
        a.X = np.full((4, 30001), .25, dtype='float32') if dense else csr_matrix(np.full((4, 30001), .25, dtype='float32'))
        a.write_h5ad(path)

    def run_probe(self, path):
        output = io.StringIO()
        with patch.object(sys, 'argv', [str(SCRIPT), str(path)]), patch.object(sys, 'excepthook', sys.excepthook), patch.dict(os.environ, {'LSB_JOBID': '1'}), contextlib.redirect_stdout(output):
            runpy.run_path(str(SCRIPT))
        return [ast.literal_eval(line) for line in output.getvalue().splitlines()]

    def test_sparse_and_dense_flags_without_expression_values(self):
        import h5py
        for dense in (False, True):
            with self.subTest(dense=dense), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'invented.h5ad';self.fixture(path,dense)
                sizes=[];original=h5py.Dataset.__getitem__
                def bounded(node, selection):
                    value=original(node,selection);sizes.append(value.size)
                    self.assertLessEqual(value.size,100000)
                    return value
                with patch.object(h5py.Dataset,'__getitem__',bounded):
                    results=self.run_probe(path)
                self.assertEqual(len(results),2)
                by_path={r['matrix']:r for r in results}
                self.assertFalse(by_path['X']['sample_integer'])
                self.assertTrue(by_path['raw/X']['sample_integer'])
                self.assertTrue(all(r['sample_finite'] and r['sample_nonnegative'] for r in results))
                self.assertTrue(sizes)
                self.assertEqual(set(results[0]), {'input_number','matrix','sampled_values','sample_informative','sample_finite','sample_nonnegative','sample_integer'})

    def test_external_links_rejected_before_any_values(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'invented.h5ad';self.fixture(path)
            with h5py.File(path,'a') as f:
                del f['raw/X'];f['raw/X']=h5py.ExternalLink('unopened.h5','X')
            with patch.object(h5py.Dataset,'__getitem__',side_effect=self.fail):
                with self.assertRaises(AssertionError):self.run_probe(path)

    def test_paste_format_and_compute_job_guard(self):
        source=SCRIPT.read_text()
        self.assertTrue(source.isascii())
        self.assertNotIn('__',source)
        self.assertNotIn(chr(34),source)
        self.assertNotIn(chr(39),source)
        self.assertFalse(any(line.startswith((' ', '\t')) for line in source.splitlines()))
        with patch.dict(os.environ, {'LSB_JOBID': '0'}),patch.object(sys,'argv',[str(SCRIPT),'never-open.h5ad']),patch.object(sys,'excepthook',sys.excepthook):
            with self.assertRaises(AssertionError):runpy.run_path(str(SCRIPT))

    def test_uncaught_error_is_sanitized_and_empty_samples_are_uninformative(self):
        import subprocess
        import h5py
        result=subprocess.run([sys.executable,'-E',str(SCRIPT),'never-open-private-path.h5ad'],
                              env={**os.environ,'LSB_JOBID':'1'},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(ast.literal_eval(result.stdout.strip()),{'probe_failed':True})
        self.assertEqual(result.stderr,'')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'invented.h5ad';self.fixture(path)
            with h5py.File(path,'a') as f:
                del f['X/data'];f['X'].create_dataset('data',shape=(0,),dtype='float32')
            report=self.run_probe(path)
            self.assertFalse(next(r for r in report if r['matrix']=='X')['sample_informative'])


if __name__ == '__main__':
    unittest.main()
