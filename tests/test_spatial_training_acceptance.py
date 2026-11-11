"""Lead-owned offline synthetic acceptance for the optional spatial backend."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(n) for n in ('anndata', 'novae', 'scipy'))


@unittest.skipUnless(AVAILABLE, 'Optional neural environment required')
class SpatialAcceptance(unittest.TestCase):
    def fixture(self):
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        rng = np.random.default_rng(11)
        xy = np.array([(x, y) for x in range(8) for y in range(8)], dtype=float) * 10
        xy = np.concatenate([xy, xy, xy])
        rates = np.ones((192, 32))
        rates[xy[:, 0] < 40, :8] = 6
        rates[xy[:, 0] >= 40, 8:16] = 6
        counts = csr_matrix(rng.poisson(rates).astype('float32'))
        a = ad.AnnData(X=counts.copy(), obs=pd.DataFrame({
            'donor_id': ['synthetic_train'] * 128 + ['synthetic_test'] * 64,
            'section_id': ['section_A'] * 64 + ['section_B'] * 64 + ['section_C'] * 64,
        }, index=[f'synthetic_cell_{i}' for i in range(192)]),
            var=pd.DataFrame(index=[f'synthetic_gene_{i}' for i in range(32)]))
        a.layers['counts'] = counts
        a.obsm['spatial'] = xy + rng.normal(0, .05, xy.shape)
        return a

    def config(self):
        return dict(mode='synthetic_smoke', train_donors=['synthetic_train'],
                    test_donors=['synthetic_test'], seed=11, max_epochs=2,
                    batch_size=16, accelerator='cpu', num_workers=0,
                    max_minutes=2, radius_um=18.0, domain_level=2)

    def test_actual_offline_training_and_boundaries(self):
        from ftd_prediction.spatial_training import run_spatial
        a = self.fixture()
        original = a.layers['counts'].copy()
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'fit'
            result = run_spatial(a, self.config(), output)
            self.assertEqual(result['method'], 'novae')
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['encoder_updated'])
            self.assertTrue(result['reload_verified'])
            self.assertGreater(result['training_steps'], 0)
            self.assertEqual(result['metrics']['cross_section_edges'], 0)
            self.assertTrue(result['artifacts'])
            for relative in result['artifacts'].values():
                path = (output / relative).resolve()
                self.assertTrue(path.is_relative_to(output.resolve()))
                self.assertTrue(path.exists())
            with self.assertRaises((ValueError, FileExistsError)):
                run_spatial(a, self.config(), output)
        self.assertEqual((a.layers['counts'] != original).nnz, 0)
        self.assertNotIn('novae_latent', a.obsm)

    def test_section_with_multiple_donors_rejected(self):
        from ftd_prediction.spatial_training import run_spatial
        a = self.fixture()
        a.obs['section_id'] = 'invalid_shared_section'
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                run_spatial(a, self.config(), Path(root) / 'bad')


if __name__ == '__main__':
    unittest.main()
