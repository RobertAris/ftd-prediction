"""Lead-owned, synthetic acceptance for the optional RNA backend."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(n) for n in ('anndata', 'scvi', 'scipy'))


@unittest.skipUnless(AVAILABLE, 'Optional neural environment required')
class RNAAcceptance(unittest.TestCase):
    def fixture(self):
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        rng = np.random.default_rng(11)
        counts = csr_matrix(rng.poisson(3, (64, 32)).astype('float32'))
        a = ad.AnnData(X=counts.copy(), obs=pd.DataFrame({
            'donor_id': ['synthetic_train'] * 32 + ['synthetic_test'] * 32,
            'cell_type': (['A'] * 16 + ['B'] * 16) * 2,
        }, index=[f'synthetic_cell_{i}' for i in range(64)]),
            var=pd.DataFrame(index=[f'gene_{i:02d}' for i in range(32)]))
        a.layers['counts'] = counts
        return a

    def config(self):
        return dict(train_donors=['synthetic_train'], test_donors=['synthetic_test'],
                    seed=11, max_epochs=2, max_genes=16, n_hidden=16,
                    n_latent=4, n_layers=1, batch_size=16, accelerator='cpu',
                    max_minutes=2, num_threads=1, kl_warmup_epochs=1)

    def test_actual_fit_preserves_inputs_and_roundtrips(self):
        import numpy as np
        from ftd_prediction.rna_training import run_rna
        a = self.fixture()
        original = a.layers['counts'].copy()
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'fit'
            result = run_rna(a, self.config(), output)
            self.assertEqual(result['method'], 'scvi')
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['encoder_updated'])
            self.assertTrue(result['reload_verified'])
            self.assertGreater(result['training_steps'], 0)
            self.assertEqual(result['train_donors'], ['synthetic_train'])
            self.assertEqual(result['test_donors'], ['synthetic_test'])
            self.assertEqual(len(result['selected_genes']), 16)
            self.assertTrue(result['artifacts'])
            for relative in result['artifacts'].values():
                path = (output / relative).resolve()
                self.assertTrue(path.is_relative_to(output.resolve()))
                self.assertTrue(path.exists())
            with self.assertRaises((ValueError, FileExistsError)):
                run_rna(a, self.config(), output)
        self.assertEqual((a.layers['counts'] != original).nnz, 0)
        np.testing.assert_array_equal(a.X.toarray(), original.toarray())

    def test_overlapping_donor_split_rejected(self):
        from ftd_prediction.rna_training import run_rna
        cfg = self.config()
        cfg['test_donors'] = ['synthetic_train']
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                run_rna(self.fixture(), cfg, Path(root) / 'bad')


if __name__ == '__main__':
    unittest.main()
