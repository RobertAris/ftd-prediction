"""Invented sparse fixtures for explicit H5AD count and mapping semantics."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(n) for n in ('anndata', 'h5py', 'scipy'))


@unittest.skipUnless(AVAILABLE, 'Optional H5AD environment required')
class H5ADTests(unittest.TestCase):
    def fixture(self, path):
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        a = ad.AnnData(X=csr_matrix(np.full((8, 4), .5)),
            obs=pd.DataFrame({'donor': ['d1']*4+['d2']*4,
                              'section': ['s1']*4+['s2']*4,
                              'types': ['A','B']*4}, index=[f'c{i}' for i in range(8)]),
            var=pd.DataFrame(index=['g1','g2','g3','g4']))
        a.layers['counts'] = csr_matrix(np.arange(1,33).reshape(8,4))
        a.obsm['spatial'] = np.arange(16).reshape(8,2).astype(float)
        a.write_h5ad(path)
        return a

    def config(self):
        return dict(assay='rna', count_source='layers/counts',
                    counts_provenance={'kind':'measured_umi','evidence_ref':'synthetic-generator'},
                    mapping_verified=True, mapping_evidence_ref='synthetic-generator',
                    obs_columns={'donor_id':'donor','cell_type':'types'})

    def test_schema_reads_no_dataset_values(self):
        import h5py
        from ftd_prediction.h5ad_io import inspect_schema
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'a.h5ad'; self.fixture(path)
            with patch.object(h5py.Dataset,'__getitem__',side_effect=AssertionError('dataset read')):
                result=inspect_schema(path)
            self.assertEqual(result['matrices']['layers/counts']['shape'],[8,4])
            self.assertEqual(result['obs_columns'],['donor','section','types'])

    def test_explicit_sparse_counts_and_sampling(self):
        import numpy as np
        from scipy import sparse
        from ftd_prediction.h5ad_io import load_h5ad
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'a.h5ad'; original=self.fixture(path)
            cfg=self.config(); cfg['cell_cap_per_donor']=2
            a=load_h5ad(path,cfg); b=load_h5ad(path,cfg)
            self.assertTrue(sparse.isspmatrix_csr(a.layers['counts']))
            self.assertEqual(a.n_obs,4)
            self.assertEqual(a.obs_names.tolist(),b.obs_names.tolist())
            np.testing.assert_array_equal(a.layers['counts'].toarray(),original[a.obs_names].layers['counts'].toarray())

    def test_normalized_X_and_missing_provenance_rejected(self):
        from ftd_prediction.h5ad_io import load_h5ad
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'a.h5ad'; self.fixture(path)
            cfg=self.config(); cfg['count_source']='X'
            with self.assertRaises(ValueError): load_h5ad(path,cfg)
            cfg=self.config(); cfg['counts_provenance']={'kind':'sct','evidence_ref':'wrong'}
            with self.assertRaises(ValueError): load_h5ad(path,cfg)

    def test_query_donor_changes_do_not_change_training_sample(self):
        from ftd_prediction.h5ad_io import load_h5ad
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'a.h5ad'; original=self.fixture(path)
            cfg=self.config(); cfg['cell_cap_per_donor']=2
            before=load_h5ad(path,cfg)
            original=original[original.obs.donor=='d2'].copy()
            original.write_h5ad(path)
            after=load_h5ad(path,cfg)
            self.assertEqual(before[before.obs.donor_id=='d2'].obs_names.tolist(),after.obs_names.tolist())

    def test_raw_uses_its_own_gene_axis(self):
        from ftd_prediction.h5ad_io import load_h5ad
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'a.h5ad'; a=self.fixture(path)
            a.X=a.layers['counts'].copy(); a.raw=a.copy()
            a=a[:,:2].copy(); a.write_h5ad(path)
            cfg=self.config(); cfg['count_source']='raw/X'
            loaded=load_h5ad(path,cfg)
            self.assertEqual(loaded.var_names.tolist(),['g1','g2','g3','g4'])

    def test_spatial_mapping_and_random_thinning(self):
        from ftd_prediction.h5ad_io import load_h5ad
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'a.h5ad'; self.fixture(path)
            cfg=self.config(); cfg.update(assay='spatial',coordinate_units='um',spatial_key='spatial')
            cfg['obs_columns']['section_id']='section'
            self.assertEqual(load_h5ad(path,cfg).obsm['spatial'].shape,(8,2))
            cfg['cell_cap_per_donor']=2
            with self.assertRaises(ValueError): load_h5ad(path,cfg)

    def test_memory_bound_and_external_links_rejected(self):
        import h5py
        from ftd_prediction.h5ad_io import load_h5ad, inspect_schema
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'a.h5ad'; self.fixture(path)
            cfg=self.config(); cfg['max_matrix_bytes']=1
            with self.assertRaises(ValueError): load_h5ad(path,cfg)
            with h5py.File(path,'a') as f:
                del f['layers/counts']
                f['layers/counts']=h5py.ExternalLink('never-open.h5','X')
            with self.assertRaises(ValueError): inspect_schema(path)

    def test_private_sample_mapping_and_ambiguous_or_missing_keys(self):
        from ftd_prediction.h5ad_io import load_h5ad
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);path=root/'a.h5ad';self.fixture(path)
            mapping=root/'mapping.csv'
            mapping.write_text('sample_id,donor_id\nd1,person1\nd2,person1\n')
            cfg=self.config();cfg['obs_columns']={'sample_id':'donor','cell_type':'types'}
            cfg.update(mapping_file=str(mapping),mapping_keys=['sample_id'])
            a=load_h5ad(path,cfg)
            self.assertEqual(a.obs.donor_id.unique().tolist(),['person1'])
            mapping.write_text('sample_id,donor_id\nd1,person1\n')
            with self.assertRaisesRegex(ValueError,'cover'):load_h5ad(path,cfg)
            mapping.write_text('sample_id,donor_id\nd1,person1\nd1,person2\nd2,person1\n')
            with self.assertRaisesRegex(ValueError,'ambiguous'):load_h5ad(path,cfg)

    def test_spatial_mapping_uses_explicit_composite_key(self):
        from ftd_prediction.h5ad_io import load_h5ad
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);path=root/'a.h5ad';self.fixture(path)
            mapping=root/'mapping.csv'
            mapping.write_text('sample_id,acquisition_region,donor_id,section_id\nd1,s1,p1,physical1\nd2,s2,p2,physical2\n')
            cfg=self.config();cfg.update(assay='spatial',coordinate_units='um',spatial_key='spatial',mapping_file=str(mapping),mapping_keys=['sample_id','acquisition_region'])
            cfg['obs_columns']={'sample_id':'donor','acquisition_region':'section'}
            a=load_h5ad(path,cfg)
            self.assertEqual(a.obs.section_id.tolist(),['physical1']*4+['physical2']*4)
            mapping.write_text('sample_id,acquisition_region,donor_id,section_id\nd1,s1,p1,shared\nd2,s2,p2,shared\n')
            with self.assertRaisesRegex(ValueError,'multiple donors'):load_h5ad(path,cfg)


if __name__ == '__main__': unittest.main()
