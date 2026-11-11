"""Synthetic boundary/provenance tests independent of optional model training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(n) for n in ('anndata', 'h5py', 'scipy'))


@unittest.skipUnless(AVAILABLE, 'Optional H5AD environment required')
class NeuralRunnerTests(unittest.TestCase):
    def fixture(self, root, spatial=False):
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        source = root/'input.h5ad'
        matrix = csr_matrix(np.arange(1,33).reshape(8,4))
        a=ad.AnnData(X=matrix.copy(),obs=pd.DataFrame({
            'donor':['d1']*4+['d2']*4,'section':['s1']*4+['s2']*4,'type':['A','B']*4,
        },index=[f'c{i}' for i in range(8)]),var=pd.DataFrame(index=['g1','g2','g3','g4']))
        a.layers['counts']=matrix.copy(); a.obsm['spatial']=np.arange(16).reshape(8,2).astype(float)
        a.write_h5ad(source)
        cfg=dict(schema_version=1,data_kind='synthetic',assay='spatial' if spatial else 'rna',
            input_file=str(source),canonical_input_roots=[],count_source='layers/counts',
            counts_provenance=dict(kind='measured_umi',evidence_ref='synthetic-generator'),
            mapping_verified=True,mapping_evidence_ref='synthetic-generator',
            obs_columns=dict(donor_id='donor',cell_type='type'),
            split_strategy='leave_one_donor_out',model=dict(seed=11,max_epochs=2))
        if spatial:
            cfg.update(coordinate_units='um',spatial_key='spatial')
            cfg['obs_columns']['section_id']='section'
        path=root/'config.json';path.write_text(json.dumps(cfg))
        return path,cfg

    def fake_backend(self,adata,config,output):
        output.mkdir()
        (output/'synthetic-check.txt').write_text('deterministic fixture only; no model training')
        self.assertFalse(set(config['train_donors']) & set(config['test_donors']))
        return dict(status='completed',method='test_fixture',train_donors=config['train_donors'],
                    test_donors=config['test_donors'],artifacts={'fixture':'synthetic-check.txt'})

    def test_fold_and_complete_private_input_provenance(self):
        import pandas as pd
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root,spatial=True)
            out=execute(path,root/'out',mode='synthetic',execution_context='local',backend=self.fake_backend)
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['status'],'completed')
            self.assertEqual(record['input_sha256'],hashlib.sha256(Path(cfg['input_file']).read_bytes()).hexdigest())
            self.assertIn('anndata',record['environment']['packages'])
            self.assertTrue(record['code']['source_sha256'])
            mapping=pd.read_csv(out/'private_cell_mapping.csv')
            self.assertEqual(mapping.cell_id.tolist(),[f'c{i}' for i in range(8)])
            self.assertEqual(mapping.x_um.tolist(),list(range(0,16,2)))
            splits=json.loads((out/'private_splits.json').read_text())
            self.assertEqual(len(splits),2)
            for artifact in record['artifacts']:
                self.assertEqual(hashlib.sha256((out/artifact['path']).read_bytes()).hexdigest(),artifact['sha256'])
            with self.assertRaises(FileExistsError):
                execute(path,out,mode='synthetic',execution_context='local',backend=self.fake_backend)

    def test_study_scope_rejected_before_data_read(self):
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            cfg['data_kind']='study';path.write_text(json.dumps(cfg))
            with patch('ftd_prediction.neural.load_h5ad') as loader:
                with self.assertRaises(ValueError):
                    execute(path,root/'out',mode='study',execution_context='local',backend=self.fake_backend)
                loader.assert_not_called()
            self.assertFalse((root/'out').exists())

    def test_source_outside_canonical_root_rejected(self):
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            cfg.update(data_kind='study',canonical_input_roots=[str(root/'different')]);path.write_text(json.dumps(cfg))
            with patch.dict('os.environ',{'LSB_JOBID':'123'}),patch('ftd_prediction.neural.load_h5ad') as loader:
                with self.assertRaises(ValueError): execute(path,root/'out',mode='study',backend=self.fake_backend)
                loader.assert_not_called()

    def test_configuration_mutation_retains_failed_record(self):
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            def changing(a,c,o):
                result=self.fake_backend(a,c,o)
                path.write_text(path.read_text()+' ')
                return result
            with self.assertRaises(ValueError):
                execute(path,root/'out',mode='synthetic',execution_context='local',backend=changing)
            self.assertEqual(json.loads((root/'out/run.json').read_text())['status'],'failed')
            self.assertFalse((root/'out/summary.json').exists())

    def test_invalid_mapping_does_not_fit(self):
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            cfg['mapping_verified']=False;path.write_text(json.dumps(cfg))
            with patch('ftd_prediction.neural.load_h5ad', wraps=__import__('ftd_prediction.h5ad_io',fromlist=['load_h5ad']).load_h5ad):
                with self.assertRaises(ValueError):
                    execute(path,root/'out',mode='synthetic',execution_context='local',backend=lambda *a:self.fail('fit called'))
            self.assertEqual(json.loads((root/'out/run.json').read_text())['status'],'failed')

    def test_prespecified_fold_and_no_synthetic_fallback(self):
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            cfg['fold_indices']=[1];path.write_text(json.dumps(cfg))
            out=execute(path,root/'out',mode='synthetic',execution_context='local',backend=self.fake_backend)
            self.assertEqual(json.loads((out/'private_splits.json').read_text())[0]['test_donors'],['d2'])
            cfg.update(data_kind='study',model=dict(mode='synthetic_smoke'))
            path.write_text(json.dumps(cfg))
            with patch.dict('os.environ',{'LSB_JOBID':'123'}):
                with self.assertRaises(ValueError): execute(path,root/'bad',mode='study',backend=self.fake_backend)

    def test_version_failure_precedes_data_read_and_has_private_traceback(self):
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            with patch('ftd_prediction.neural._validate_environment',side_effect=ValueError('test-version-mismatch')),patch('ftd_prediction.neural.load_h5ad') as loader:
                with self.assertRaises(ValueError):
                    execute(path,root/'out',mode='synthetic',execution_context='local')
                loader.assert_not_called()
            self.assertIn('test-version-mismatch',(root/'out/private_execution.log').read_text())
            self.assertEqual(json.loads((root/'out/run.json').read_text())['status'],'failed')

    def test_descriptive_strategy_rejects_requested_holdout_fold(self):
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            cfg.update(split_strategy='all_donor_descriptive',fold_indices=[1])
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'Fold indices'):
                execute(path,root/'out',mode='synthetic',execution_context='local',
                        backend=lambda *a:self.fail('fit called'))
            self.assertEqual(json.loads((root/'out/run.json').read_text())['status'],'failed')

    def test_private_logging_releases_import_created_handlers(self):
        import io
        import logging
        import sys
        from ftd_prediction.neural import _private_logging
        console=io.StringIO()
        old=logging.getLogger('ftd_old_console')
        new=logging.getLogger('ftd_import_console')
        existing=logging.StreamHandler(console)
        old.addHandler(existing)
        try:
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'private.log'
                with _private_logging(path):
                    created=logging.StreamHandler(sys.stderr)
                    new.addHandler(created)
                    old.warning('invented_private_old')
                    new.warning('invented_private_new')
                self.assertEqual(console.getvalue(),'')
                self.assertIs(existing.stream,console)
                self.assertFalse(created.stream.closed)
                self.assertIn('invented_private_new',path.read_text())
                self.assertIn('invented_private_old',path.read_text())
        finally:
            old.removeHandler(existing)
            for handler in new.handlers[:]:
                new.removeHandler(handler)

    def test_mapping_hash_recorded_and_mutation_fails(self):
        from ftd_prediction.neural import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            mapping=root/'mapping.csv';mapping.write_text('sample_id,donor_id\nd1,p1\nd2,p2\n')
            cfg.update(mapping_file=str(mapping),mapping_keys=['sample_id'],obs_columns={'sample_id':'donor'})
            path.write_text(json.dumps(cfg))
            out=execute(path,root/'good',mode='synthetic',execution_context='local',backend=self.fake_backend)
            inputs=json.loads((out/'run.json').read_text())['input_files']
            self.assertEqual(len(inputs),2)
            self.assertIn(hashlib.sha256(mapping.read_bytes()).hexdigest(),[x['sha256'] for x in inputs])
            def mutate(a,c,o):
                result=self.fake_backend(a,c,o)
                mapping.write_text(mapping.read_text()+'\n')
                return result
            with self.assertRaisesRegex(ValueError,'changed'):
                execute(path,root/'bad',mode='synthetic',execution_context='local',backend=mutate)
            self.assertEqual(json.loads((root/'bad/run.json').read_text())['status'],'failed')

    def test_mapping_symlink_retarget_cannot_change_loaded_crosswalk(self):
        from ftd_prediction.neural import execute
        from ftd_prediction.h5ad_io import load_h5ad
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture(root)
            original=root/'original.csv';original.write_text('sample_id,donor_id\nd1,p1\nd2,p2\n')
            other=root/'other.csv';other.write_text('sample_id,donor_id\nd1,wrong1\nd2,wrong2\n')
            link=root/'link.csv';link.symlink_to(original)
            cfg.update(mapping_file=str(link),mapping_keys=['sample_id'],obs_columns={'sample_id':'donor'})
            path.write_text(json.dumps(cfg))
            def retarget(source,loader_cfg):
                link.unlink();link.symlink_to(other)
                return load_h5ad(source,loader_cfg)
            with patch('ftd_prediction.neural.load_h5ad',side_effect=retarget):
                out=execute(path,root/'out',mode='synthetic',execution_context='local',backend=self.fake_backend)
            splits=json.loads((out/'private_splits.json').read_text())
            self.assertEqual(splits[0]['train_donors'],['p2'])
            self.assertEqual(splits[0]['test_donors'],['p1'])
            inputs=json.loads((out/'run.json').read_text())['input_files']
            self.assertIn(str(original.resolve()),[x['path'] for x in inputs])


if __name__ == '__main__': unittest.main()
