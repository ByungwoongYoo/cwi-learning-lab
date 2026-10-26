"""Bounded unit/integration evidence. No quiz or user assessment."""
import hashlib, json, shutil, tempfile, unittest
from pathlib import Path
import numpy as np
from training_lab.common import ROOT, FormatError, digest, frozen_config, json_read, json_write
from training_lab.data import load_data, prepare, source_hashes, _image
from training_lab.train import read_checkpoint
from training_lab.verify import gradient_check, negative_tests
RUN=ROOT/'artifacts/technical_smoke'

class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c,cls.ch=frozen_config();cls.data,cls.dm=load_data(RUN/'data')
    def test_frozen_scope(self):
        self.assertEqual(self.c['original_ids'],[4,6]);self.assertEqual(self.c['partition'],'train')
        self.assertEqual(self.c['architecture'],[7,16,1]);self.assertEqual(len(source_hashes()),8)
    def test_sample_ids_fixed_unique(self):
        idx=self.data['indices'];self.assertEqual(idx.shape,(8192,2))
        self.assertEqual(len(np.unique(idx,axis=0)),8192)
    def test_scaler_sample_only(self):
        np.testing.assert_allclose(self.data['x'].mean(0),0,atol=1e-12)
        np.testing.assert_allclose(self.data['x'].std(0),1,atol=1e-12)
    def test_targets_equal_original_mask_samples(self):
        for j,oid in enumerate([4,6]):
            gt=_image(ROOT/f'examples/public_train/original_{oid:03d}_40kV_mask.tiff',True)
            idx=self.data['indices'][j*4096:(j+1)*4096,1]
            np.testing.assert_array_equal(self.data['y'][j*4096:(j+1)*4096,0],gt.ravel()[idx])
    def test_class_imbalance_explicit(self):
        self.assertEqual(int(self.data['y'].sum()),1)
        summary=json_read(RUN/'uninterrupted/summary.json')
        self.assertEqual(summary['train_confusion'],{'TP':0,'FN':1,'TN':8191,'FP':0})
    def test_prepare_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as t:
            m=prepare(Path(t)/'data')
            self.assertEqual(m,self.dm)
    def test_gradient_all_parameters_synthetic(self):
        r=gradient_check();self.assertTrue(r['passed']);self.assertEqual(r['parameters_checked'],145)
    def test_partial_checkpoint_mid_epoch(self):
        m,a=read_checkpoint(RUN/'interrupted/checkpoint',self.c,self.ch,self.dm)
        self.assertEqual(m['progress'],{'epoch':1,'cursor':1280,'step':37})
        self.assertEqual(len(m['losses']),37)
    def test_final_state_exact_equality(self):
        m,a=read_checkpoint(RUN/'uninterrupted/checkpoint',self.c,self.ch,self.dm)
        n,b=read_checkpoint(RUN/'resumed/checkpoint',self.c,self.ch,self.dm)
        self.assertEqual(m,n)
        for k in a: np.testing.assert_array_equal(a[k],b[k])
    def test_distinct_processes(self):
        sums=[json_read(RUN/p/'summary.json') for p in ['uninterrupted','interrupted','resumed']]
        self.assertEqual(len({s['pid'] for s in sums}),3)
        self.assertEqual([s['optimizer_steps_this_process'] for s in sums],[128,37,91])
    def test_corrupt_checkpoint_rejections(self):
        cases=negative_tests(RUN/'interrupted/checkpoint',self.c,self.ch,self.dm)
        self.assertGreaterEqual(len(cases),20);self.assertTrue(all(c['rejected'] for c in cases))
    def test_data_schema_rejects_wrong_counts(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'data';shutil.copytree(RUN/'data',p)
            m=json_read(p/'manifest.json');m['counts']['4']['samples']=True;json_write(p/'manifest.json',m)
            with self.assertRaises(FormatError):load_data(p)
    def test_loss_finite_not_performance_claim(self):
        r=json_read(RUN/'verification.json');self.assertTrue(r['parity']['passed'])
        self.assertIn('generalization',r['claims_excluded'])
        for s in r['runs'].values():self.assertTrue(s['all_training_losses_finite'])
    def test_original_model_freeze_unchanged_when_present(self):
        p=ROOT.parent/'cwi_learning_lab/cwi_lab/models.json'
        if not p.exists(): self.skipTest('Stage1 separate archive not present, expected in standalone package')
        self.assertEqual(digest(p.read_bytes()),'0b6bce8f46b677c834048994c3ea72cba26e4d6a4172cd88018190ed7cbb9416')
if __name__=='__main__':unittest.main()
