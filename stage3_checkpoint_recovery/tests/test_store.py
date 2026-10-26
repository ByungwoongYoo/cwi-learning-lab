import unittest,json,uuid,shutil,os,hashlib
from pathlib import Path
from recovery_lab.store import *
from recovery_lab.experiment import setup_run
class StoreTests(unittest.TestCase):
    def setUp(self):
        self.root=setup_run('test_'+uuid.uuid4().hex[:12]);self.source=self.root/'work/source/checkpoint'
        shutil.copytree(PROJECT/'vendor/stage2/artifacts/technical_smoke/interrupted/checkpoint',self.source)
    def seed(self): return commit(self.root,self.source)
    def invalid(self):
        report=inspect(self.root);self.assertIsNone(report['selected'])
        with self.assertRaises(StoreError): latest(self.root)
    def test_empty_fails_closed(self): self.invalid()
    def test_good_commit_and_select(self):
        p=self.seed();selected,report=latest(self.root);self.assertEqual(p,selected);self.assertEqual(report['selected']['step'],37)
    def test_duplicate_never_overwrites(self):
        p=self.seed();before=(p/'state.npz').read_bytes()
        with self.assertRaises(StoreError): commit(self.root,self.source)
        self.assertEqual(before,(p/'state.npz').read_bytes())
    def test_wrong_step_name(self):
        p=self.seed();p.rename(p.parent/'g000064');self.invalid()
    def test_partial_manifest(self):
        p=self.seed();(p/'manifest.json').unlink();self.invalid()
    def test_corrupt_archive(self):
        p=self.seed();(p/'state.npz').write_bytes(b'bad');self.invalid()
    def test_nan_json(self):
        p=self.seed();m=json.loads((p/'manifest.json').read_text());m['initial_loss']=float('nan');(p/'manifest.json').write_text(json.dumps(m));self.invalid()
    def test_duplicate_json_key(self):
        p=self.seed();f=p/'manifest.json';f.write_text(f.read_text().replace('{','{"schema_version":1,',1));self.invalid()
    def test_schema_boolean(self):
        p=self.seed();m=json.loads((p/'manifest.json').read_text());m['schema_version']=True;(p/'manifest.json').write_text(json.dumps(m));self.invalid()
    def test_oversized_manifest(self):
        p=self.seed();(p/'manifest.json').write_bytes(b' '*(131073));self.invalid()
    def test_oversized_archive(self):
        p=self.seed();(p/'state.npz').write_bytes(b'0'*(4*1024*1024+1));self.invalid()
    def test_unexpected_member(self):
        p=self.seed();(p/'extra').write_text('x');self.invalid()
    def test_checkpoint_symlink(self):
        p=self.root/'generations/g000037';p.symlink_to(self.source,target_is_directory=True);self.invalid()
    def test_member_symlink(self):
        p=self.seed();f=p/'state.npz';f.unlink();f.symlink_to(self.source/'state.npz');self.invalid()
    def test_hardlink(self):
        p=self.seed();f=p/'state.npz';f.unlink();os.link(self.source/'state.npz',f);self.invalid()
    def test_pending_not_selected(self):
        shutil.copytree(self.source,self.root/'generations/.pending-test');self.invalid()
    def test_noncanonical_alias_not_selected(self):
        shutil.copytree(self.source,self.root/'generations/g37');self.invalid()
    def test_generation_count_bound(self):
        for i in range(101): (self.root/'generations'/str(i)).mkdir()
        with self.assertRaises(StoreError): inspect(self.root)
    def test_traversal_names(self):
        for name in ['../x','/tmp/x','a/b','a\\b','.', '..','x'*49,'']:
            with self.subTest(name=name),self.assertRaises(StoreError): run_root(name)
    def test_outside_path(self):
        with self.assertRaises(StoreError): checked_path(Path('/tmp/outside'))
    def test_run_symlink(self):
        link=RUNS/('symlink_'+uuid.uuid4().hex[:12]);link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(StoreError): run_root(link.name)
    def test_generations_symlink(self):
        (self.root/'generations').rmdir();(self.root/'generations').symlink_to(self.root/'work',target_is_directory=True)
        with self.assertRaises(StoreError): inspect(self.root)
    def test_data_symlink(self):
        f=self.root/'data/data.npz';f.unlink();f.symlink_to(PROJECT/'vendor/stage2/artifacts/technical_smoke/data/data.npz')
        with self.assertRaises(StoreError): commit(self.root,self.source)
    def test_dry_run_preserves_bytes(self):
        p=self.seed();before=(p/'state.npz').read_bytes();r=inspect(self.root);self.assertEqual(r['cleanup'],'DRY_RUN_ONLY_NO_FILES_DELETED');self.assertEqual(before,(p/'state.npz').read_bytes())
if __name__=='__main__':unittest.main()
