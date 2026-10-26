import unittest,uuid,shutil,json
from unittest.mock import Mock
from recovery_lab import store
from live_recovery.supervisor import create,supervise,observed,validate_marker

class SupervisorTests(unittest.TestCase):
    def root(self):
        r=create('test_'+uuid.uuid4().hex[:12]);self.addCleanup(shutil.rmtree,r);return r
    def cp(self,r,step=128):
        source=store.PROJECT/'vendor/stage2/artifacts/technical_smoke'/('uninterrupted' if step==128 else 'interrupted')/'checkpoint'
        return shutil.copytree(source,r/'generations'/f'g{step:06d}')
    def test_empty_start(self): self.assertEqual(observed(self.root())[0],0)
    def test_terminal_zero_launches(self):
        r=self.root();self.cp(r);self.assertEqual(supervise(r)['launches'],[])
    def test_all_invalid_fail_closed(self):
        r=self.root();cp=self.cp(r);(cp/'state.npz').write_bytes(b'broken')
        with self.assertRaisesRegex(store.StoreError,'NO_VALID_CHECKPOINT'): supervise(r)
    def test_partial_only_fail_closed(self):
        r=self.root();(r/'generations/.pending-only').mkdir()
        with self.assertRaisesRegex(store.StoreError,'NO_VALID_CHECKPOINT'): supervise(r)
    def test_plan_limits(self):
        r=self.root()
        for kwargs in ({'max_launches':0},{'max_launches':True},{'max_launches':13},{'max_stagnant':0},{'max_stagnant':True},{'plan':[{'hook':'arbitrary','step':1}]},{'plan':[{'hook':'after_backward','step':True}]},{'plan':[{'hook':'after_backward','step':129}]}):
            with self.subTest(kwargs=kwargs),self.assertRaises(store.StoreError): supervise(r,**kwargs)
    def test_impossible_checkpoint_hooks(self):
        r=self.root()
        for hook in ['before_commit','array_closed','published']:
            with self.subTest(hook=hook),self.assertRaisesRegex(store.StoreError,'multiple'):supervise(r,[{'hook':hook,'step':37}])
    def test_unreachable_plan(self):
        r=self.root();self.cp(r,37)
        with self.assertRaisesRegex(store.StoreError,'Unreachable'): supervise(r,[{'hook':'after_backward','step':37}])
    def test_marker_identity(self):
        p=Mock(pid=123);p.poll.return_value=None;fault={'hook':'after_backward','step':37}
        row={'pid':123,'token':'a','attempt':'x','kind':'paused_fault','hook':'after_backward','target_step':37}
        validate_marker(row,p,'a','x',fault)
        for key,value in [('pid',124),('token','b'),('attempt','y'),('kind','finished'),('hook','published'),('target_step',38)]:
            with self.subTest(key=key),self.assertRaises(store.StoreError): validate_marker({**row,key:value},p,'a','x',fault)
        p.poll.return_value=0
        with self.assertRaises(store.StoreError): validate_marker(row,p,'a','x',fault)
    def test_real_restart_cap(self):
        r=self.root()
        with self.assertRaisesRegex(store.StoreError,'RESTART_LIMIT'): supervise(r,[{'hook':'after_backward','step':37}],max_launches=1)
        self.assertEqual(observed(r)[0],32)
    def test_real_no_progress_limit(self):
        r=self.root();self.cp(r,37)
        with self.assertRaisesRegex(store.StoreError,'NO_PROGRESS_LIMIT'): supervise(r,[{'hook':'after_backward','step':38}]*2,max_stagnant=2)
        self.assertEqual(observed(r)[0],37)
    def test_symlink_run_rejected(self):
        r=self.root();name='link_'+uuid.uuid4().hex[:12];link=store.RUNS/name;link.symlink_to(r);self.addCleanup(link.unlink)
        with self.assertRaises(store.StoreError):store.run_root(name)
    def test_traversal_run_rejected(self):
        for name in ['../outside','/tmp','x/y','',None]:
            with self.subTest(name=name),self.assertRaises(store.StoreError):store.run_root(name)
if __name__=='__main__':unittest.main()
