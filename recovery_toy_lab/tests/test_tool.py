import unittest, tempfile, os, json, subprocess, sys
from pathlib import Path
from unittest.mock import Mock
from recovery_tool.data import generated,prepare,load_data
from recovery_tool.common import FormatError, json_read
from recovery_tool import store
from recovery_tool.supervisor import create,observed,supervise,validate_marker
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.old=store.RUNS;store.RUNS=Path(self.t.name)
 def tearDown(self):store.RUNS=self.old;self.t.cleanup()
 def test_balanced_deterministic(self):
  import numpy as np
  a=generated();b=generated();self.assertEqual(a['y'].sum(),4096)
  for k in a:self.assertTrue(np.array_equal(a[k],b[k]))
 def test_prepare_reload(self):
  root=Path(self.t.name)/'data';prepare(root);a,m=load_data(root);self.assertEqual(m['positive'],4096)
 def test_tamper_data(self):
  root=Path(self.t.name)/'data';prepare(root);(root/'data.npz').write_bytes(b'bad')
  with self.assertRaises(FormatError):load_data(root)
 def test_existing_run_no_overwrite(self):
  root=create('a');marker=root/'keep';marker.write_text('same')
  with self.assertRaises(FileExistsError):create('a')
  self.assertEqual(marker.read_text(),'same')
 def test_run_names(self):
  for name in ['../x','/tmp','a/b','',True]:
   with self.assertRaises(store.StoreError):store.run_root(name,True)
 def test_symlink_run(self):
  (store.RUNS/'link').symlink_to('/tmp')
  with self.assertRaises(store.StoreError):store.run_root('link')
 def test_empty(self):self.assertEqual(observed(create('a'))[0],0)
 def test_invalid_partial(self):
  root=create('a');(root/'generations/.partial').mkdir()
  with self.assertRaisesRegex(store.StoreError,'NO_VALID'):supervise(root)
 def test_limits_and_plan(self):
  root=create('a')
  for kw in [{'max_launches':True},{'max_stagnant':0},{'plan':[{'hook':'bad','step':1}]},{'plan':[{'hook':'published','step':37}]}]:
   with self.assertRaises(store.StoreError):supervise(root,**kw)
 def test_marker(self):
  p=Mock(pid=123);p.poll.return_value=None;f={'hook':'after_backward','step':37};r={'pid':123,'token':'x','attempt':'a','kind':'paused_fault','hook':f['hook'],'target_step':37}
  validate_marker(r,p,'x','a',f)
  with self.assertRaises(store.StoreError):validate_marker(r,p,'wrong','a',f)
 def test_duplicate_json(self):
  q=Path(self.t.name)/'j';q.write_text('{"x":1,"x":2}')
  with self.assertRaises(FormatError):json_read(q)
 def test_json_nonfinite(self):
  q=Path(self.t.name)/'j';q.write_text('{"x":NaN}')
  with self.assertRaises(FormatError):json_read(q)
if __name__=='__main__':unittest.main()
