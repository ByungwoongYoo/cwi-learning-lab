import unittest,torch,numpy as np
from tradeoff_lab.experiment import dataset,model,metrics,plan
class Tests(unittest.TestCase):
 def test_plan(self): self.assertEqual(plan()['batch_sizes'],[1,32,256])
 def test_shape(self): x,y=dataset(100,101);self.assertEqual(x.shape,(100,64));self.assertEqual(y.dtype,torch.int64)
 def test_repeat(self): a,b=dataset(100,101);c,d=dataset(100,101);self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(b,d))
 def test_split(self): a,b=dataset(100,101);c,d=dataset(100,202);self.assertFalse(torch.equal(a,c))
 def test_model(self): self.assertEqual(model()(torch.zeros(2,64)).shape,(2,2))
 def test_finite(self): x,y=dataset(100,303);self.assertTrue(torch.isfinite(x).all())
 def test_metrics(self): self.assertEqual(metrics(torch.tensor([[5.,-5.],[-5.,5.]]),torch.tensor([0,1]))['accuracy'],1.)
if __name__=='__main__':unittest.main()
