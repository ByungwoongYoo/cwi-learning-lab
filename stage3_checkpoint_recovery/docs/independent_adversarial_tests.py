import sys,json,shutil,hashlib,uuid,zipfile,io,os,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
if not (P/'recovery_lab').is_dir(): P=P/'cwi_checkpoint_recovery_stage3'
sys.path.insert(0,str(P))
from recovery_lab.store import *
from recovery_lab.experiment import setup_run
OUT=Path(__file__).parent
results=[];started=time.monotonic()
S=P/'vendor/stage2/artifacts/technical_smoke/interrupted/checkpoint'
F=P/'vendor/stage2/artifacts/technical_smoke/uninterrupted/checkpoint'
def copy(root,src=S,name='g000037'):
 p=root/'generations'/name;shutil.copytree(src,p);return p
def meta(p,fn):
 f=p/'manifest.json';m=json.loads(f.read_text());fn(m);f.write_text(json.dumps(m))
def rewrite(p,arrays):
 np.savez(p/'state.npz',**arrays);rehash(p)
def rehash(p):meta(p,lambda m:m.update(archive_sha256=hashlib.sha256((p/'state.npz').read_bytes()).hexdigest()))
def arrays(p,fn):
 with np.load(p/'state.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 fn(a);rewrite(p,a)
def rejected(name,mut):
 root=setup_run('audit_'+uuid.uuid4().hex[:12]);p=copy(root);mut(p)
 try:r=inspect(root);ok=r['selected'] is None;detail=r
 except Exception as e:ok=False;detail={'uncaught':type(e).__name__,'error':str(e)}
 results.append({'case':name,'passed':ok,'detail':detail})
for key,value in [('schema_version',True),('schema_version',2),('progress',{'step':37,'epoch':1,'cursor':0}),('progress',{'step':True,'epoch':0,'cursor':256}),('config_sha256','0'*64),('data_sha256','0'*64),('sample_indices_sha256','0'*64),('runtime',{}),('optimizer',{}),('losses',[1]*36),('initial_loss',-1),('sampler_rng',{}),('archive_sha256','0'*64)]:
 rejected('meta_'+key+'_'+str(value)[:20],lambda p,k=key,v=value:meta(p,lambda m:m.update({k:v})))
rejected('extra_manifest_key',lambda p:meta(p,lambda m:m.update(extra=1)))
rejected('duplicate_json',lambda p:(p/'manifest.json').write_text((p/'manifest.json').read_text().replace('{','{"schema_version":1,',1)))
rejected('nonfinite_json',lambda p:meta(p,lambda m:m.update(initial_loss=float('nan'))))
for name,key,fn in [('shape','p0',lambda a:a[:1]),('dtype','p0',lambda a:a.astype('float32')),('object','p0',lambda a:a.astype(object)),('nan','p0',lambda a:np.full_like(a,np.nan)),('permutation','permutation',lambda a:np.zeros_like(a)),('rng','torch_rng',lambda a:np.zeros_like(a))]:
 rejected('array_'+name,lambda p,k=key,f=fn:arrays(p,lambda a:a.update({k:f(a[k])})))
rejected('extra_npz_member',lambda p:arrays(p,lambda a:a.update(extra=np.zeros(1))))
def zip_mut(p,kind):
 f=p/'state.npz';raw=f.read_bytes()
 with zipfile.ZipFile(io.BytesIO(raw)) as z:entries=[(x.filename,z.read(x)) for x in z.infolist()]
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for n,data in entries:
   if kind=='path' and n=='p0.npy':n='../p0.npy'
   if kind=='bomb' and n=='p0.npy':data=b'0'*(4*1024*1024+1)
   z.writestr(n,data)
  if kind=='duplicate':z.writestr(entries[0][0],entries[0][1])
 f.write_bytes(b.getvalue());rehash(p)
for k in ['path','bomb','duplicate']:rejected('zip_'+k,lambda p,k=k:zip_mut(p,k))
rejected('manifest_symlink',lambda p:((p/'manifest.json').unlink(),(p/'manifest.json').symlink_to(S/'manifest.json')))
rejected('state_symlink',lambda p:((p/'state.npz').unlink(),(p/'state.npz').symlink_to(S/'state.npz')))
rejected('extra_file',lambda p:(p/'extra').write_text('bad'))
rejected('missing_state',lambda p:(p/'state.npz').unlink())
rejected('filename_step_mismatch',lambda p:p.rename(p.parent/'g000128'))
root=setup_run('audit_select_'+uuid.uuid4().hex[:8]);old=copy(root);new=copy(root,F,'g000128');os.utime(old,(2000000000,2000000000));selected,r=latest(root)
results.append({'case':'validated_step_over_mtime','passed':selected==new,'detail':r})
(new/'state.npz').write_bytes(b'broken');selected,r=latest(root)
results.append({'case':'corrupt_latest_falls_back','passed':selected==old,'detail':r})
(old/'manifest.json').write_text('{}')
try:latest(root);ok=False
except StoreError:ok=True
results.append({'case':'all_invalid_fails_closed','passed':ok})
root=setup_run('audit_dup_'+uuid.uuid4().hex[:8]);source=root/'work/source';shutil.copytree(S,source);cp=commit(root,source);before=(cp/'state.npz').read_bytes();arrays(source,lambda a:a.update(p0=a['p0']+1))
try:commit(root,source);ok=False
except StoreError:ok=True
results.append({'case':'same_step_conflict_no_overwrite','passed':ok and before==(cp/'state.npz').read_bytes()})
result={'cases':results,'count':len(results),'passed':all(x['passed'] for x in results),'elapsed':time.monotonic()-started}
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({'count':len(results),'passed':result['passed'],'failed':[x for x in results if not x['passed']],'elapsed':result['elapsed']},indent=2))
