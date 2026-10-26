from .store import *
import subprocess,json,hashlib
import numpy as np

def file_hashes(p): return {str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest() for x in p.rglob('*') if x.is_file()}

def setup_run(name):
    root=run_root(name,True);(root/'generations').mkdir();(root/'work').mkdir()
    shutil.copytree(PROJECT/'vendor/stage2/artifacts/technical_smoke/data',root/'data')
    return root

def compare(a,b,root):
    ma,aa=validate(a,root);mb,ab=validate(b,root)
    eq={k:bool(np.array_equal(aa[k],ab[k])) for k in aa}
    return {'metadata_equal':ma==mb,'arrays_equal':eq,'passed':ma==mb and all(eq.values())}

def verify(name):
    started=time.monotonic();root=setup_run(name);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
    records=[];faults=[];resumes=[]
    def child(label,args,env_extra=None,kill_at=None):
        if time.monotonic()-started>540: raise StoreError('540-second experiment guard')
        out=root/'work'/label;out.mkdir()
        with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
            p=subprocess.Popen([sys.executable]+args,cwd=PROJECT,env={**env,**(env_extra or {})},stdout=stdout,stderr=stderr)
            try:
                if kill_at:
                    deadline=time.monotonic()+40
                    while not kill_at.exists():
                        if p.poll() is not None: raise StoreError('Writer exited before requested hook')
                        if time.monotonic()>deadline: raise TimeoutError('Hook wait limit')
                        time.sleep(.02)
                    p.kill()
                code=p.wait(timeout=40)
            finally:
                if p.poll() is None: p.kill();p.wait(timeout=5)
        records.append({'label':label,'argv':args,'pid':p.pid,'returncode':code})
        if code!=(-9 if kill_at else 0): raise StoreError('Child failed: '+label)
    def train(label,stop,resume=None):
        out=root/'work'/(label+'_output')
        args=['-m','recovery_lab','_train','--run',name,'--output',label+'_output','--stop',str(stop)]
        if resume: args+=['--resume',resume.name]
        child(label,args)
        return out/'checkpoint'
    ref=train('reference',128)
    cp32=train('first32',32);commit(root,cp32)
    prev=file_hashes(root/'generations/g000032')
    # One logical interrupted trajectory, with two prepublication crashes and one postpublication crash.
    for i,(step,fault) in enumerate([(37,'array_closed'),(48,'validated'),(64,None),(96,'published'),(128,None)]):
        selected,report=latest(root);resumes.append({'target_step':step,'selected_step':report['selected']['step'],'report':report})
        checkpoint=train('segment'+str(step),step,selected)
        if fault:
            marker=root/'work'/('marker'+str(step));args=['-m','recovery_lab','_commit','--run',name,'--source','segment'+str(step)+'_output']
            child('writer'+str(step),args,{'RECOVERY_FAULT_STAGE':fault,'RECOVERY_FAULT_MARKER':str(marker)},marker)
            faults.append({'step':step,'hook':fault,'observed':marker.read_text(),'after':inspect(root)})
        else: commit(root,checkpoint)
        if step==64:
            # Controlled corruption only inside this run; preserve corrupt generation, do not delete.
            damaged=root/'generations/g000064/state.npz';raw=bytearray(damaged.read_bytes());raw[len(raw)//2]^=1;damaged.write_bytes(raw)
    if [q['selected_step'] for q in resumes]!=[32,32,32,32,96]: raise StoreError('Unexpected recovery selection trajectory')
    if [q['after']['selected']['step'] for q in faults]!=[32,32,96]: raise StoreError('Unexpected post-kill state visibility')
    selected,report=latest(root)
    if report['selected']['step']!=128: raise StoreError('Final generation must be128')
    parity=compare(ref,selected,root)
    if not parity['passed']: raise StoreError('Final state parity failure')
    if file_hashes(root/'generations/g000032')!=prev: raise StoreError('Previous checkpoint changed')
    # Stored checkpoint equality implies identical evaluation state, but evaluate scores explicitly as well.
    from training_lab.train import make_model
    import torch
    _,dm=load_data(root/'data');data,_=load_data(root/'data');pred=[]
    for cp in (ref,selected):
        _,st=validate(cp,root);model=make_model()
        with torch.no_grad():
            for j,param in enumerate(model.parameters()): param.copy_(torch.from_numpy(st[f'p{j}']))
            model.eval();pred.append(torch.sigmoid(model(torch.from_numpy(data['x']))).numpy())
    predictions_equal=bool(np.array_equal(*pred));assert predictions_equal
    result={'status':'PASS_SINGLE_PROCESS_CPU_RECOVERY','run':name,'elapsed_seconds':time.monotonic()-started,'parity':parity,'predictions_equal':predictions_equal,'resumes':resumes,'faults':faults,'final_store':report,'commands':records,'previous32_unchanged':True,'limitations':['same runtime CPU toy model','no power-loss/fsync durability proof','no concurrent writers','no authenticity signatures','no detection improvement','AI assisted, user competence not assessed']}
    (root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');return result
