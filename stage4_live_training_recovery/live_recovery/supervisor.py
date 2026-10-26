"""Bounded parent of actual training workers. No shell, PID search or process groups."""
import json,os,subprocess,sys,time,uuid,hashlib,shutil
import numpy as np
from recovery_lab import store
PLAN=[{'hook':'after_backward','step':37},{'hook':'after_optimizer','step':48},{'hook':'before_commit','step':64},{'hook':'array_closed','step':96},{'hook':'published','step':128}]
ENV={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONHASHSEED':'0'}

def create(name):
    root=store.run_root(name,True);(root/'generations').mkdir();(root/'work').mkdir()
    shutil.copytree(store.PROJECT/'vendor/stage2/artifacts/technical_smoke/data',root/'data');return root

def observed(root):
    report=store.inspect(root)
    if report['selected']: return report['selected']['step'],report
    if report['invalid']: raise store.StoreError('NO_VALID_CHECKPOINT: refusing silent reset')
    return 0,report

def validate_marker(row,p,token,attempt,fault):
    if type(row) is not dict or row.get('pid')!=p.pid or row.get('token')!=token or row.get('attempt')!=attempt or row.get('kind')!='paused_fault' or row.get('hook')!=fault['hook'] or row.get('target_step')!=fault['step']: raise store.StoreError('Worker marker identity mismatch')
    if p.poll() is not None: raise store.StoreError('Worker exited before kill')

def supervise(root,plan=PLAN,max_launches=8,max_stagnant=4):
    if type(max_launches) is not int or not 1<=max_launches<=12 or type(max_stagnant) is not int or not 1<=max_stagnant<=8: raise store.StoreError('Invalid restart limits')
    if type(plan) is not list or len(plan)>12 or any(type(f) is not dict or set(f)!={'hook','step'} or f['hook'] not in {'after_backward','after_optimizer','before_commit','array_closed','published'} or type(f['step']) is not int or not 1<=f['step']<=128 for f in plan): raise store.StoreError('Invalid plan')
    if any(f['hook'] in {'before_commit','array_closed','published'} and f['step']%32 for f in plan): raise store.StoreError('Checkpoint fault must target a multiple of32')
    start=time.monotonic();records=[];stagnant=0
    for i in range(max_launches):
        before,report=observed(root)
        if before==128: return {'status':'COMPLETE','launches':records,'terminal_step':128,'no_duplicate_terminal_launch':True}
        if time.monotonic()-start>540: raise store.StoreError('Total time guard')
        fault=plan[i] if i<len(plan) else None
        if fault and fault['step']<=before: raise store.StoreError('Unreachable fault plan')
        attempt='attempt'+str(i);work=store.checked_path(root/'work'/attempt);work.mkdir(exist_ok=False);token=uuid.uuid4().hex
        args=[sys.executable,'-m','live_recovery','_worker','--run',root.name,'--attempt',attempt,'--token',token]
        if fault: args+=['--fault',json.dumps(fault)]
        row=None
        with (work/'stdout.txt').open('w') as out,(work/'stderr.txt').open('w') as err:
            p=subprocess.Popen(args,cwd=store.PROJECT,env=ENV,stdout=out,stderr=err)
            try:
                deadline=time.monotonic()+40
                if fault:
                    marker=work/'marker.json'
                    while not marker.exists():
                        if p.poll() is not None: raise store.StoreError('Training process exited before fault marker')
                        if time.monotonic()>deadline: raise store.StoreError('Worker marker timeout')
                        time.sleep(.01)
                    store.checked_path(marker)
                    if marker.stat().st_size>4096: raise store.StoreError('Marker exceeds bound')
                    row=json.loads(marker.read_text());validate_marker(row,p,token,attempt,fault)
                    p.kill() # only the direct, still-live subprocess object just created
                rc=p.wait(timeout=max(.1,deadline-time.monotonic()))
            finally:
                if p.poll() is None: p.kill();p.wait(timeout=5)
        if rc!=(-9 if fault else 0): raise store.StoreError('Unexpected training subprocess return code')
        after,report=observed(root);records.append({'attempt':attempt,'pid':p.pid,'token':token,'resumed_step':before,'selected_after':after,'returncode':rc,'fault':fault,'marker':row})
        (root/'SUPERVISOR_PROGRESS.json').write_text(json.dumps(records,indent=2)+'\n')
        if after==128: return {'status':'COMPLETE','launches':records,'terminal_step':128,'no_duplicate_terminal_launch':True}
        stagnant=stagnant+1 if after<=before else 0
        if stagnant>=max_stagnant: raise store.StoreError('NO_PROGRESS_LIMIT')
    raise store.StoreError('RESTART_LIMIT')

def verify(name):
    start=time.monotonic();root=create(name)
    (root/'PLAN.json').write_text(json.dumps({'faults':PLAN,'checkpoint_every':32,'max_launches':8,'max_stagnant':4,'expected_resumes':[0,32,32,32,64],'expected_selected_after':[32,32,32,64,128],'limits':'single writer same runtime CPU; not power loss/distributed'},indent=2)+'\n')
    args=[sys.executable,'-m','training_lab.train','--data',str(root/'data'),'--out',str(root/'reference')]
    env={**ENV,'PYTHONPATH':str(store.PROJECT/'vendor/stage2')}
    with (root/'reference.stdout').open('w') as out,(root/'reference.stderr').open('w') as err:
        subprocess.run(args,cwd=store.PROJECT,env=env,check=True,stdout=out,stderr=err,timeout=40)
    result=supervise(root);records=result['launches']
    if [r['resumed_step'] for r in records]!=[0,32,32,32,64] or [r['selected_after'] for r in records]!=[32,32,32,64,128]: raise store.StoreError('Unexpected trajectory')
    final,report=store.latest(root);ma,aa=store.validate(root/'reference/checkpoint',root);mb,ab=store.validate(final,root)
    equal={k:bool(np.array_equal(aa[k],ab[k])) for k in aa};metadata_equal=ma==mb
    if not metadata_equal or not all(equal.values()): raise store.StoreError('State mismatch')
    from training_lab.train import make_model,evaluate
    from training_lab.data import load_data
    import torch
    data,_=load_data(root/'data');model=make_model()
    with torch.no_grad():
        for i,p in enumerate(model.parameters()): p.copy_(torch.from_numpy(ab[f'p{i}']))
    _,scores=evaluate(model,torch.from_numpy(data['x']),torch.from_numpy(data['y']))
    with np.load(root/'reference/predictions.npz',allow_pickle=False) as z: predictions_equal=bool(np.array_equal(scores,z['scores']))
    if not predictions_equal: raise store.StoreError('Prediction mismatch')
    # A second supervisor call must recognize terminal state without spawning or writing.
    terminal=supervise(root,[])
    if terminal['launches']: raise store.StoreError('Duplicated final training')
    result.update({'status':'PASS_LIVE_CPU_TRAINING_RESTART','metadata_equal':metadata_equal,'arrays_equal':equal,'predictions_equal':predictions_equal,'second_supervisor_launches':0,'elapsed_seconds':time.monotonic()-start,'limitations':['fault barriers deterministically pause actual training PID before SIGKILL','not arbitrary-time fault sampling','not power-loss durability','not concurrent writers','not GPU/distributed','tiny TRAIN resubstitution, no detection improvement','AI_ASSISTED; user competence NOT_ASSESSED']})
    (root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');return result
