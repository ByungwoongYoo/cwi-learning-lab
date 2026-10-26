"""One actual training process; test hooks pause this same PID for supervisor kill."""
import json,os,time
from pathlib import Path
import numpy as np
from . import store
from recovery_tool import train as t
from recovery_tool.data import load_data
import torch

HOOKS={'after_backward','after_optimizer','before_commit','array_closed','published'}
def run(name,attempt,token,fault=None):
    if type(attempt) is not str or not store.NAME.fullmatch(attempt) or type(token) is not str or len(token)!=32 or any(c not in '0123456789abcdef' for c in token): raise store.StoreError('Invalid launch identity')
    root=store.run_root(name);work=store.checked_path(root/'work'/attempt)
    c,ch,dm=store.bindings(root);t.setup(c);a,_=load_data(root/'data')
    x=torch.from_numpy(a['x']);y=torch.from_numpy(a['y']);model=t.make_model()
    opt=torch.optim.SGD(model.parameters(),lr=c['learning_rate'],momentum=c['momentum'],weight_decay=c['weight_decay'],foreach=False)
    rng=np.random.default_rng(c['sampler_seed']);permutation=rng.permutation(len(x));p={'epoch':0,'cursor':0,'step':0};losses=[]
    initial_loss,_=t.evaluate(model,x,y);report=store.inspect(root)
    if report['selected']:
        cp,report=store.latest(root);meta,state=store.validate(cp,root)
        with torch.no_grad():
            for i,param in enumerate(model.parameters()):
                param.copy_(torch.from_numpy(state[f'p{i}']));opt.state[param]['momentum_buffer']=torch.from_numpy(state[f'v{i}']).clone()
        rng.bit_generator.state=meta['sampler_rng'];torch.set_rng_state(torch.from_numpy(state['torch_rng']))
        permutation=state['permutation'];p=meta['progress'];losses=meta['losses'];initial_loss=meta['initial_loss']
    elif report['invalid']: raise store.StoreError('NO_VALID_CHECKPOINT: refusing fresh start over invalid generations')
    start=p['step'];completed_here=0
    def event(kind,**kw):
        row={'kind':kind,'pid':os.getpid(),'token':token,'attempt':attempt,'saved_progress_step':p['step'],'optimizer_calls_this_process':completed_here,**kw}
        with (work/'events.jsonl').open('a') as f: f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()
        return row
    event('started',resumed_from=start)
    def hook(label,step):
        if fault and fault=={'hook':label,'step':step}:
            row=event('paused_fault',hook=label,target_step=step)
            tmp=work/'marker.tmp';tmp.write_text(json.dumps(row));os.replace(tmp,work/'marker.json')
            while True: time.sleep(.02)
    def store_hook(label): hook(label,p['step'])
    store.hook=store_hook # scoped to this child; publication hook stays in live training PID
    while p['step']<128:
        step=p['step']+1;idx=permutation[p['cursor']:p['cursor']+c['batch_size']]
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[idx]),y[idx])
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        if any(q.grad is None or not torch.isfinite(q.grad).all() for q in model.parameters()): raise RuntimeError('Nonfinite gradient')
        hook('after_backward',step);opt.step();completed_here+=1;hook('after_optimizer',step)
        losses.append(float(loss.detach()));p['step']=step;p['cursor']+=len(idx)
        if p['cursor']==len(x):
            p['epoch']+=1;p['cursor']=0
            if p['epoch']<c['epochs']: permutation=rng.permutation(len(x))
        event('step_completed',step=step)
        if step%32==0:
            hook('before_commit',step)
            source=work/f'cp{step:06d}';t.save_checkpoint(source,c,ch,dm,model,opt,rng,permutation,p,losses,initial_loss)
            store.commit(root,source);event('checkpoint_published',step=step)
    event('finished',resumed_from=start)
