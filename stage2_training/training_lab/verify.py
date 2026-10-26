"""One frozen technical experiment; fresh OS processes, no tuning loop."""
from __future__ import annotations
import argparse, copy, datetime, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, FormatError, digest, json_read, json_write, frozen_config, checked_npz
from .data import prepare, load_data
from .train import make_model, read_checkpoint

def gradient_check():
    # Independent finite-difference smoke on synthetic values only; no real label tuning.
    torch.manual_seed(1701);model=make_model();model.eval()
    x=torch.linspace(-1,1,35,dtype=torch.float64).reshape(5,7);y=torch.tensor([[0.],[1.],[0.],[1.],[0.]],dtype=torch.float64)
    def loss(): return torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
    model.zero_grad();loss().backward();eps=1e-6;errors=[];relative=[]
    for p in model.parameters():
        analytic=p.grad.detach().clone().flatten()
        for j in range(p.numel()):
            with torch.no_grad():
                flat=p.flatten();base=float(flat[j]);flat[j]=base+eps;plus=float(loss());flat[j]=base-eps;minus=float(loss());flat[j]=base
            finite=(plus-minus)/(2*eps);a=float(analytic[j]);errors.append(abs(finite-a));relative.append(abs(finite-a)/max(1e-8,abs(finite)+abs(a)))
    result={'data':'SYNTHETIC_ONLY','method':'central finite differences of all 145 parameters versus autograd, dropout disabled for deterministic differentiation','parameters_checked':len(errors),'epsilon':eps,'max_absolute_error':max(errors),'max_scaled_relative_error':max(relative),'absolute_tolerance':1e-6,'passed':max(errors)<1e-6}
    if not result['passed']: raise AssertionError(result)
    return result

def negative_tests(checkpoint,c,ch,dm):
    results=[]
    def case(name,mutator):
        with tempfile.TemporaryDirectory(prefix='negative-checkpoint-') as tmp:
            target=Path(tmp)/'copy';shutil.copytree(checkpoint,target);mutator(target)
            try: read_checkpoint(target,c,ch,dm)
            except (FormatError,FileNotFoundError) as exc: results.append({'case':name,'rejected':True,'exception':type(exc).__name__,'reason':str(exc)})
            else: raise AssertionError('Unsafe acceptance: '+name)
    def meta_change(p,key,value):
        m=json_read(p/'manifest.json');m[key]=value;json_write(p/'manifest.json',m)
    def array_change(p,key,transform):
        with np.load(p/'state.npz',allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        arrays[key]=transform(arrays[key]);np.savez(p/'state.npz',**arrays)
        m=json_read(p/'manifest.json');m['archive_sha256']=digest((p/'state.npz').read_bytes());json_write(p/'manifest.json',m)
    case('truncated_archive',lambda p:(p/'state.npz').write_bytes((p/'state.npz').read_bytes()[:100]))
    def corrupt(p):
        b=bytearray((p/'state.npz').read_bytes());b[len(b)//2]^=1;(p/'state.npz').write_bytes(b)
    case('corrupt_archive',corrupt)
    case('wrong_hash',lambda p:meta_change(p,'archive_sha256','0'*64))
    case('wrong_schema',lambda p:meta_change(p,'schema_version',2))
    case('wrong_config_binding',lambda p:meta_change(p,'config_sha256','0'*64))
    case('wrong_data_binding',lambda p:meta_change(p,'data_sha256','0'*64))
    case('optimizer_mismatch',lambda p:meta_change(p,'optimizer',{'name':'Adam'}))
    def bool_alias(p):
        m=json_read(p/'manifest.json');m['optimizer']['dampening']=False;json_write(p/'manifest.json',m)
    case('optimizer_boolean_type_alias',bool_alias)
    case('inconsistent_step_cursor',lambda p:meta_change(p,'progress',{'epoch':1,'cursor':0,'step':37}))
    case('invalid_sampler_rng',lambda p:meta_change(p,'sampler_rng',{}))
    case('duplicate_permutation_rehashed',lambda p:array_change(p,'permutation',lambda a:np.zeros_like(a)))
    case('wrong_parameter_shape_rehashed',lambda p:array_change(p,'p0',lambda a:a[:1]))
    case('wrong_parameter_dtype_rehashed',lambda p:array_change(p,'p0',lambda a:a.astype(np.float32)))
    case('nonfinite_parameter_rehashed',lambda p:array_change(p,'p0',lambda a:np.full_like(a,np.nan)))
    def duplicate(p):
        raw=(p/'manifest.json').read_text();(p/'manifest.json').write_text(raw.replace('{','{"schema_version":1,',1))
    case('duplicate_json_key',duplicate)
    def nonfinite(p):
        raw=(p/'manifest.json').read_text();m=json_read(p/'manifest.json');raw=raw.replace(str(m['initial_loss']),'NaN',1);(p/'manifest.json').write_text(raw)
    case('nonfinite_json',nonfinite)
    def broken_validhash(p):
        raw=(p/'state.npz').read_bytes()[:100];(p/'state.npz').write_bytes(raw);meta_change(p,'archive_sha256',digest(raw))
    case('truncated_archive_rehashed',broken_validhash)
    case('object_parameter_rehashed',lambda p:array_change(p,'p0',lambda a:a.astype(object)))
    case('oversized_decoded_archive_rehashed',lambda p:array_change(p,'p0',lambda a:np.zeros((600000,),dtype=np.float64)))
    case('oversized_json',lambda p:(p/'manifest.json').write_text(' '*131073))
    return results

def experiment(out):
    c,ch=frozen_config();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    grad=gradient_check();json_write(out/'gradient_check.json',grad)
    dm=prepare(out/'data')
    json_write(out/'pretraining_freeze.json',{'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_sha256':ch,'dataset_archive_sha256':dm['archive_sha256'],'sample_indices_sha256':dm['sample_indices_sha256'],'training_runs_started':0,'configuration_search':False})
    commands=[]
    for label,extra in [('uninterrupted',[]),('interrupted',['--stop-step',str(c['checkpoint_step'])]),('resumed',['--resume',str(out/'interrupted/checkpoint')])]:
        command=[sys.executable,'-m','training_lab.train','--data',str(out/'data'),'--out',str(out/label)]+extra
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
        start=time.perf_counter();result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
        (out/f'{label}_stdout.txt').write_text(result.stdout);(out/f'{label}_stderr.txt').write_text(result.stderr)
        commands.append({'label':label,'argv':command,'cwd':str(ROOT),'returncode':result.returncode,'subprocess_wall_seconds':time.perf_counter()-start})
        json_write(out/'process_commands.json',commands)
        if result.returncode: raise RuntimeError(f'{label} failed: {result.stderr}')
    full,fa=read_checkpoint(out/'uninterrupted/checkpoint',c,ch,dm);resumed,ra=read_checkpoint(out/'resumed/checkpoint',c,ch,dm)
    sums={name:json_read(out/name/'summary.json') for name in ('uninterrupted','interrupted','resumed')}
    def predictions(name):
        p=out/name/'predictions.npz';return checked_npz(p,digest(p.read_bytes()),{'scores':((8192,1),'float64')})['scores']
    fp=predictions('uninterrupted');rp=predictions('resumed')
    eq={k:bool(np.array_equal(fa[k],ra[k])) for k in fa}
    comparisons={'parameters_bitwise_equal':all(eq[f'p{i}'] for i in range(4)),'momentum_buffers_bitwise_equal':all(eq[f'v{i}'] for i in range(4)),'predictions_bitwise_equal':bool(np.array_equal(fp,rp)),'torch_rng_bitwise_equal':eq['torch_rng'],'sampler_permutation_bitwise_equal':eq['permutation'],'sampler_rng_exact_equal':full['sampler_rng']==resumed['sampler_rng'],'loss_history_exact_equal':full['losses']==resumed['losses'],'progress_exact_equal':full['progress']==resumed['progress'],'all_checkpoint_metadata_exact_equal':full==resumed,'three_distinct_process_ids':len({s['pid'] for s in sums.values()})==3,'max_parameter_abs_diff':max(float(np.max(np.abs(fa[f'p{i}']-ra[f'p{i}']))) for i in range(4)),'max_prediction_abs_diff':float(np.max(np.abs(fp-rp))),'steps':{'full':full['progress']['step'],'interrupted':sums['interrupted']['optimizer_steps_total'],'resumed_total':resumed['progress']['step']}}
    comparisons['passed']=all(v for k,v in comparisons.items() if k.endswith('_equal') or k=='three_distinct_process_ids') and comparisons['steps']=={'full':128,'interrupted':37,'resumed_total':128}
    if not comparisons['passed']: raise AssertionError(comparisons)
    negatives=negative_tests(out/'interrupted/checkpoint',c,ch,dm)
    result={'status':'PASS_TECHNICAL_TRAINING_AND_RESTART_ONLY','config_sha256':ch,'data':dm,'gradient_check':grad,'parity':comparisons,'negative_checkpoint_tests':negatives,'runs':sums,'claims_excluded':['generalization','improved_scientific_accuracy','full_87_example_training','large_model_training','GPU','multi_node','user_independent_competency'],'authorship':'AI_ASSISTED','user_competency':'NOT_ASSESSED'}
    json_write(out/'verification.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=experiment(a.out);print(json.dumps({'status':r['status'],'parity':r['parity'],'negative_tests':len(r['negative_checkpoint_tests'])},indent=2))
if __name__=='__main__':main()
