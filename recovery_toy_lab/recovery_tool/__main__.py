"""CLI help and dependency doctor work without importing optional runtime dependencies."""
import argparse,json,os,sys,platform
from pathlib import Path

def doctor():
    from importlib.metadata import version,PackageNotFoundError
    d={'python':platform.python_version(),'platform':sys.platform,'dependencies':{},'status':'PASS'}
    for name in ['numpy','torch']:
        try:d['dependencies'][name]=version(name)
        except PackageNotFoundError:d['dependencies'][name]='MISSING';d['status']='MISSING_DEPENDENCY'
    if sys.platform!='linux':d['status']='UNSUPPORTED_PLATFORM'
    return d

def main():
    p=argparse.ArgumentParser(description='Reproduce fixed CPU checkpoint recovery faults; Linux, NumPy and PyTorch required.')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    for cmd in ['demo','inspect']:
        a=sub.add_parser(cmd);a.add_argument('--output',required=True,help='Fresh run directory for demo; existing run for inspect. Parent must exist.')
    w=sub.add_parser('_worker',help=argparse.SUPPRESS)
    for arg in ['run','attempt','token']:w.add_argument('--'+arg,required=True)
    w.add_argument('--fault')
    a=p.parse_args()
    if a.command=='doctor':
        d=doctor();print(json.dumps(d,indent=2));return 0 if d['status']=='PASS' else 3
    d=doctor()
    if d['status']!='PASS':print(json.dumps(d),file=sys.stderr);return 3
    try:
        if a.command!='_worker':
            out=Path(a.output).absolute()
            if '..' in out.parts or not out.parent.is_dir() or any(q.is_symlink() for q in [out,*out.parents]):raise ValueError('Unsafe or missing output parent')
            os.environ['RECOVERY_TOOL_RUNS']=str(out.parent)
        from . import store
        if a.command=='demo':
            from .supervisor import verify
            result=verify(out.name);result['environment']=d
            result['dataset']={'kind':'synthetic','rows':8192,'positive':4096,'negative':4096}
            (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
            (out/'REPORT.txt').write_text('PASS: five fixed training-process faults recovered exactly.\n8192 synthetic rows, 4096 positives / 4096 negatives.\nSingle Linux CPU runtime; supervisor remains alive; no power-loss or distributed guarantee.\n')
        elif a.command=='inspect': result=store.inspect(store.run_root(out.name))
        else:
            from .worker import run
            result=run(a.run,a.attempt,a.token,json.loads(a.fault) if a.fault else None)
        print(json.dumps(result,indent=2));return 0
    except (ValueError,OSError,RuntimeError) as e:
        print(json.dumps({'status':'FAIL','error_type':type(e).__name__,'message':str(e)}),file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
