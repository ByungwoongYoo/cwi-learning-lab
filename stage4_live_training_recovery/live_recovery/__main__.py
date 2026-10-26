import argparse,json
from recovery_lab import store
from .supervisor import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','inspect','_worker']);p.add_argument('--run',required=True);p.add_argument('--attempt');p.add_argument('--token');p.add_argument('--fault');a=p.parse_args()
    if a.command=='verify': r=verify(a.run)
    elif a.command=='inspect': r=store.inspect(store.run_root(a.run))
    else:
        from .worker import run
        r=run(a.run,a.attempt,a.token,json.loads(a.fault) if a.fault else None)
    print(json.dumps(r,indent=2))
if __name__=='__main__': main()
