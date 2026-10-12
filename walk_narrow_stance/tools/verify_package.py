"""Verify every published payload and every archived runtime file without training."""
from pathlib import Path
import hashlib,json,tarfile,argparse
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--runtimes',action='store_true');args=a.parse_args();m=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text());failed=[]
 for r in m['files']:
  p=ROOT/r['path']
  if not p.is_file() or p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:failed.append(r['path'])
 if args.runtimes:
  descriptors=json.loads((ROOT/'runtimes/snapshot_manifest.json').read_text())
  with tarfile.open(ROOT/'runtimes/base_project.tar.gz') as t:base={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
  for name,d in descriptors.items():
   hashes=base.copy()
   with tarfile.open(ROOT/'runtimes'/d['overlay']) as t:
    for r in t.getmembers():
     if r.isfile():hashes[r.name]=hashlib.sha256(t.extractfile(r).read()).hexdigest()
   for n in d['deleted']:hashes.pop(n,None)
   if hashes!=d['hashes']:failed.append('runtime:'+name)
 if failed:raise SystemExit('FAILED: '+repr(failed))
 print('PASS:',len(m['files']),'payload files;',len(descriptors) if args.runtimes else 0,'runtime snapshots; no training executed')
if __name__=='__main__':main()
