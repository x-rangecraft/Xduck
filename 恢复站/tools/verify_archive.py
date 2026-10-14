"""Read-only integrity check. Usage: uv run --no-project tools/verify_archive.py."""
from pathlib import Path
import hashlib,json,re,sys
root=Path(__file__).resolve().parents[1]
errors=[];checked=0
for line in (root/'CHECKSUMS.sha256').read_text().splitlines():
 expected,relative=line.split('  ',1);p=root/relative
 if not p.is_file():errors.append('missing: '+relative)
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append('hash: '+relative)
 checked+=1
for p in root.rglob('*.json'):
 try:json.loads(p.read_text())
 except Exception as exc:errors.append(str(p.relative_to(root))+': '+str(exc))
for p in [root/'README.md',*(root/'docs').glob('*.md')]:
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in link or link.startswith('#'):continue
  if not (p.parent/link.split('#')[0]).exists():errors.append('broken link '+str(p.relative_to(root))+' → '+link)
ledger=json.loads((root/'TRAINING_LEDGER.json').read_text())
if ledger['selected_updates']!=3150:errors.append('lineage does not sum to 3150')
expected=0
for row in ledger['selected_lineage']:
 if row['lineage_start']!=expected:errors.append('lineage discontinuity')
 expected=row['lineage_end']
 if row['lineage_end']-row['lineage_start']!=row['updates']:errors.append('incorrect update count')
if expected!=3150:errors.append('wrong final checkpoint')
print(json.dumps({'ok':not errors,'hashed_files_checked':checked,'selected_updates':expected,'errors':errors},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
