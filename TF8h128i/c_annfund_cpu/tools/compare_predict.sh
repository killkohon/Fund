#!/usr/bin/env bash
set -euo pipefail
CODE=${1:?code}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
"$ROOT/c_annfund/annfund" --predict "$CODE" | tee /tmp/pred_cuda.txt
"$ROOT/c_annfund_cpu/annfund" --predict "$CODE" | tee /tmp/pred_cpu.txt
python3 - <<'PY'
import re,sys
def load(path):
    rows=[]
    for line in open(path):
        m=re.match(r"(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", line)
        if m: rows.append([float(m.group(i)) for i in range(2,5)])
    return rows
a,b=load("/tmp/pred_cuda.txt"),load("/tmp/pred_cpu.txt")
assert len(a)==len(b) and a, "empty/mismatch"
rel=0
for x,y in zip(a,b):
  for u,v in zip(x,y):
    den=max(abs(u),1e-8); rel=max(rel, abs(u-v)/den)
print("max_rel", rel)
sys.exit(0 if rel < 1e-3 else 1)
PY
