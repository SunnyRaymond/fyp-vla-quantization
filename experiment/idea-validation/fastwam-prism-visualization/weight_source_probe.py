"""Bounded source/metadata read through the existing trusted SSH controller."""
import control
import shlex

code, output = control.command(
    "python3 -c " + shlex.quote("import os,json; p='/scratch/users/ntu/yguo017/fastwam-prism-visualization-20261012/results/full/weight_analysis/data/modules/m0005.json'; assert os.stat(p).st_size<262144; d=json.load(open(p)); c=next(iter(d['contexts'].values())); print(json.dumps({'module_bytes':os.stat(p).st_size,'field_bytes':{k:len(json.dumps(v,separators=(',',':'))) for k,v in d.items()},'context_keys':list(c),'context_field_bytes':{k:len(json.dumps(v,separators=(',',':'))) for k,v in c.items()}}))"),
    timeout=35,
)
print(output)
raise SystemExit(code)
