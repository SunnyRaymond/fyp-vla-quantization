import json, pathlib, subprocess, time
out=pathlib.Path('/scratch/users/ntu/yguo017/dino-wm-wall/artifacts/16170841.pbs101')
for _ in range(30):
    rows=[json.loads(x) for x in (out/'logs.json').read_text().splitlines() if x.strip()]
    rows=[r for r in rows if 'mpc/success_rate' in r]
    if rows and rows[-1]['step']>=7:
        last=rows[-1]
        before=subprocess.run(['qstat','-f','16170841.pbs101'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        (out/'qstat_before_stop.txt').write_text(before.stdout+before.stderr)
        result=subprocess.run(['qdel','16170841.pbs101'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        record={'last_completed_round':last,'qdel_returncode':result.returncode,'qdel_output':result.stdout+result.stderr,'reason':'User requested stopping at the next completed round after round 6','unix_time':time.time()}
        (out/'stop_record.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
        break
    time.sleep(20)
else:
    raise SystemExit('Stop checkpoint not reached within 10 minutes')

