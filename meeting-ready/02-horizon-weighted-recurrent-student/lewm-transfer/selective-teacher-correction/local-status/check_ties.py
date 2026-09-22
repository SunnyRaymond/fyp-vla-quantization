import json
s=json.load(open(r'''D:/Downloads/Final Year Project/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/local-status/25269182.pbs101/selective_teacher_correction_summary.json''',encoding='utf-8'))
b=[x for x in s['calibration_evaluation']['blocks'] if float(x['uncertainty'])==1.0]
r=sorted(b,key=lambda x:(-float(x['global_rank_disagreement_secondary']),str(x['pairing_key'])))
for i in range(len(r)):
 print(i+1,repr(float(r[i]['global_rank_disagreement_secondary'])),r[i]['pairing_key'])
print('g36',repr(float(r[35]['global_rank_disagreement_secondary'])),'g37',repr(float(r[36]['global_rank_disagreement_secondary'])))
tau=(float(r[35]['global_rank_disagreement_secondary'])+float(r[36]['global_rank_disagreement_secondary']))/2
print('tau',repr(tau),'calls',sum(float(x['uncertainty'])>1.0 or (float(x['uncertainty'])==1.0 and float(x['global_rank_disagreement_secondary'])>tau) for x in s['calibration_evaluation']['blocks']))
