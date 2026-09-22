import json, math
p = r'''D:/Downloads/Final Year Project/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/local-status/25269182.pbs101/selective_teacher_correction_summary.json'''
s=json.load(open(p,encoding='utf-8'))
blocks=s['calibration_evaluation']['blocks']
r=sorted(blocks,key=lambda b:(-float(b['global_rank_disagreement_secondary']), str(b.get('pairing_key',''))))
for i in range(39):
 print(i+1,repr(float(r[i]['global_rank_disagreement_secondary'])),r[i]['pairing_key'],repr(float(r[i]['uncertainty'])))
print('u1',sum(float(b['uncertainty'])==1.0 for b in blocks),'g>',sum(float(b['global_rank_disagreement_secondary'])>((float(r[35]['global_rank_disagreement_secondary'])+float(r[36]['global_rank_disagreement_secondary']))/2) for b in blocks))
