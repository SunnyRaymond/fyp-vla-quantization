set +e
qstat -xf 25273781.pbs101 2>&1
printf '\n--- artifact listing ---\n'
ls -la /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction-tiebreak/artifacts/25273781.pbs101 2>&1
printf '\n--- status files ---\n'
cat /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction-tiebreak/artifacts/25273781.pbs101/job_status 2>&1
cat /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction-tiebreak/artifacts/25273781.pbs101/final_exit_status.txt 2>&1
printf '\n--- log tail ---\n'
tail -n 18 /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction-tiebreak/artifacts/25273781.pbs101/job.log 2>&1
