set +e
qstat -f 25273781.pbs101 2>&1
printf '\n--- artifact state ---\n'
if test -d /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction-tiebreak/artifacts/25273781.pbs101; then ls -la /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction-tiebreak/artifacts/25273781.pbs101; printf '\n--- job log tail ---\n'; tail -n 20 /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction-tiebreak/artifacts/25273781.pbs101/job.log 2>&1; else echo 'artifact directory absent'; fi
