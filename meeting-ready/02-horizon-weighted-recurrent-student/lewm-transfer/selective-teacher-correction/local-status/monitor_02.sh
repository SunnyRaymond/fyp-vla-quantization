set +e
qstat -f 25269182.pbs101 2>&1
printf '\n--- artifact state ---\n'
if test -d /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/artifacts/25269182.pbs101; then ls -la /scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/artifacts/25269182.pbs101; else echo 'artifact directory absent'; fi
