python3 - <<'PY'
import zipfile
z=zipfile.ZipFile('/scratch/users/ntu/yguo017/dino-wm-wall/downloads/wall_single-local.zip')
print('\n'.join(z.namelist()[:12]))
c=zipfile.ZipFile('/scratch/users/ntu/yguo017/dino-wm-wall/downloads/outputs-local.zip')
print(c.read('outputs/wall_single/hydra.yaml').decode())
PY
