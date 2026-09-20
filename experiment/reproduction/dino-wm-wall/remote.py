"""Use the existing verified SSH connection without exposing credentials."""
import argparse
import sys
import stat
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'nscc-access'))
from aspire2a_shell import connect
import paramiko

p = argparse.ArgumentParser()
p.add_argument('--command')
p.add_argument('--script')
p.add_argument('--put', nargs=2)
p.add_argument('--get', nargs=2)
p.add_argument('--get-dir', nargs=2)
a = p.parse_args()
j, n = connect()
n.channel_timeout = 30
try:
    if a.put or a.get or a.get_dir:
        sftp = paramiko.SFTPClient.from_transport(n)
        sftp.get_channel().settimeout(60)
        if a.put:
            sftp.put(*a.put)
        if a.get:
            sftp.get(*a.get)
        if a.get_dir:
            source, destination = a.get_dir
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            count = 0
            for entry in sftp.listdir_attr(source):
                if not stat.S_ISREG(entry.st_mode):
                    continue
                if Path(entry.filename).name != entry.filename:
                    raise ValueError('Unexpected remote filename')
                target = destination / entry.filename
                partial = target.with_name(target.name + '.partial')
                sftp.get(source.rstrip('/') + '/' + entry.filename, str(partial))
                if partial.stat().st_size != entry.st_size:
                    raise RuntimeError('Transferred size mismatch: ' + entry.filename)
                partial.replace(target)
                count += 1
            print(f'Transferred {count} files with matching sizes')
        sftp.close()
    else:
        command = Path(a.script).read_text(encoding='utf-8') if a.script else a.command
        channel = n.open_session(timeout=30)
        channel.settimeout(300)
        channel.exec_command(command)
        import threading
        def copy(stream, target):
            for line in iter(stream.readline, b''):
                target.write(line.decode('utf-8', errors='replace'))
                target.flush()
        err = threading.Thread(target=copy, args=(channel.makefile_stderr('rb'), sys.stderr))
        err.start()
        copy(channel.makefile('rb'), sys.stdout)
        err.join()
        raise SystemExit(channel.recv_exit_status())
finally:
    n.close()
    j.close()
