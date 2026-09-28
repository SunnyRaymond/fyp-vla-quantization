"""User-authorized one-time relay of the fixed 74.6 MB bundle; no model work locally."""
import json
import socket
import sys
import time
from pathlib import Path

import paramiko

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK.parents[2] / 'nscc-access'))
from aspire2a_shell import ASPIRE2A_HOST, NSCC_KNOWN_HOSTS, read_credentials, verify_host_key
from autodl_control import connect as connect_autodl

SOURCE = '/scratch/users/ntu/yguo017/lewm-rolling-ball-async/bundles/rolling-ball-lewm-epoch100.tar.gz'
DEST = '/root/autodl-tmp/rolling-ball-lewm/incoming/rolling-ball-lewm-epoch100.tar.gz'
EXPECTED_BYTES = 74570166


def absent(sftp, path):
    try:
        sftp.stat(path)
    except IOError as exc:
        if exc.errno == 2:
            return
        raise
    raise RuntimeError('Refusing an existing transfer destination')


def main():
    if sys.argv[1:] != ['--authorized-single-bundle-relay']:
        raise RuntimeError('This exact one-time relay requires explicit user authorization')
    partial = TASK / 'bundle-relay.download.partial'
    lock = TASK / 'bundle-relay.lock'
    start = time.monotonic()
    nscc = None
    rented = None
    downloaded = False
    created = False
    with lock.open('x') as owned_lock:
        try:
            credentials = read_credentials()
            nscc = paramiko.Transport(socket.create_connection((ASPIRE2A_HOST, 22), timeout=25))
            nscc.start_client(timeout=25)
            verify_host_key(nscc, ASPIRE2A_HOST, NSCC_KNOWN_HOSTS)
            nscc.auth_password(credentials['NSCC_USERNAME'], credentials['NSCC_PASSWORD'])
            if not nscc.is_authenticated():
                raise RuntimeError('ASPIRE2A authentication failed')
            with paramiko.SFTPClient.from_transport(nscc) as source:
                source.get_channel().settimeout(120)
                if source.stat(SOURCE).st_size != EXPECTED_BYTES:
                    raise RuntimeError('Frozen source archive byte count changed')
                with partial.open('xb') as local:
                    created = True
                    source.getfo(SOURCE, local)
                if partial.stat().st_size != EXPECTED_BYTES:
                    raise RuntimeError('Source download incomplete')
                downloaded = True
            nscc.close()
            nscc = None
            print('source_retrieved_bytes=' + str(EXPECTED_BYTES), flush=True)
            rented, _ = connect_autodl()
            with paramiko.SFTPClient.from_transport(rented) as destination:
                destination.get_channel().settimeout(120)
                remote_partial = DEST + '.sftp.partial'
                absent(destination, DEST)
                absent(destination, remote_partial)
                with partial.open('rb') as local:
                    destination.putfo(local, remote_partial, file_size=EXPECTED_BYTES, confirm=True)
                if destination.stat(remote_partial).st_size != EXPECTED_BYTES:
                    raise RuntimeError('Rented upload incomplete')
                destination.rename(remote_partial, DEST)
                remote_bytes = destination.stat(DEST).st_size
                if remote_bytes != EXPECTED_BYTES:
                    raise RuntimeError('Final archive byte count differs')
            result = {'status': 'PASS', 'route': 'user-authorized single-bundle SFTP relay',
                      'source': SOURCE, 'destination': DEST, 'bytes': remote_bytes,
                      'transfer_wall_s': time.monotonic() - start,
                      'model_loaded_locally': False, 'closed_loop_evaluated': False,
                      'host_key_verification': 'pinned on both SSH connections'}
            (TASK / 'RELAY_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
            partial.unlink()
            print(json.dumps(result), flush=True)
        except Exception:
            if created and partial.exists() and not downloaded:
                partial.unlink()  # Only this script's incomplete regular file.
            raise
        finally:
            if nscc is not None:
                nscc.close()
            if rented is not None:
                rented.close()
            owned_lock.close()
            lock.unlink()


if __name__ == '__main__':
    main()
