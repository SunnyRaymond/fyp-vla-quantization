"""One authenticated archive upload on the user-provided AutoDL HTTPS service."""
import hmac
import json
import os
import platform
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

BASE = Path('/root/autodl-tmp/rolling-ball-lewm')
EXPECTED_BYTES = 74570166
EXPECTED_HOST = 'autodl-container-e7e742ba1d-c91a1edb'


def main():
    if platform.system() != 'Linux' or platform.node() != EXPECTED_HOST:
        raise RuntimeError('Expected explicitly rented Linux host')
    if os.environ.get('PBS_JOBID') or os.environ.get('PBS_NODEFILE'):
        raise RuntimeError('Not a PBS entry point')
    token = (BASE / 'incoming/.upload_token').read_text().strip()
    if len(token) != 64:
        raise RuntimeError('Invalid task upload token')
    final = BASE / 'incoming/rolling-ball-lewm-epoch100.tar.gz'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log request headers or credentials.

        def reply(self, code, value):
            body = (json.dumps(value) + '\n').encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'status': 'NOT_FOUND'})
            return self.reply(200, {'status': 'READY', 'expected_bytes': EXPECTED_BYTES,
                                    'archive_present': final.exists()})

        def do_POST(self):
            if self.path != '/upload':
                return self.reply(404, {'status': 'NOT_FOUND'})
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                return self.reply(403, {'status': 'FORBIDDEN'})
            if self.headers.get('Content-Length') != str(EXPECTED_BYTES):
                return self.reply(400, {'status': 'WRONG_CONTENT_LENGTH'})
            if final.exists():
                return self.reply(409, {'status': 'ARCHIVE_ALREADY_PRESENT'})
            partial = final.with_name(final.name + '.http.partial')
            created = False
            try:
                self.connection.settimeout(120)
                with partial.open('xb') as out:
                    created = True
                    remaining = EXPECTED_BYTES
                    while remaining:
                        data = self.rfile.read(min(1024 * 1024, remaining))
                        if not data:
                            raise RuntimeError('Incomplete upload')
                        out.write(data)
                        remaining -= len(data)
                os.link(partial, final)  # Atomic publish, refuses an existing destination.
                partial.unlink()
                result = {'status': 'PASS', 'bytes': EXPECTED_BYTES, 'path': str(final)}
                (BASE / 'incoming/http_upload_result.json').write_text(json.dumps(result) + '\n')
                self.reply(200, result)
            except Exception as exc:
                if created and partial.exists():
                    partial.unlink()
                self.reply(500, {'status': 'FAIL', 'error': type(exc).__name__})

    print('ready_port=6006', flush=True)
    HTTPServer(('0.0.0.0', 6006), Handler).serve_forever()


if __name__ == '__main__':
    main()
