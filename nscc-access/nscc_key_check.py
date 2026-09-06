from __future__ import annotations

import io
import re
import sys
from pathlib import Path


CREDENTIAL_PATH = Path(__file__).with_name("nscc-credentials.env")
KNOWN_HOSTS_PATH = Path(r"C:\Users\Raymond\.ssh\known_hosts_nscc_aspire2a")


def read_field(text: str, name: str) -> str:
    prefix = f"{name}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return ""


def read_private_key(text: str) -> str:
    marker = "NSCC_SSH_PKEY="
    start = text.find(marker)
    if start < 0:
        return ""

    value = text[start + len(marker) :]
    begin = value.find("-----BEGIN ")
    if begin >= 0:
        value = value[begin:]
        end_match = re.search(r"-----END [^-\r\n]+ PRIVATE KEY-----", value)
        if end_match:
            value = value[: end_match.end()]
    else:
        value = value.splitlines()[0] if value.splitlines() else value

    value = value.strip().strip('"').strip("'")
    if "\\n" in value and "\n" not in value:
        value = value.replace("\\r\\n", "\n").replace("\\n", "\n")
    return value.strip() + "\n" if value else ""


def load_key(key_text: str):
    import paramiko

    loaders = [paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey]
    errors: list[str] = []
    for key_class in loaders:
        try:
            return key_class.from_private_key(io.StringIO(key_text))
        except paramiko.PasswordRequiredException:
            raise RuntimeError("The private key is encrypted and needs its passphrase.")
        except Exception as exc:  # Try the next supported key type.
            errors.append(type(exc).__name__)
    raise RuntimeError("The saved value is not a supported OpenSSH private key.")


def main() -> int:
    import paramiko

    text = CREDENTIAL_PATH.read_text(encoding="utf-8-sig")
    host = read_field(text, "NSCC_HOST").strip()
    username = read_field(text, "NSCC_USERNAME").strip()
    key_text = read_private_key(text)

    if not host or not username or not key_text:
        raise RuntimeError("Host, username, or NSCC_SSH_PKEY is missing.")

    key = load_key(key_text)
    print(f"Private key parsed: {key.get_name()}, {key.get_bits()} bits")
    print(f"Private key fingerprint: {key.fingerprint}")

    client = paramiko.SSHClient()
    client.load_host_keys(str(KNOWN_HOSTS_PATH))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            hostname=host,
            username=username,
            pkey=key,
            allow_agent=False,
            look_for_keys=False,
            timeout=20,
            banner_timeout=20,
            auth_timeout=20,
        )
    except paramiko.BadAuthenticationType as exc:
        allowed = ", ".join(exc.allowed_types)
        print(f"Server rejected key authentication; allowed methods: {allowed}")
        return 2
    except paramiko.AuthenticationException:
        print("The server offered key authentication but rejected this key.")
        return 3

    _, stdout, stderr = client.exec_command("hostname; id -un", timeout=20)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    client.close()
    print("Key authentication succeeded.")
    if output:
        print(output)
    if error:
        print(error, file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Key check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
