"""Validate an interactively entered HF token and save it on ASPIRE2A."""
import getpass
import json
import urllib.error
import urllib.request

import aspire2a_shell


def main():
    token = getpass.getpass("New Hugging Face READ token (hidden): ").strip()
    if not token.startswith("hf_") or any(c.isspace() for c in token):
        raise ValueError("Invalid token format; nothing saved.")
    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            account = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("HF authentication failed (HTTP %s); nothing saved." % exc.code) from None
    if not account.get("name"):
        raise RuntimeError("HF returned no account name; nothing saved.")
    jump, connection = aspire2a_shell.connect()
    try:
        import paramiko
        with paramiko.SFTPClient.from_transport(connection) as sftp:
            home = sftp.normalize(".")
            directory = home + "/.cache"
            for path in (directory, directory + "/huggingface"):
                try:
                    sftp.stat(path)
                except FileNotFoundError:
                    sftp.mkdir(path, mode=0o700)
            target = directory + "/huggingface/token"
            try:
                sftp.lstat(target)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("A token already exists; nothing overwritten.")
            # Exclusive creation prevents replacing a credential created concurrently.
            with sftp.file(target, "wx") as stream:
                stream.chmod(0o600)
                stream.write(token + "\n")
                stream.flush()
            print("Authenticated account:", account["name"])
            print("Saved on ASPIRE2A:", target)
            print('For jobs overriding HF_HOME, set HF_TOKEN_PATH="$HOME/.cache/huggingface/token".')
    finally:
        connection.close()
        jump.close()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print("Login not completed:", str(exc))
        raise SystemExit(1)
