#!/usr/bin/env python3
"""Bounded, credential-free public HTTPS/DNS probe for a genuine PBS CPU host."""
from __future__ import annotations

import argparse
import errno
import os
import signal
import socket
import ssl
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from pathlib import Path


ENDPOINTS = (
    ("huggingface_revision", "huggingface.co",
     "https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets/revision/main"),
    ("github_api_root", "api.github.com", "https://api.github.com/"),
)
TIMEOUT_SECONDS = 15
DNS_TIMEOUT_SECONDS = 8


def safe_reason(exc: BaseException) -> str:
    status = getattr(exc, "code", None)
    if isinstance(status, int):
        return f"HTTP_STATUS_{status}"
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "TLS_CERTIFICATE_VERIFICATION_FAILED"
    if isinstance(reason, ssl.SSLError):
        return "TLS_ERROR"
    if isinstance(reason, socket.gaierror):
        value = reason.errno
        return f"DNS_RESOLUTION_FAILED errno={value if value is not None else 'unknown'}"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "CONNECTION_TIMEOUT"
    if isinstance(reason, ConnectionRefusedError):
        return "CONNECTION_REFUSED"
    if isinstance(reason, OSError):
        lowered = str(reason).lower()
        if "proxy" in lowered or "tunnel connection failed" in lowered:
            return "PROXY_CONNECTION_FAILED"
        if "certificate verify" in lowered or "local issuer" in lowered:
            return "TLS_CERTIFICATE_VERIFICATION_FAILED"
        if "timed out" in lowered or "timeout" in lowered:
            return "CONNECTION_TIMEOUT"
        name = errno.errorcode.get(reason.errno, "OS_ERROR") if reason.errno is not None else "OS_ERROR"
        return f"{name} errno={reason.errno if reason.errno is not None else 'unknown'}"
    if isinstance(reason, str):
        lowered = reason.lower()
        rules = (
            (("certificate verify", "local issuer"), "TLS_CERTIFICATE_VERIFICATION_FAILED"),
            (("temporary failure in name resolution", "name or service not known",
              "nodename nor servname", "getaddrinfo failed"), "DNS_RESOLUTION_FAILED"),
            (("proxy", "tunnel connection failed"), "PROXY_CONNECTION_FAILED"),
            (("timed out", "timeout"), "CONNECTION_TIMEOUT"),
            (("network is unreachable", "no route to host"), "NETWORK_UNREACHABLE"),
            (("connection refused",), "CONNECTION_REFUSED"),
        )
        for needles, label in rules:
            if any(needle in lowered for needle in needles):
                return label
    return f"NETWORK_REASON_CLASS_{type(reason).__name__}"


def allocation_host() -> str:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS allocation identity is missing")
    host = socket.gethostname().split(".", 1)[0]
    nodes = {line.split(".", 1)[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if "login" in host.lower() or host not in nodes:
        raise RuntimeError("Current host is not a listed PBS compute node")
    return host


def dns_check(host: str) -> bool:
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("DNS timeout")))
    signal.alarm(DNS_TIMEOUT_SECONDS)
    try:
        return bool(socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def sanitized_proxies() -> dict[str, str]:
    """Use proxy hosts without forwarding URL userinfo, path, or query parameters."""
    proxies = {}
    for scheme in ("http", "https"):
        raw = (os.environ.get(f"{scheme}_proxy") or os.environ.get(f"{scheme.upper()}_PROXY")
               or os.environ.get("all_proxy") or os.environ.get("ALL_PROXY"))
        if not raw:
            continue
        try:
            parsed = urlsplit(raw if "://" in raw else f"{scheme}://{raw}")
            if not parsed.hostname or parsed.scheme not in ("http", "https"):
                continue
            hostname = parsed.hostname
            if ":" in hostname:
                hostname = f"[{hostname}]"
            port = parsed.port
            authority = f"{hostname}:{port}" if port else hostname
            proxies[scheme] = f"{parsed.scheme}://{authority}"
        except ValueError:
            continue
    return proxies


def http_check(label: str, url: str, opener, context: str) -> bool:
    request = Request(url, headers={"User-Agent": "fastwam-calibration-network-probe/1.0"})
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            status = int(response.status)
        print(f"HTTP context={context} endpoint={label} status={status}", flush=True)
        return 200 <= status < 400
    except HTTPError as exc:
        print(f"HTTP context={context} endpoint={label} status={int(exc.code)}", flush=True)
        exc.close()
        return False
    except Exception as exc:
        print(f"NETWORK_ERROR context={context} endpoint={label} error_class={type(exc).__name__} "
              f"reason={safe_reason(exc)}", flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", choices=("host", "container"), required=True)
    context = parser.parse_args().context
    try:
        host = allocation_host()
    except Exception as exc:
        print(f"PBS_GUARD_FAIL context={context} error_class={type(exc).__name__}", flush=True)
        return 2
    print(f"HOST context={context} hostname={host}", flush=True)
    proxy_names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                   "http_proxy", "https_proxy", "all_proxy", "no_proxy")
    configured = [name for name in proxy_names if os.environ.get(name)]
    proxies = sanitized_proxies()
    print(f"PROXY_ENV context={context} configured={'yes' if configured else 'no'} "
          f"names={','.join(configured) or 'none'} credentials=excluded", flush=True)
    ca_names = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO")
    ca_configured = [name for name in ca_names if os.environ.get(name)]
    print(f"TLS_CA_ENV context={context} configured={'yes' if ca_configured else 'no'} "
          f"names={','.join(ca_configured) or 'none'}", flush=True)
    opener = build_opener(ProxyHandler(proxies))

    all_ok = True
    for label, host_name, url in ENDPOINTS:
        try:
            resolved = dns_check(host_name)
            print(f"DNS context={context} host={host_name} status={'ok' if resolved else 'failed'}", flush=True)
        except Exception as exc:
            print(f"DNS context={context} host={host_name} status=failed error_class={type(exc).__name__} "
                  f"reason={safe_reason(exc)}", flush=True)
            resolved = False
        if resolved or proxies:
            all_ok = http_check(label, url, opener, context) and all_ok
        else:
            print(f"HTTP context={context} endpoint={label} status=skipped_dns_failed", flush=True)
            all_ok = False
    print(f"NETWORK_PROBE_RESULT context={context} status={'ok' if all_ok else 'failed'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
