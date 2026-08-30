"""Spawn an assembled engine pack and check it answers BEP.

Not a host and not a test suite — the narrow question is whether the thing we
are about to publish starts, finds its weights beside itself, and evaluates a
position. A pack that builds but cannot load its network is the failure worth
catching before it reaches a download link.

Speaks the protocol's LSP-style framing: Content-Length, blank line, JSON.
"""

import json
import subprocess
import sys

# The opening position and a money-game match id.
OPENING_POSITION = "4HPwATDgc/ABMA"
MONEY_MATCH = "MIEFAAAAAAAA"


def send(proc, obj):
    body = json.dumps(obj).encode("utf-8")
    proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    proc.stdin.write(body)
    proc.stdin.flush()


def read(proc):
    length = None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("engine closed stdout before replying")
        line = line.decode("ascii").strip()
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1])
        elif line == "":
            break
    if length is None:
        raise SystemExit("no Content-Length in reply")
    return json.loads(proc.stdout.read(length).decode("utf-8"))


def main() -> int:
    exe = sys.argv[1]
    proc = subprocess.Popen(
        [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    send(proc, {"jsonrpc": "2.0", "id": 1, "method": "describe"})
    described = read(proc)
    if "error" in described:
        raise SystemExit(f"describe failed: {described['error']}")
    result = described["result"]
    print(
        f"describe: {result['engine']['displayName']} {result['engine']['version']} "
        f"protocol={result['protocolVersion']}"
    )

    send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "evaluatePosition",
            "params": {
                "positionId": OPENING_POSITION,
                "matchId": MONEY_MATCH,
                "level": "1ply",
            },
        },
    )
    evaluated = read(proc)
    if "error" in evaluated:
        raise SystemExit(f"evaluatePosition failed: {evaluated['error']}")
    ev = evaluated["result"]
    print(f"opening: equity={ev['Equity']:+.4f} win={ev['WinProb']:.4f}")

    # The player on roll holds a small edge in the opening position. A network
    # that failed to load, or a board that arrived mirrored, does not land here.
    if not 0.45 < ev["WinProb"] < 0.60:
        raise SystemExit(f"opening win probability {ev['WinProb']} is not credible")

    send(proc, {"jsonrpc": "2.0", "id": 3, "method": "shutdown"})
    read(proc)
    proc.wait(timeout=30)
    if proc.returncode != 0:
        raise SystemExit(f"engine exited {proc.returncode} on shutdown")
    print("pack answers the protocol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
