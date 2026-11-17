import argparse
import signal
import socket
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake embedding server for e2e tests")
    parser.add_argument("--zmq-port", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--passages-file", type=str, default="")
    parser.add_argument("--distance-metric", type=str, default="")
    parser.add_argument("--embedding-mode", type=str, default="sentence-transformers")
    parser.add_argument("--enable-warmup", action="store_true")
    parser.add_argument("--daemon-mode", action="store_true")
    parser.add_argument("--daemon-ttl", type=int, default=0)
    args = parser.parse_args()

    stop = {"value": False}
    last_activity = time.time()

    def _handle_signal(signum, frame):
        stop["value"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Bind and keep accepting connections so _check_port sees the process as alive.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.zmq_port))
    server.listen(16)
    server.settimeout(0.2)

    print("fake server stdout", flush=True)
    print("fake server stderr", file=sys.stderr, flush=True)

    try:
        while not stop["value"]:
            if args.daemon_mode and args.daemon_ttl > 0:
                if (time.time() - last_activity) >= args.daemon_ttl:
                    break
            try:
                conn, _addr = server.accept()
                # liveness probes connect+close without sending payload; do not
                # treat them as activity so TTL expiry can still be observed.
                try:
                    conn.settimeout(0.01)
                    payload = conn.recv(1)
                    if payload:
                        last_activity = time.time()
                except Exception:
                    pass
                conn.close()
            except socket.timeout:
                pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
