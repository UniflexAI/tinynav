#!/usr/bin/env python3
"""Host-side TCP proxy: local listen port -> Looper HTTP (default 169.254.10.1:80).

Example:
  python3 scripts/looper_http_proxy.py
  # then open http://127.0.0.1:8801/
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal

LOG = logging.getLogger("looper_http_proxy")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _handle(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_host: str,
    upstream_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        up_reader, up_writer = await asyncio.wait_for(
            asyncio.open_connection(upstream_host, upstream_port),
            timeout=5.0,
        )
    except Exception as exc:
        LOG.warning("upstream %s:%s connect failed from %s: %s", upstream_host, upstream_port, peer, exc)
        client_writer.close()
        await client_writer.wait_closed()
        return

    LOG.info("open %s -> %s:%s", peer, upstream_host, upstream_port)
    t1 = asyncio.create_task(_pipe(client_reader, up_writer))
    t2 = asyncio.create_task(_pipe(up_reader, client_writer))
    done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for w in (client_writer, up_writer):
        try:
            w.close()
            await w.wait_closed()
        except Exception:
            pass
    LOG.info("close %s", peer)


async def _main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _ask_stop() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _ask_stop)
        except NotImplementedError:
            pass

    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, args.upstream_host, args.upstream_port),
        host=args.listen_host,
        port=args.listen_port,
        reuse_address=True,
    )
    socks = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    LOG.info(
        "listening %s  ->  %s:%s",
        socks,
        args.upstream_host,
        args.upstream_port,
    )
    async with server:
        await stop.wait()
    LOG.info("shutting down")


def main() -> None:
    p = argparse.ArgumentParser(description="TCP proxy Looper HTTP to a local port")
    p.add_argument("--listen-host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    p.add_argument("--listen-port", type=int, default=8801, help="local port (default 8801)")
    p.add_argument("--upstream-host", default="169.254.10.1", help="Looper IP")
    p.add_argument("--upstream-port", type=int, default=80, help="Looper HTTP port")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
