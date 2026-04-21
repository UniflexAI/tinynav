import argparse
import asyncio
import json
import os

from aiohttp import ClientConnectorError, ClientSession, WSCloseCode, WSMsgType, web


async def index(request: web.Request) -> web.FileResponse:
    root = request.app["root"]
    return web.FileResponse(os.path.join(root, "index.html"))


async def static_file(request: web.Request) -> web.StreamResponse:
    root = request.app["root"]
    path = request.match_info["path"]
    target = os.path.abspath(os.path.join(root, path))
    if not target.startswith(root):
        raise web.HTTPForbidden()
    if not os.path.isfile(target):
        raise web.HTTPNotFound()
    return web.FileResponse(target)


async def ws_proxy(request: web.Request) -> web.WebSocketResponse:
    upstream_url = request.app["upstream_url"]
    client_ws = web.WebSocketResponse(heartbeat=20.0)
    await client_ws.prepare(request)

    async with ClientSession() as session:
        try:
            async with session.ws_connect(upstream_url, heartbeat=20.0) as upstream_ws:
                async def client_to_upstream() -> None:
                    async for msg in client_ws:
                        if msg.type == WSMsgType.TEXT:
                            await upstream_ws.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await upstream_ws.send_bytes(msg.data)
                        elif msg.type == WSMsgType.CLOSE:
                            await upstream_ws.close()

                async def upstream_to_client() -> None:
                    async for msg in upstream_ws:
                        if msg.type == WSMsgType.TEXT:
                            await client_ws.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await client_ws.send_bytes(msg.data)
                        elif msg.type == WSMsgType.CLOSE:
                            await client_ws.close()

                await asyncio.gather(client_to_upstream(), upstream_to_client())
        except ClientConnectorError as exc:
            await client_ws.close(
                code=WSCloseCode.INTERNAL_ERROR,
                message=f"rosbridge unreachable: {exc}".encode()[:123],
            )

    return client_ws


async def pois_api(request: web.Request) -> web.Response:
    pois_json_path = request.app["pois_json_path"]
    if not os.path.isfile(pois_json_path):
        return web.json_response({"pois": []})

    try:
        with open(pois_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        return web.json_response({"pois": [], "error": str(exc)}, status=500)

    pois = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            pos = value.get("position")
            if not isinstance(pos, list) or len(pos) < 2:
                continue
            try:
                idx = int(key)
            except Exception:
                continue
            pois.append({
                "index": idx,
                "name": value.get("name", f"POI_{idx}"),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]) if len(pos) > 2 else 0.0,
            })
    pois.sort(key=lambda x: x["index"])
    return web.json_response({"pois": pois})

async def save_pois_api(request: web.Request) -> web.Response:
    pois_json_path = request.app["pois_json_path"]
    try:
        payload = await request.json()
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"invalid json: {exc}"}, status=400)

    items = payload.get("pois")
    if not isinstance(items, list):
        return web.json_response({"ok": False, "error": "pois must be a list"}, status=400)

    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
            x = float(item.get("x"))
            y = float(item.get("y"))
            z = float(item.get("z", 0.0))
        except Exception:
            continue
        name = str(item.get("name", f"POI_{idx}"))
        out[str(idx)] = {
            "name": name,
            "position": [x, y, z],
        }

    try:
        os.makedirs(os.path.dirname(pois_json_path), exist_ok=True)
        with open(pois_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)

    return web.json_response({"ok": True, "count": len(out)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve TinyNav mobile control page")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--rosbridge-url",
        default="ws://127.0.0.1:9090",
        help="Upstream rosbridge websocket URL",
    )
    parser.add_argument(
        "--pois-json-path",
        default="/tinynav/tinynav_db/map/pois.json",
        help="Path to pois.json for map overlay and goto",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    app = web.Application()
    app["root"] = root
    app["upstream_url"] = args.rosbridge_url
    app["pois_json_path"] = args.pois_json_path
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_proxy)
    app.router.add_get("/api/pois", pois_api)
    app.router.add_post("/api/pois", save_pois_api)
    app.router.add_get("/{path:.*}", static_file)

    print(f"[mobile-control] serving {root} on http://{args.host}:{args.port}")
    print(f"[mobile-control] proxy websocket /ws -> {args.rosbridge_url}")
    print(f"[mobile-control] pois json -> {args.pois_json_path}")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
