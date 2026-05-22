{{flutter_js}}
{{flutter_build_config}}

_flutter.loader.load({
  config: {
    // Keep Flutter CanvasKit/skwasm resources on the robot web server.
    // Without this, Flutter may fetch CanvasKit from www.gstatic.com at runtime.
    canvasKitBaseUrl: "canvaskit/",
    // Keep Flutter fallback font fetches local instead of fonts.gstatic.com.
    fontFallbackBaseUrl: "fonts/"
  },
  serviceWorkerSettings: {
    serviceWorkerVersion: {{flutter_service_worker_version}}
  }
});
