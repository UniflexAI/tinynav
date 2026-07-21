#version 460 core
#include <flutter/runtime_effect.glsl>

// Recolours the ESDF heatmap layer.
//
// The source image (uSrc) arrives already colourised with a JET-ish colormap
// baked in upstream (robot planning node → backend). This shader recovers a
// scalar t in [0,1] from each pixel, then looks up a new colour from a 256x1
// palette LUT (uLut) built from the _kEsdfHeatmap gradient in
// esdf_colormap_layer.dart. This keeps the palette a single source of truth in
// Dart; the shader stays generic.
//
// Uniform layout (must match _EsdfShaderPainter in esdf_colormap_layer.dart):
//   setFloat(0,1) -> uSize
//   setFloat(2)   -> uOpacity
//   setImageSampler(0) -> uSrc
//   setImageSampler(1) -> uLut

uniform vec2 uSize;      // draw area in pixels
uniform float uOpacity;  // layer opacity [0,1]
uniform sampler2D uSrc;  // JET-colourised ESDF source
uniform sampler2D uLut;  // 256x1 palette, x=0 danger .. x=1 far-field

out vec4 fragColor;

// Saturation below this = treat as background / no-data.
const float kMinSaturation = 0.06;

// ── JET → scalar recovery ──────────────────────────────────────────────────
// Uses hue angle rather than an RGB-curve match: hue is far more robust to
// JPEG compression noise and brightness variation than per-channel matching.
// JET sweeps hue red(danger) → yellow → green → cyan → blue(safe), so hue maps
// (approximately) monotonically to the scalar. Orientation: red=danger→t=0,
// blue=safe→t=1.
//
// NOTE: this is the calibration-sensitive block. If danger/safe come out
// reversed or the no-data cutoff is wrong on real frames, adjust HERE only.
float jetToT(vec3 c) {
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  float chroma = mx - mn;

  // Near-gray / background pixels carry no field value → push to far-field.
  if (chroma < kMinSaturation) {
    return 1.0;
  }

  float hue;
  if (mx == c.r) {
    hue = mod((c.g - c.b) / chroma, 6.0);
  } else if (mx == c.g) {
    hue = (c.b - c.r) / chroma + 2.0;
  } else {
    hue = (c.r - c.g) / chroma + 4.0;
  }
  hue /= 6.0; // [0,1): red=0, green=1/3, blue=2/3

  // Map red(0)→t=0 (danger) .. blue(2/3)→t=1 (safe). Hues beyond blue
  // (magenta) are clamped to the far-field end.
  return clamp(hue / (2.0 / 3.0), 0.0, 1.0);
}

void main() {
  vec2 uv = FlutterFragCoord().xy / uSize;
  vec4 src = texture(uSrc, uv);

  float t = jetToT(src.rgb);
  vec3 col = texture(uLut, vec2(t, 0.5)).rgb;

  // Premultiplied alpha (Flutter fragment shader convention).
  fragColor = vec4(col * uOpacity, uOpacity);
}
