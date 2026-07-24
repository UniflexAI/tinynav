#version 460 core
#include <flutter/runtime_effect.glsl>

// Colourises the ESDF heatmap layer.
//
// The source image (uSrc) is a single-channel scalar field produced by the
// backend: 0 = danger (on/near an obstacle) .. 1 = far-field. This shader maps
// that scalar straight through a 256x1 palette LUT (uLut) built from the
// _kEsdfHeatmap gradient in esdf_colormap_layer.dart, keeping the palette a
// single source of truth in Dart while the shader stays generic.
//
// (The scalar used to arrive JET-colourised and was recovered here by hue;
// now the backend inverts the colourmap and ships the raw scalar instead,
// which is smaller and removes the compression artefacts the hue recovery had
// to fight.)
//
// Uniform layout (must match _EsdfShaderPainter in esdf_colormap_layer.dart):
//   setFloat(0,1) -> uSize
//   setFloat(2)   -> uOpacity
//   setImageSampler(0) -> uSrc
//   setImageSampler(1) -> uLut

uniform vec2 uSize;      // draw area in pixels
uniform float uOpacity;  // layer opacity [0,1]
uniform sampler2D uSrc;  // single-channel ESDF scalar, r=0 danger .. r=1 far
uniform sampler2D uLut;  // 256x1 palette, x=0 danger .. x=1 far-field

out vec4 fragColor;

void main() {
  vec2 uv = FlutterFragCoord().xy / uSize;
  float t = texture(uSrc, uv).r;   // 0 = danger .. 1 = far-field
  vec3 col = texture(uLut, vec2(t, 0.5)).rgb;

  // Premultiplied alpha (Flutter fragment shader convention).
  fragColor = vec4(col * uOpacity, uOpacity);
}
