import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

/// Single source of truth for the ESDF heatmap palette.
///
/// Ordered danger→safe (t=0 → t=1): dark red core → coral → amber → olive →
/// teal → deep cyan → navy far-field. Rasterized into a 256×1 LUT texture and
/// fed to the `esdf_colormap` fragment shader. Tweak colours here only — the
/// shader samples this gradient, it does not hardcode it.
const LinearGradient _kEsdfHeatmap = LinearGradient(
  colors: [
    Color(0xFF7A1E18), // 0.00 danger core
    Color(0xFFB43A24), // 0.15 danger edge
    Color(0xFFD8903D), // 0.30 warning amber
    Color(0xFF8FA66A), // 0.45 transition olive
    Color(0xFF238B8F), // 0.60 safe teal
    Color(0xFF0B5D7C), // 0.78 far-field cyan
    Color(0xFF06111D), // 1.00 background navy
  ],
  stops: [0.0, 0.15, 0.30, 0.45, 0.60, 0.78, 1.0],
);

/// Renders the ESDF heatmap by colourising the backend's scalar field through
/// the `esdf_colormap` fragment shader using the [_kEsdfHeatmap] palette.
///
/// The source bytes are a single-channel scalar image (0 = danger .. 1 =
/// far-field); the shader maps each pixel through the palette LUT. Until the
/// shader/image are ready (or if the shader asset fails to load) it falls back
/// to [Image.memory], which shows the raw scalar as grayscale — degraded but
/// still readable, and only during the brief one-time shader load.
class EsdfColormapLayer extends StatefulWidget {
  final Uint8List bytes;
  final double opacity;

  const EsdfColormapLayer({
    super.key,
    required this.bytes,
    this.opacity = 0.85,
  });

  @override
  State<EsdfColormapLayer> createState() => _EsdfColormapLayerState();
}

class _EsdfColormapLayerState extends State<EsdfColormapLayer> {
  // Shared across all instances — the program and palette LUT are built once.
  static ui.FragmentProgram? _program;
  static ui.Image? _lut;
  static Future<void>? _assetsFuture;

  ui.FragmentShader? _shader;
  ui.Image? _srcImage;
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    _loadAssets();
    _decode(widget.bytes);
  }

  @override
  void didUpdateWidget(EsdfColormapLayer old) {
    super.didUpdateWidget(old);
    if (!identical(old.bytes, widget.bytes)) {
      _decode(widget.bytes);
    }
  }

  Future<void> _loadAssets() async {
    try {
      _assetsFuture ??= _buildSharedAssets();
      await _assetsFuture;
    } catch (_) {
      // Leave _ready false → graceful fallback to Image.memory.
      return;
    }
    if (!mounted) return;
    final program = _program;
    if (program == null || _lut == null) return;
    setState(() {
      _shader ??= program.fragmentShader();
      _ready = true;
    });
  }

  static Future<void> _buildSharedAssets() async {
    _program =
        await ui.FragmentProgram.fromAsset('shaders/esdf_colormap.frag');
    _lut = await _buildLut();
  }

  static Future<ui.Image> _buildLut() {
    const int width = 256;
    const int height = 1;
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final rect = Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble());
    canvas.drawRect(
      rect,
      Paint()..shader = _kEsdfHeatmap.createShader(rect),
    );
    return recorder.endRecording().toImage(width, height);
  }

  void _decode(Uint8List bytes) {
    ui.decodeImageFromList(bytes, (img) {
      if (!mounted) {
        img.dispose();
        return;
      }
      final old = _srcImage;
      setState(() => _srcImage = img);
      old?.dispose();
    });
  }

  @override
  void dispose() {
    _srcImage?.dispose();
    _shader?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final shader = _shader;
    final lut = _lut;
    final src = _srcImage;

    if (!_ready || shader == null || lut == null || src == null) {
      return Opacity(
        opacity: widget.opacity,
        child: Image.memory(
          widget.bytes,
          fit: BoxFit.fill,
          gaplessPlayback: true,
        ),
      );
    }

    return CustomPaint(
      size: Size.infinite,
      painter: _EsdfShaderPainter(
        shader: shader,
        src: src,
        lut: lut,
        opacity: widget.opacity,
      ),
    );
  }
}

class _EsdfShaderPainter extends CustomPainter {
  final ui.FragmentShader shader;
  final ui.Image src;
  final ui.Image lut;
  final double opacity;

  const _EsdfShaderPainter({
    required this.shader,
    required this.src,
    required this.lut,
    required this.opacity,
  });

  @override
  void paint(Canvas canvas, Size size) {
    shader
      ..setFloat(0, size.width)
      ..setFloat(1, size.height)
      ..setFloat(2, opacity)
      ..setImageSampler(0, src)
      ..setImageSampler(1, lut);
    canvas.drawRect(Offset.zero & size, Paint()..shader = shader);
  }

  @override
  bool shouldRepaint(_EsdfShaderPainter old) =>
      !identical(old.src, src) ||
      !identical(old.lut, lut) ||
      !identical(old.shader, shader) ||
      old.opacity != opacity;
}
