import 'package:flutter/material.dart';

import '../core/models.dart';

/// Paints robot pose (blue arrow) and POI markers (amber dots) on top of the
/// map PNG.  Coordinate conversion: world → image-pixel → canvas-pixel.
class MapOverlayPainter extends CustomPainter {
  final MapInfo mapInfo;
  final Pose? pose;
  final List<Poi> pois;

  const MapOverlayPainter({
    required this.mapInfo,
    this.pose,
    this.pois = const [],
  });

  /// World (x, y) → image pixel, matching the flip in map_renderer.py:
  ///   img = np.flipud(img.transpose(1, 0, 2))  → row 0 = max-Y in world
  Offset _worldToImage(double wx, double wy) {
    final px = (wx - mapInfo.originX) / mapInfo.resolution;
    final py = mapInfo.height - (wy - mapInfo.originY) / mapInfo.resolution;
    return Offset(px, py);
  }

  /// Image pixel → canvas pixel (accounts for display scaling).
  Offset _imageToCanvas(Offset img, Size canvas) {
    return Offset(
      img.dx * canvas.width / mapInfo.width,
      img.dy * canvas.height / mapInfo.height,
    );
  }

  @override
  void paint(Canvas canvas, Size size) {
    _drawPois(canvas, size);
    _drawPose(canvas, size);
  }

  void _drawPois(Canvas canvas, Size size) {
    final fill = Paint()..color = Colors.amber..style = PaintingStyle.fill;
    final border = Paint()
      ..color = Colors.orange.shade800
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    const labelStyle = TextStyle(
      color: Colors.black87,
      fontSize: 11,
      fontWeight: FontWeight.bold,
      shadows: [Shadow(blurRadius: 3, color: Colors.white)],
    );

    for (final poi in pois) {
      final c = _imageToCanvas(_worldToImage(poi.x, poi.y), size);
      canvas.drawCircle(c, 7, fill);
      canvas.drawCircle(c, 7, border);

      final tp = TextPainter(
        text: TextSpan(text: poi.name, style: labelStyle),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, c + const Offset(10, -6));
    }
  }

  void _drawPose(Canvas canvas, Size size) {
    if (pose == null) return;
    final c = _imageToCanvas(_worldToImage(pose!.x, pose!.y), size);

    canvas.save();
    canvas.translate(c.dx, c.dy);
    canvas.rotate(pose!.yaw);
    _drawDog(canvas);
    canvas.restore();
  }

  // Top-down robot dog. Local frame: -y = forward (head), +y = back (tail).
  void _drawDog(Canvas canvas) {
    const body  = Color(0xFF1565C0);
    const dark  = Color(0xFF0D47A1);
    const light = Color(0xFF42A5F5);

    final fill    = (Color c) => Paint()..color = c..style = PaintingStyle.fill;
    final outline = Paint()
      ..color = Colors.white.withOpacity(0.85)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;

    // ── Body ──────────────────────────────────────────────────────────────
    final bodyRR = RRect.fromRectAndRadius(
      const Rect.fromLTWH(-6, -7, 12, 11), const Radius.circular(3));
    canvas.drawRRect(bodyRR, fill(body));
    canvas.drawRRect(bodyRR, outline);

    // ── Neck ──────────────────────────────────────────────────────────────
    canvas.drawRect(const Rect.fromLTWH(-3, -12, 6, 5), fill(body));

    // ── Head ──────────────────────────────────────────────────────────────
    final headRR = RRect.fromRectAndRadius(
      const Rect.fromLTWH(-5, -18, 10, 7), const Radius.circular(3));
    canvas.drawRRect(headRR, fill(light));
    canvas.drawRRect(headRR, outline);

    // ── Ears ──────────────────────────────────────────────────────────────
    canvas.drawRRect(
      RRect.fromRectAndRadius(const Rect.fromLTWH(-7, -20, 3, 4), const Radius.circular(1)),
      fill(dark));
    canvas.drawRRect(
      RRect.fromRectAndRadius(const Rect.fromLTWH(4, -20, 3, 4), const Radius.circular(1)),
      fill(dark));

    // ── Eyes (two white dots) ──────────────────────────────────────────────
    canvas.drawCircle(const Offset(-2, -15), 1.2, fill(Colors.white));
    canvas.drawCircle(const Offset( 2, -15), 1.2, fill(Colors.white));

    // ── Front legs ────────────────────────────────────────────────────────
    for (final x in [-10.0, 6.0]) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(Rect.fromLTWH(x, -6, 4, 9), const Radius.circular(2)),
        fill(dark));
      canvas.drawRRect(
        RRect.fromRectAndRadius(Rect.fromLTWH(x, -6, 4, 9), const Radius.circular(2)),
        outline);
    }

    // ── Back legs ─────────────────────────────────────────────────────────
    for (final x in [-10.0, 6.0]) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(Rect.fromLTWH(x, 2, 4, 9), const Radius.circular(2)),
        fill(dark));
      canvas.drawRRect(
        RRect.fromRectAndRadius(Rect.fromLTWH(x, 2, 4, 9), const Radius.circular(2)),
        outline);
    }

    // ── Tail ──────────────────────────────────────────────────────────────
    final tail = Path()
      ..moveTo(-2, 4)
      ..quadraticBezierTo(5, 6, 4, 12)
      ..lineTo(2, 12)
      ..quadraticBezierTo(3, 7, -2, 6)
      ..close();
    canvas.drawPath(tail, fill(dark));
  }

  @override
  bool shouldRepaint(MapOverlayPainter old) =>
      old.pose != pose || old.pois != pois;
}
