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

    // Blue circle
    canvas.drawCircle(c, 10, Paint()..color = Colors.blue);

    // White direction arrow (points in +yaw direction)
    canvas.save();
    canvas.translate(c.dx, c.dy);
    canvas.rotate(pose!.yaw);
    final arrow = Path()
      ..moveTo(0, -14)
      ..lineTo(5, 2)
      ..lineTo(-5, 2)
      ..close();
    canvas.drawPath(arrow, Paint()..color = Colors.white);
    canvas.restore();
  }

  @override
  bool shouldRepaint(MapOverlayPainter old) =>
      old.pose != pose || old.pois != pois;
}
