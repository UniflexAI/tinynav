import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/models.dart';

/// Paints robot pose (arrow), POI markers, and global path on the SLAM map PNG.
/// Coordinate conversion: world → image-pixel → canvas-pixel.
class MapOverlayPainter extends CustomPainter {
  final MapInfo mapInfo;
  final Pose? pose;
  final List<Poi> pois;
  final Poi? starredPoi;
  final List<TrajPoint> globalPath;
  final List<TrajPoint> finalGlobalPath;
  final bool showGlobalPath;

  const MapOverlayPainter({
    required this.mapInfo,
    this.pose,
    this.pois = const [],
    this.starredPoi,
    this.globalPath = const [],
    this.finalGlobalPath = const [],
    this.showGlobalPath = true,
  });

  /// World (x, y) → image pixel.
  /// Matches map_renderer.py: img = np.flipud(img) → row 0 = max-Y in world.
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
    if (showGlobalPath) {
      _drawPath(
        canvas,
        size,
        finalGlobalPath,
        const Color(0xFFFF5252),
        strokeWidth: 3.4,
        goalRadius: 8,
      );
      _drawPath(
        canvas,
        size,
        globalPath,
        const Color(0xFF69F0AE),
        strokeWidth: 3.0,
        goalRadius: 7,
      );
    }
    _drawPois(canvas, size);
    _drawStarredPoi(canvas, size);
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

  Path _starPath(Offset center, double outerRadius, double innerRadius) {
    final path = Path();
    const points = 5;
    const startAngle = -math.pi / 2;
    for (var i = 0; i < points * 2; i++) {
      final radius = i.isEven ? outerRadius : innerRadius;
      final angle = startAngle + i * math.pi / points;
      final p = Offset(
        center.dx + math.cos(angle) * radius,
        center.dy + math.sin(angle) * radius,
      );
      if (i == 0) {
        path.moveTo(p.dx, p.dy);
      } else {
        path.lineTo(p.dx, p.dy);
      }
    }
    path.close();
    return path;
  }

  void _drawStarredPoi(Canvas canvas, Size size) {
    final poi = starredPoi;
    if (poi == null) return;

    final c = _imageToCanvas(_worldToImage(poi.x, poi.y), size);
    final star = _starPath(c, 14, 6);
    canvas.drawPath(
      star,
      Paint()
        ..color = const Color(0xFFFFD54F)
        ..style = PaintingStyle.fill,
    );
    canvas.drawPath(
      star,
      Paint()
        ..color = Colors.black87
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0,
    );

    const labelStyle = TextStyle(
      color: Color(0xFFFFF8E1),
      fontSize: 12,
      fontWeight: FontWeight.bold,
      shadows: [Shadow(blurRadius: 4, color: Colors.black)],
    );
    final tp = TextPainter(
      text: TextSpan(text: poi.name, style: labelStyle),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, c + const Offset(16, -8));
  }

  void _drawPose(Canvas canvas, Size size) {
    if (pose == null) return;
    final c = _imageToCanvas(_worldToImage(pose!.x, pose!.y), size);

    // Map canvas: screen-right = world+X, screen-up = world+Y (Y-flipped).
    // yaw=0 → arrow points right (+X direction).
    final cosY = math.cos(pose!.yaw);
    final sinY = math.sin(pose!.yaw);
    final tip   = Offset(c.dx + cosY * 14, c.dy - sinY * 14);
    final left  = Offset(c.dx - sinY *  6, c.dy - cosY *  6);
    final right = Offset(c.dx + sinY *  6, c.dy + cosY *  6);
    final base  = Offset(c.dx - cosY *  5, c.dy + sinY *  5);

    final arrow = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(left.dx, left.dy)
      ..lineTo(base.dx, base.dy)
      ..lineTo(right.dx, right.dy)
      ..close();
    canvas.drawPath(arrow, Paint()..color = Colors.white);
    canvas.drawPath(arrow,
        Paint()..color = Colors.black45..style = PaintingStyle.stroke..strokeWidth = 1.0);
  }

  /// Draw a path from map_node (already in map frame).
  /// The last point is the nav target — drawn as a distinct marker.
  void _drawPath(
    Canvas canvas,
    Size size,
    List<TrajPoint> points,
    Color color, {
    required double strokeWidth,
    required double goalRadius,
  }) {
    if (points.length < 2) return;

    final linePaint = Paint()
      ..color = color.withOpacity(0.9)
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final path = Path();
    bool first = true;
    for (final pt in points) {
      final c = _imageToCanvas(_worldToImage(pt.x, pt.y), size);
      if (first) {
        path.moveTo(c.dx, c.dy);
        first = false;
      } else {
        path.lineTo(c.dx, c.dy);
      }
    }
    canvas.drawPath(path, linePaint);

    // Target marker at path end.
    final goal = points.last;
    final gc = _imageToCanvas(_worldToImage(goal.x, goal.y), size);
    canvas.drawCircle(gc, goalRadius, Paint()..color = color);
    canvas.drawCircle(gc, goalRadius,
        Paint()..color = Colors.white..style = PaintingStyle.stroke..strokeWidth = 2.0);
    // Inner dot.
    canvas.drawCircle(gc, 3, Paint()..color = Colors.white);
  }

  @override
  bool shouldRepaint(MapOverlayPainter old) =>
      old.pose != pose ||
      old.pois != pois ||
      old.starredPoi != starredPoi ||
      old.globalPath != globalPath ||
      old.finalGlobalPath != finalGlobalPath ||
      old.showGlobalPath != showGlobalPath;
}
