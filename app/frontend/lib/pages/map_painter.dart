import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/models.dart';

/// Paints robot pose, POI markers, trajectory and global path on top of the
/// map PNG.  Coordinate conversion: world → image-pixel → canvas-pixel.
class MapOverlayPainter extends CustomPainter {
  final MapInfo mapInfo;
  final Pose? pose;
  final List<Poi> pois;
  final List<TrajPoint> trajectory;
  final List<TrajPoint> globalPath;
  final Pose? odomPose;
  final bool showTrajectory;
  final bool showGlobalPath;

  const MapOverlayPainter({
    required this.mapInfo,
    this.pose,
    this.pois = const [],
    this.trajectory = const [],
    this.globalPath = const [],
    this.odomPose,
    this.showTrajectory = false,
    this.showGlobalPath = true,
  });

  /// World (x, y) → image pixel, matching the flip in map_renderer.py:
  ///   img = np.flipud(img)  → row 0 = max-Y in world
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
    if (showGlobalPath) _drawGlobalPath(canvas, size);
    if (showTrajectory) _drawTrajectory(canvas, size);
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

    // Map canvas: screen-right = world+X, screen-up = world+Y (Y-flipped).
    // Use explicit geometry: yaw=0 → arrow points right.
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

  /// Draw planning trajectory (odom frame). Convert to map frame via pose − odomPose.
  void _drawTrajectory(Canvas canvas, Size size) {
    if (trajectory.length < 2 || pose == null || odomPose == null) return;
    // Offset from odom frame to map frame.
    final dx = pose!.x - odomPose!.x;
    final dy = pose!.y - odomPose!.y;

    final paint = Paint()
      ..color = Colors.cyanAccent.withOpacity(0.85)
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final path = Path();
    bool first = true;
    for (final pt in trajectory) {
      final c = _imageToCanvas(_worldToImage(pt.x + dx, pt.y + dy), size);
      if (first) {
        path.moveTo(c.dx, c.dy);
        first = false;
      } else {
        path.lineTo(c.dx, c.dy);
      }
    }
    canvas.drawPath(path, paint);

    // Goal dot.
    final goal = trajectory.last;
    final gc = _imageToCanvas(_worldToImage(goal.x + dx, goal.y + dy), size);
    canvas.drawCircle(gc, 5, Paint()..color = Colors.cyanAccent);
  }

  /// Draw global path from map_node (already in map frame).
  void _drawGlobalPath(Canvas canvas, Size size) {
    if (globalPath.length < 2) return;

    final paint = Paint()
      ..color = const Color(0xFF69F0AE).withOpacity(0.9)
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final path = Path();
    bool first = true;
    for (final pt in globalPath) {
      final c = _imageToCanvas(_worldToImage(pt.x, pt.y), size);
      if (first) {
        path.moveTo(c.dx, c.dy);
        first = false;
      } else {
        path.lineTo(c.dx, c.dy);
      }
    }
    canvas.drawPath(path, paint);

    // Goal dot.
    final goal = globalPath.last;
    final gc = _imageToCanvas(_worldToImage(goal.x, goal.y), size);
    canvas.drawCircle(gc, 6, Paint()..color = const Color(0xFF69F0AE));
    canvas.drawCircle(gc, 6,
        Paint()..color = Colors.white..style = PaintingStyle.stroke..strokeWidth = 1.5);
  }

  @override
  bool shouldRepaint(MapOverlayPainter old) =>
      old.pose != pose ||
      old.pois != pois ||
      old.trajectory != trajectory ||
      old.globalPath != globalPath ||
      old.odomPose != odomPose ||
      old.showTrajectory != showTrajectory ||
      old.showGlobalPath != showGlobalPath;
}
