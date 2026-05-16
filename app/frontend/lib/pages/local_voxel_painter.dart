import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/models.dart';

class LocalVoxelPainter extends CustomPainter {
  final List<VoxelPoint> points;
  final List<TrajPoint> trajectory;
  final List<TrajPoint> globalPath;
  final List<TrajPoint> footprint;
  final TrajPoint? navTargetPose;
  final Pose? odomPose;

  const LocalVoxelPainter({
    required this.points,
    this.trajectory = const [],
    this.globalPath = const [],
    this.footprint = const [],
    this.navTargetPose,
    this.odomPose,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final bg = Paint()..color = const Color(0xFF0F1621);
    canvas.drawRect(Offset.zero & size, bg);

    final gridPaint = Paint()
      ..color = Colors.white.withOpacity(0.07)
      ..strokeWidth = 1;
    for (var i = 1; i < 4; i++) {
      final x = size.width * i / 4;
      final y = size.height * i / 4;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    final pose = odomPose;
    if (points.isEmpty || pose == null) {
      _drawEmpty(canvas, size);
      return;
    }

    const worldW = 10.0;
    const worldH = 10.0;
    final scaleX = size.width / worldW;
    final scaleY = size.height / worldH;
    final center = Offset(size.width / 2, size.height / 2);

    final sorted = [...points]..sort((a, b) => a.z.compareTo(b.z));
    for (final p in sorted) {
      final sx = center.dx + (p.x - pose.x) * scaleX;
      final sy = center.dy - (p.y - pose.y) * scaleY - (p.z * math.min(scaleX, scaleY) * 0.08);
      if (sx < -8 || sx > size.width + 8 || sy < -8 || sy > size.height + 8) continue;

      final zNorm = ((p.z + 0.4) / 1.2).clamp(0.0, 1.0);
      final color = Color.lerp(const Color(0xFF25D0FF), const Color(0xFFFFB020), zNorm)!;
      canvas.drawCircle(Offset(sx, sy), 2.0, Paint()..color = color.withOpacity(0.82));
    }

    _drawPath(canvas, center, scaleX, scaleY, globalPath, const Color(0xFF69F0AE), 2.6);
    _drawPath(canvas, center, scaleX, scaleY, trajectory, Colors.cyanAccent, 2.6);
    _drawFootprint(canvas, center, scaleX, scaleY);
    if (navTargetPose != null) {
      _drawNavTarget(canvas, center, scaleX, scaleY, navTargetPose!);
    }
    _drawRobotArrow(canvas, center, pose.yaw);
  }

  void _drawEmpty(Canvas canvas, Size size) {
    final tp = TextPainter(
      text: const TextSpan(
        text: 'Waiting for 3D voxel data…',
        style: TextStyle(color: Colors.white54, fontSize: 13),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: size.width);
    tp.paint(canvas, Offset((size.width - tp.width) / 2, (size.height - tp.height) / 2));
  }

  Offset _toCanvas(Offset center, double scaleX, double scaleY, TrajPoint p, Pose pose) =>
      Offset(center.dx + (p.x - pose.x) * scaleX, center.dy - (p.y - pose.y) * scaleY);

  void _drawPath(
    Canvas canvas,
    Offset center,
    double scaleX,
    double scaleY,
    List<TrajPoint> pathPoints,
    Color color,
    double strokeWidth,
  ) {
    final pose = odomPose;
    if (pose == null || pathPoints.length < 2) return;
    final path = Path();
    for (var i = 0; i < pathPoints.length; i++) {
      final c = _toCanvas(center, scaleX, scaleY, pathPoints[i], pose);
      if (i == 0) {
        path.moveTo(c.dx, c.dy);
      } else {
        path.lineTo(c.dx, c.dy);
      }
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = color.withOpacity(0.9)
        ..strokeWidth = strokeWidth
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round,
    );
    final end = _toCanvas(center, scaleX, scaleY, pathPoints.last, pose);
    canvas.drawCircle(end, 4.5, Paint()..color = color);
  }

  void _drawFootprint(Canvas canvas, Offset center, double scaleX, double scaleY) {
    final pose = odomPose;
    if (pose == null || footprint.length < 3) return;
    final pts = footprint.map((p) => _toCanvas(center, scaleX, scaleY, p, pose)).toList();
    final path = Path()..moveTo(pts.first.dx, pts.first.dy);
    for (final p in pts.skip(1)) {
      path.lineTo(p.dx, p.dy);
    }
    path.close();
    canvas.drawPath(path, Paint()..color = const Color(0xFF64B5F6).withOpacity(0.22));
    canvas.drawPath(
      path,
      Paint()
        ..color = const Color(0xFF29B6F6)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.6,
    );
  }

  void _drawNavTarget(Canvas canvas, Offset center, double scaleX, double scaleY, TrajPoint target) {
    final pose = odomPose;
    if (pose == null) return;
    final c = _toCanvas(center, scaleX, scaleY, target, pose);
    canvas.drawCircle(
      c,
      8,
      Paint()
        ..color = const Color(0xFFFF6D00)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.4,
    );
    canvas.drawCircle(c, 3, Paint()..color = const Color(0xFFFF6D00));
  }

  void _drawRobotArrow(Canvas canvas, Offset center, double yaw) {
    final cosY = math.cos(yaw);
    final sinY = math.sin(yaw);

    final tip = Offset(center.dx + cosY * 8, center.dy - sinY * 8);
    final left = Offset(center.dx - sinY * 3.5, center.dy - cosY * 3.5);
    final right = Offset(center.dx + sinY * 3.5, center.dy + cosY * 3.5);
    final base = Offset(center.dx - cosY * 3, center.dy + sinY * 3);

    final path = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(left.dx, left.dy)
      ..lineTo(base.dx, base.dy)
      ..lineTo(right.dx, right.dy)
      ..close();

    canvas.drawPath(path, Paint()..color = Colors.white);
    canvas.drawPath(
      path,
      Paint()
        ..color = Colors.black45
        ..style = PaintingStyle.stroke
        ..strokeWidth = 0.8,
    );
  }

  @override
  bool shouldRepaint(covariant LocalVoxelPainter oldDelegate) =>
      oldDelegate.points != points ||
      oldDelegate.trajectory != trajectory ||
      oldDelegate.globalPath != globalPath ||
      oldDelegate.footprint != footprint ||
      oldDelegate.navTargetPose != navTargetPose ||
      oldDelegate.odomPose != odomPose;
}
