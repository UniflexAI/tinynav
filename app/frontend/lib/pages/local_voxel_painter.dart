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
  final double viewYaw;

  const LocalVoxelPainter({
    required this.points,
    this.trajectory = const [],
    this.globalPath = const [],
    this.footprint = const [],
    this.navTargetPose,
    this.odomPose,
    this.viewYaw = 0.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(
        Offset.zero & size, Paint()..color = const Color(0xFF0F1621));

    final pose = odomPose;
    if (pose == null) {
      _drawEmpty(canvas, size);
      return;
    }

    final scale = math.min(size.width, size.height) / 8.0;
    final center = Offset(size.width / 2, size.height * 0.62);

    _drawGroundGrid(canvas, center, scale);

    final zBase = pose.z ?? 0.0;
    final zRange = _zRange(points, zBase);
    final sorted = [...points]
      ..sort((a, b) => _depth(a, pose).compareTo(_depth(b, pose)));
    for (final p in sorted) {
      final rz = p.z - zBase;
      final c = _project3d(center, scale, p.x - pose.x, p.y - pose.y, rz);
      if (c.dx < -10 ||
          c.dx > size.width + 10 ||
          c.dy < -10 ||
          c.dy > size.height + 10) continue;
      final color = _heightColor(_zNorm(rz, zRange.$1, zRange.$2));
      canvas.drawCircle(c, 2.25, Paint()..color = color.withOpacity(0.92));
    }

    _drawPath(canvas, center, scale, globalPath, const Color(0xFF69F0AE), 2.6);
    _drawPath(canvas, center, scale, trajectory, Colors.cyanAccent, 2.6);
    _drawFootprint(canvas, center, scale);
    if (navTargetPose != null)
      _drawNavTarget(canvas, center, scale, navTargetPose!);
    _drawRobotArrow(canvas, center, scale, pose.yaw);
  }

  Offset _project3d(
      Offset center, double scale, double dx, double dy, double z) {
    // Isometric-ish projection. viewYaw rotates the world around +Z before
    // projection so users can inspect the local voxel map from any side.
    final cosYaw = math.cos(viewYaw);
    final sinYaw = math.sin(viewYaw);
    final rx = dx * cosYaw - dy * sinYaw;
    final ry = dx * sinYaw + dy * cosYaw;
    final sx = center.dx + (rx - ry) * scale * 0.72;
    final sy = center.dy - (rx + ry) * scale * 0.36 - z * scale * 0.75;
    return Offset(sx, sy);
  }

  double _depth(VoxelPoint p, Pose pose) {
    final dx = p.x - pose.x;
    final dy = p.y - pose.y;
    final cosYaw = math.cos(viewYaw);
    final sinYaw = math.sin(viewYaw);
    final rx = dx * cosYaw - dy * sinYaw;
    final ry = dx * sinYaw + dy * cosYaw;
    return rx + ry + p.z;
  }

  (double, double) _zRange(List<VoxelPoint> pts, double zBase) {
    if (pts.isEmpty) return (-0.4, 0.8);
    final zs = pts.map((p) => p.z - zBase).toList()..sort();
    final loIdx = (zs.length * 0.05).floor().clamp(0, zs.length - 1).toInt();
    final hiIdx = (zs.length * 0.95).floor().clamp(0, zs.length - 1).toInt();
    final lo = zs[loIdx];
    final hi = zs[hiIdx];
    final mid = (lo + hi) * 0.5;
    final span = math.max(hi - lo, 0.6);
    return (mid - span * 0.5, mid + span * 0.5);
  }

  double _zNorm(double z, double zMin, double zMax) =>
      ((z - zMin) / math.max(zMax - zMin, 0.001)).clamp(0.0, 1.0).toDouble();

  Color _heightColor(double t) {
    const stops = [
      Color(0xFF064E3B), // low: dark green
      Color(0xFF22C55E), // lower-mid: vivid green
      Color(0xFFFACC15), // mid: yellow
      Color(0xFFF97316), // high: orange
      Color(0xFFEF4444), // highest: red
    ];
    final scaled = (t * (stops.length - 1)).clamp(0.0, stops.length - 1.0);
    final i = scaled.floor().clamp(0, stops.length - 2).toInt();
    final localT = scaled - i;
    return Color.lerp(stops[i], stops[i + 1], localT)!;
  }

  TrajPoint _rel(TrajPoint p, Pose pose) =>
      TrajPoint(p.x - pose.x, p.y - pose.y);

  void _drawGroundGrid(Canvas canvas, Offset center, double scale) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.07)
      ..strokeWidth = 1;
    for (var i = -4; i <= 4; i++) {
      final a = _project3d(center, scale, i.toDouble(), -4, 0);
      final b = _project3d(center, scale, i.toDouble(), 4, 0);
      final c = _project3d(center, scale, -4, i.toDouble(), 0);
      final d = _project3d(center, scale, 4, i.toDouble(), 0);
      canvas.drawLine(a, b, paint);
      canvas.drawLine(c, d, paint);
    }
  }

  void _drawPath(Canvas canvas, Offset center, double scale,
      List<TrajPoint> pts, Color color, double strokeWidth) {
    final pose = odomPose;
    if (pose == null || pts.length < 2) return;
    final path = Path();
    for (var i = 0; i < pts.length; i++) {
      final rp = _rel(pts[i], pose);
      final c = _project3d(center, scale, rp.x, rp.y, 0.06);
      if (i == 0) {
        path.moveTo(c.dx, c.dy);
      } else {
        path.lineTo(c.dx, c.dy);
      }
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = color.withOpacity(0.92)
        ..strokeWidth = strokeWidth
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round,
    );
  }

  void _drawFootprint(Canvas canvas, Offset center, double scale) {
    final pose = odomPose;
    if (pose == null || footprint.length < 3) return;
    final pts = footprint.map((p) {
      final rp = _rel(p, pose);
      return _project3d(center, scale, rp.x, rp.y, 0.08);
    }).toList();
    final path = Path()..moveTo(pts.first.dx, pts.first.dy);
    for (final p in pts.skip(1)) {
      path.lineTo(p.dx, p.dy);
    }
    path.close();
    canvas.drawPath(
        path, Paint()..color = const Color(0xFF64B5F6).withOpacity(0.22));
    canvas.drawPath(
      path,
      Paint()
        ..color = const Color(0xFF29B6F6)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.6,
    );
  }

  void _drawNavTarget(
      Canvas canvas, Offset center, double scale, TrajPoint target) {
    final pose = odomPose;
    if (pose == null) return;
    final rp = _rel(target, pose);
    final c = _project3d(center, scale, rp.x, rp.y, 0.12);
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

  void _drawRobotArrow(Canvas canvas, Offset center, double scale, double yaw) {
    final cosY = math.cos(yaw);
    final sinY = math.sin(yaw);
    final tip = _project3d(center, scale, cosY * 0.22, sinY * 0.22, 0.18);
    final left = _project3d(center, scale, -sinY * 0.10, cosY * 0.10, 0.18);
    final right = _project3d(center, scale, sinY * 0.10, -cosY * 0.10, 0.18);
    final base = _project3d(center, scale, -cosY * 0.08, -sinY * 0.08, 0.18);
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
          ..strokeWidth = 0.8);
  }

  void _drawEmpty(Canvas canvas, Size size) {
    final tp = TextPainter(
      text: const TextSpan(
        text: 'Waiting for 3D voxel data…',
        style: TextStyle(color: Colors.white54, fontSize: 13),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: size.width);
    tp.paint(canvas,
        Offset((size.width - tp.width) / 2, (size.height - tp.height) / 2));
  }

  @override
  bool shouldRepaint(covariant LocalVoxelPainter oldDelegate) =>
      oldDelegate.points != points ||
      oldDelegate.trajectory != trajectory ||
      oldDelegate.globalPath != globalPath ||
      oldDelegate.footprint != footprint ||
      oldDelegate.navTargetPose != navTargetPose ||
      oldDelegate.odomPose != odomPose ||
      oldDelegate.viewYaw != viewYaw;
}
