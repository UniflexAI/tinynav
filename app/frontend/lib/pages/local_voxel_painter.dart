import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/models.dart';

class LocalVoxelPainter extends CustomPainter {
  final List<VoxelPoint> points;
  final Pose? odomPose;

  const LocalVoxelPainter({required this.points, this.odomPose});

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
      oldDelegate.points != points || oldDelegate.odomPose != odomPose;
}
