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

    final cosYaw = math.cos(-pose.yaw);
    final sinYaw = math.sin(-pose.yaw);
    const rangeM = 3.0;
    final scale = math.min(size.width, size.height) / (rangeM * 2.0);
    final center = Offset(size.width / 2, size.height * 0.58);

    final sorted = [...points]..sort((a, b) => a.z.compareTo(b.z));
    for (final p in sorted) {
      final dx = p.x - pose.x;
      final dy = p.y - pose.y;
      final localX = dx * cosYaw - dy * sinYaw;
      final localY = dx * sinYaw + dy * cosYaw;
      if (localX.abs() > rangeM || localY.abs() > rangeM) continue;

      final sx = center.dx + localX * scale;
      final sy = center.dy - localY * scale - (p.z * scale * 0.22);
      final zNorm = ((p.z + 0.4) / 1.2).clamp(0.0, 1.0);
      final color = Color.lerp(const Color(0xFF25D0FF), const Color(0xFFFFB020), zNorm)!;
      canvas.drawCircle(Offset(sx, sy), 2.0, Paint()..color = color.withOpacity(0.82));
    }

    final robotPaint = Paint()..color = Colors.white;
    final path = Path()
      ..moveTo(center.dx, center.dy - 12)
      ..lineTo(center.dx - 8, center.dy + 10)
      ..lineTo(center.dx + 8, center.dy + 10)
      ..close();
    canvas.drawPath(path, robotPaint);
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

  @override
  bool shouldRepaint(covariant LocalVoxelPainter oldDelegate) =>
      oldDelegate.points != points || oldDelegate.odomPose != odomPose;
}
