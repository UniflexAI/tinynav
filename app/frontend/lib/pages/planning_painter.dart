import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/models.dart';

/// Renders robot arrow + planned trajectory on a local-centric canvas.
/// The canvas maps to the planning grid: robot is always at center.
class LocalPlanningPainter extends CustomPainter {
  final List<TrajPoint> trajectory;
  final GridInfo? gridInfo;
  final Pose? odomPose;

  const LocalPlanningPainter({
    required this.trajectory,
    this.gridInfo,
    this.odomPose,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;

    final gi = gridInfo;
    final pose = odomPose;

    // World coverage of the grid in meters (default 10 m × 10 m).
    final worldW = gi != null ? gi.width * gi.resolution : 10.0;
    final worldH = gi != null ? gi.height * gi.resolution : 10.0;
    final scaleX = size.width / worldW;
    final scaleY = size.height / worldH;

    // Draw planned trajectory.
    if (trajectory.length >= 2 && pose != null) {
      final paint = Paint()
        ..color = Colors.cyanAccent.withOpacity(0.85)
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;

      final path = Path();
      bool first = true;
      for (final pt in trajectory) {
        final px = cx + (pt.x - pose.x) * scaleX;
        final py = cy - (pt.y - pose.y) * scaleY; // y-flip: ROS +y = canvas up
        if (first) {
          path.moveTo(px, py);
          first = false;
        } else {
          path.lineTo(px, py);
        }
      }
      canvas.drawPath(path, paint);

      // Draw a dot at the trajectory goal.
      final goal = trajectory.last;
      final gx = cx + (goal.x - pose.x) * scaleX;
      final gy = cy - (goal.y - pose.y) * scaleY;
      canvas.drawCircle(
        Offset(gx, gy),
        5,
        Paint()..color = Colors.cyanAccent,
      );
    }

    _drawRobotArrow(canvas, Offset(cx, cy), pose?.yaw ?? 0.0);
  }

  void _drawRobotArrow(Canvas canvas, Offset center, double yaw) {
    final cosY = math.cos(yaw);
    final sinY = math.sin(yaw);

    final tip   = Offset(center.dx + cosY * 14, center.dy - sinY * 14);
    final left  = Offset(center.dx - sinY *  6, center.dy - cosY *  6);
    final right = Offset(center.dx + sinY *  6, center.dy + cosY *  6);
    final base  = Offset(center.dx - cosY *  5, center.dy + sinY *  5);

    final path = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(left.dx, left.dy)
      ..lineTo(base.dx, base.dy)
      ..lineTo(right.dx, right.dy)
      ..close();

    canvas.drawPath(path, Paint()..color = Colors.white);
    canvas.drawPath(path,
        Paint()..color = Colors.black45..style = PaintingStyle.stroke..strokeWidth = 1.0);
  }

  @override
  bool shouldRepaint(LocalPlanningPainter old) =>
      trajectory != old.trajectory ||
      gridInfo != old.gridInfo ||
      odomPose != old.odomPose;
}
