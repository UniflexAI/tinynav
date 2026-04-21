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

    // Draw robot dog at canvas center.
    canvas.save();
    canvas.translate(cx, cy);
    canvas.rotate(pose?.yaw ?? 0.0);
    _drawDog(canvas);
    canvas.restore();
  }

  void _drawDog(Canvas canvas) {
    const body  = Color(0xFFFFFFFF);
    const dark  = Color(0xFFCCCCCC);
    const light = Color(0xFFEEEEEE);

    Paint fill(Color c) => Paint()..color = c..style = PaintingStyle.fill;
    final outline = Paint()
      ..color = Colors.black45
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    // Body
    final bodyRR = RRect.fromRectAndRadius(
        const Rect.fromLTWH(-6, -7, 12, 11), const Radius.circular(3));
    canvas.drawRRect(bodyRR, fill(body));
    canvas.drawRRect(bodyRR, outline);

    // Neck
    canvas.drawRect(const Rect.fromLTWH(-3, -12, 6, 5), fill(body));

    // Head (forward = -y)
    final headRR = RRect.fromRectAndRadius(
        const Rect.fromLTWH(-5, -18, 10, 7), const Radius.circular(3));
    canvas.drawRRect(headRR, fill(light));
    canvas.drawRRect(headRR, outline);

    // Ears
    canvas.drawRRect(RRect.fromRectAndRadius(
        const Rect.fromLTWH(-7, -20, 3, 4), const Radius.circular(1)), fill(dark));
    canvas.drawRRect(RRect.fromRectAndRadius(
        const Rect.fromLTWH(4, -20, 3, 4), const Radius.circular(1)), fill(dark));

    // Eyes
    canvas.drawCircle(const Offset(-2, -15), 1.2, fill(Colors.black54));
    canvas.drawCircle(const Offset( 2, -15), 1.2, fill(Colors.black54));

    // Front legs
    for (final x in [-10.0, 6.0]) {
      final r = RRect.fromRectAndRadius(Rect.fromLTWH(x, -6, 4, 9), const Radius.circular(2));
      canvas.drawRRect(r, fill(dark));
      canvas.drawRRect(r, outline);
    }

    // Back legs
    for (final x in [-10.0, 6.0]) {
      final r = RRect.fromRectAndRadius(Rect.fromLTWH(x, 2, 4, 9), const Radius.circular(2));
      canvas.drawRRect(r, fill(dark));
      canvas.drawRRect(r, outline);
    }

    // Tail
    final tail = Path()
      ..moveTo(-2, 4)
      ..quadraticBezierTo(5, 6, 4, 12)
      ..lineTo(2, 12)
      ..quadraticBezierTo(3, 7, -2, 6)
      ..close();
    canvas.drawPath(tail, fill(dark));
  }

  @override
  bool shouldRepaint(LocalPlanningPainter old) =>
      trajectory != old.trajectory ||
      gridInfo != old.gridInfo ||
      odomPose != old.odomPose;
}
