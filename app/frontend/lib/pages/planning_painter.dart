import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/models.dart';

/// Renders robot arrow, local trajectory, and global path on the local planning canvas.
/// The canvas maps to the planning grid: robot is always at center.
/// Global path arrives pre-converted to odom frame by the backend.
class LocalPlanningPainter extends CustomPainter {
  final List<TrajPoint> trajectory;
  final List<TrajPoint> globalPath;
  final GridInfo? gridInfo;
  final Pose? odomPose;
  final bool showTrajectory;
  final bool showGlobalPath;

  const LocalPlanningPainter({
    required this.trajectory,
    this.globalPath = const [],
    this.gridInfo,
    this.odomPose,
    this.showTrajectory = true,
    this.showGlobalPath = true,
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

    if (showGlobalPath) _drawGlobalPath(canvas, cx, cy, scaleX, scaleY, pose);

    if (showTrajectory) _drawTrajectory(canvas, cx, cy, scaleX, scaleY, pose);

    _drawRobotArrow(canvas, Offset(cx, cy), pose?.yaw ?? 0.0);
  }

  void _drawTrajectory(Canvas canvas, double cx, double cy,
      double scaleX, double scaleY, Pose? pose) {
    if (trajectory.length < 2 || pose == null) return;

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
      final py = cy - (pt.y - pose.y) * scaleY;
      if (first) {
        path.moveTo(px, py);
        first = false;
      } else {
        path.lineTo(px, py);
      }
    }
    canvas.drawPath(path, paint);

    final goal = trajectory.last;
    canvas.drawCircle(
      Offset(cx + (goal.x - pose.x) * scaleX, cy - (goal.y - pose.y) * scaleY),
      5,
      Paint()..color = Colors.cyanAccent,
    );
  }

  /// Global path is already in odom frame (backend transforms via exact T_odom_map).
  void _drawGlobalPath(Canvas canvas, double cx, double cy,
      double scaleX, double scaleY, Pose? odomPose) {
    if (globalPath.length < 2 || odomPose == null) return;

    Offset toCanvas(TrajPoint pt) => Offset(
      cx + (pt.x - odomPose.x) * scaleX,
      cy - (pt.y - odomPose.y) * scaleY,
    );

    final linePaint = Paint()
      ..color = const Color(0xFF69F0AE).withOpacity(0.9)
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final path = Path();
    bool first = true;
    for (final pt in globalPath) {
      final c = toCanvas(pt);
      if (first) {
        path.moveTo(c.dx, c.dy);
        first = false;
      } else {
        path.lineTo(c.dx, c.dy);
      }
    }
    canvas.drawPath(path, linePaint);

    // Target marker at path end.
    final gc = toCanvas(globalPath.last);
    canvas.drawCircle(gc, 7, Paint()..color = const Color(0xFF69F0AE));
    canvas.drawCircle(gc, 7,
        Paint()..color = Colors.white..style = PaintingStyle.stroke..strokeWidth = 2.0);
    canvas.drawCircle(gc, 3, Paint()..color = Colors.white);
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
      globalPath != old.globalPath ||
      gridInfo != old.gridInfo ||
      odomPose != old.odomPose ||
      showTrajectory != old.showTrajectory ||
      showGlobalPath != old.showGlobalPath;
}
