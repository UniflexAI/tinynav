import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/models.dart';

/// Renders robot arrow, local trajectory, and global path on the local planning canvas.
/// The canvas maps to the planning grid: robot is always at center.
/// Global path arrives pre-converted to odom frame by the backend.
class LocalPlanningPainter extends CustomPainter {
  final List<TrajPoint> trajectory;
  final List<TrajPoint> globalPath;
  final List<TrajPoint> footprint;
  final GridInfo? gridInfo;
  final Pose? odomPose;
  final bool showTrajectory;
  final bool showGlobalPath;
  final TrajPoint? navTargetPose;
  final bool showGrid;
  final bool showFootprint;

  const LocalPlanningPainter({
    required this.trajectory,
    this.globalPath = const [],
    this.footprint = const [],
    this.gridInfo,
    this.odomPose,
    this.showTrajectory = true,
    this.showGlobalPath = true,
    this.navTargetPose,
    this.showGrid = true,
    this.showFootprint = true,
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

    // ── Grid lines ──────────────────────────────────────────────────────
    if (showGrid) _drawGrid(canvas, cx, cy, scaleX, scaleY, worldW, worldH, size);

    if (showGlobalPath) _drawGlobalPath(canvas, cx, cy, scaleX, scaleY, pose);

    if (showTrajectory) _drawTrajectory(canvas, cx, cy, scaleX, scaleY, pose);

    if (navTargetPose != null && pose != null)
      _drawNavTarget(canvas, cx, cy, scaleX, scaleY, pose, navTargetPose!);

    if (showFootprint) _drawFootprint(canvas, cx, cy, scaleX, scaleY, pose);

    // Small arrow on top of everything
    _drawRobotArrow(canvas, Offset(cx, cy), pose?.yaw ?? 0.0);

    // ── Scale bar ──────────────────────────────────────────────────────
    _drawScaleBar(canvas, size, scaleX);

    // ── Compass ────────────────────────────────────────────────────────
    _drawCompass(canvas, size, pose?.yaw ?? 0.0);
  }

  // ── Grid lines (1 m intervals) ────────────────────────────────────────

  void _drawGrid(Canvas canvas, double cx, double cy,
      double scaleX, double scaleY, double worldW, double worldH, Size size) {
    final gridPaint = Paint()
      ..color = const Color(0x1AFFFFFF) // very subtle white
      ..strokeWidth = 0.5;

    final textPainter = (String text) {
      final tp = TextPainter(
        text: TextSpan(
          text: text,
          style: const TextStyle(color: Color(0x55FFFFFF), fontSize: 8),
        ),
        textDirection: TextDirection.ltr,
      );
      tp.layout();
      return tp;
    };

    // Vertical lines (1 m apart)
    final halfWorldW = worldW / 2;
    final startMx = -halfWorldW;
    for (double m = startMx.ceilToDouble(); m <= halfWorldW; m += 1.0) {
      if (m == 0) continue; // skip center — robot arrow is there
      final px = cx + m * scaleX;
      if (px < 0 || px > size.width) continue;
      canvas.drawLine(Offset(px, 0), Offset(px, size.height), gridPaint);
      // label every 2m
      if (m.abs() % 2 == 0) {
        final tp = textPainter('${m.toInt()}m');
        tp.paint(canvas, Offset(px + 2, cy - 10));
      }
    }

    // Horizontal lines (1 m apart)
    final halfWorldH = worldH / 2;
    final startMy = -halfWorldH;
    for (double m = startMy.ceilToDouble(); m <= halfWorldH; m += 1.0) {
      if (m == 0) continue;
      final py = cy - m * scaleY; // y is flipped
      if (py < 0 || py > size.height) continue;
      canvas.drawLine(Offset(0, py), Offset(size.width, py), gridPaint);
      if (m.abs() % 2 == 0) {
        final tp = textPainter('${m.toInt()}m');
        tp.paint(canvas, Offset(cx + 3, py + 1));
      }
    }
  }

  // ── Scale bar (bottom-right area, above compass) ─────────────────────

  void _drawScaleBar(Canvas canvas, Size size, double scaleX) {
    // Pick a nice round scale bar length (1m, 2m, 5m)
    double barWorldM;
    if (scaleX * 1.0 > 30) {
      barWorldM = 1.0;
    } else if (scaleX * 2.0 > 30) {
      barWorldM = 2.0;
    } else {
      barWorldM = 5.0;
    }
    final barPx = barWorldM * scaleX;
    final x0 = size.width - 16 - barPx;
    final y0 = size.height - 44;

    final barPaint = Paint()
      ..color = const Color(0xAAFFFFFF)
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;

    // Main line
    canvas.drawLine(Offset(x0, y0), Offset(x0 + barPx, y0), barPaint);
    // End ticks
    canvas.drawLine(Offset(x0, y0 - 4), Offset(x0, y0 + 4), barPaint);
    canvas.drawLine(Offset(x0 + barPx, y0 - 4), Offset(x0 + barPx, y0 + 4), barPaint);

    // Label
    final tp = TextPainter(
      text: TextSpan(
        text: '${barWorldM.toInt()} m',
        style: const TextStyle(color: Color(0xCCFFFFFF), fontSize: 9, fontWeight: FontWeight.w500),
      ),
      textDirection: TextDirection.ltr,
    );
    tp.layout();
    tp.paint(canvas, Offset(x0 + barPx / 2 - tp.width / 2, y0 - 14));
  }

  // ── Compass (top-right corner) ────────────────────────────────────────

  void _drawCompass(Canvas canvas, Size size, double yaw) {
    const compassRadius = 18.0;
    final center = Offset(size.width - 28, 28);

    // Background circle
    canvas.drawCircle(
      center,
      compassRadius,
      Paint()..color = const Color(0x4D000000),
    );
    canvas.drawCircle(
      center,
      compassRadius,
      Paint()
        ..color = const Color(0x55FFFFFF)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0,
    );

    // North arrow (rotates opposite to robot yaw so that it always points north)
    final northAngle = -(yaw - math.pi / 2); // convert yaw to compass convention
    final tipLen = compassRadius * 0.7;
    final tip = Offset(
      center.dx + math.cos(northAngle) * tipLen,
      center.dy - math.sin(northAngle) * tipLen,
    );

    // Red north half
    final northPath = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(
        center.dx + math.cos(northAngle + math.pi / 2) * 4,
        center.dy - math.sin(northAngle + math.pi / 2) * 4,
      )
      ..lineTo(center.dx, center.dy)
      ..lineTo(
        center.dx + math.cos(northAngle - math.pi / 2) * 4,
        center.dy - math.sin(northAngle - math.pi / 2) * 4,
      )
      ..close();
    canvas.drawPath(northPath, Paint()..color = const Color(0xFFFF5252));

    // South half (white)
    final southAngle = northAngle + math.pi;
    final southTip = Offset(
      center.dx + math.cos(southAngle) * tipLen * 0.5,
      center.dy - math.sin(southAngle) * tipLen * 0.5,
    );
    final southPath = Path()
      ..moveTo(southTip.dx, southTip.dy)
      ..lineTo(
        center.dx + math.cos(southAngle + math.pi / 2) * 4,
        center.dy - math.sin(southAngle + math.pi / 2) * 4,
      )
      ..lineTo(center.dx, center.dy)
      ..lineTo(
        center.dx + math.cos(southAngle - math.pi / 2) * 4,
        center.dy - math.sin(southAngle - math.pi / 2) * 4,
      )
      ..close();
    canvas.drawPath(southPath, Paint()..color = const Color(0xAAFFFFFF));

    // "N" label at tip
    final labelOffset = Offset(
      tip.dx + math.cos(northAngle) * 6 - 3,
      tip.dy - math.sin(northAngle) * 6 - 5,
    );
    final tp = TextPainter(
      text: const TextSpan(
        text: 'N',
        style: TextStyle(color: Color(0xFFFF5252), fontSize: 8, fontWeight: FontWeight.bold),
      ),
      textDirection: TextDirection.ltr,
    );
    tp.layout();
    tp.paint(canvas, labelOffset);
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

  void _drawNavTarget(Canvas canvas, double cx, double cy,
      double scaleX, double scaleY, Pose odomPose, TrajPoint target) {
    final px = cx + (target.x - odomPose.x) * scaleX;
    final py = cy - (target.y - odomPose.y) * scaleY;
    final c = Offset(px, py);
    const r = 10.0;
    const arm = 6.0;
    final ring = Paint()
      ..color = const Color(0xFFFF6D00)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    final cross = Paint()
      ..color = const Color(0xFFFF6D00)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(c, r, ring);
    canvas.drawCircle(c, 3, Paint()..color = const Color(0xFFFF6D00));
    canvas.drawLine(Offset(px - r - arm, py), Offset(px - r + arm, py), cross);
    canvas.drawLine(Offset(px + r - arm, py), Offset(px + r + arm, py), cross);
    canvas.drawLine(Offset(px, py - r - arm), Offset(px, py - r + arm), cross);
    canvas.drawLine(Offset(px, py + r - arm), Offset(px, py + r + arm), cross);
  }

  void _drawRobotArrow(Canvas canvas, Offset center, double yaw) {
    final cosY = math.cos(yaw);
    final sinY = math.sin(yaw);

    final tip   = Offset(center.dx + cosY * 8, center.dy - sinY * 8);
    final left  = Offset(center.dx - sinY *  3.5, center.dy - cosY *  3.5);
    final right = Offset(center.dx + sinY *  3.5, center.dy + cosY *  3.5);
    final base  = Offset(center.dx - cosY *  3, center.dy + sinY *  3);

    final path = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(left.dx, left.dy)
      ..lineTo(base.dx, base.dy)
      ..lineTo(right.dx, right.dy)
      ..close();

    canvas.drawPath(path, Paint()..color = Colors.white);
    canvas.drawPath(path,
        Paint()..color = Colors.black45..style = PaintingStyle.stroke..strokeWidth = 0.8);
  }

  // ── Footprint ──────────────────────────────────────────────────────────

  void _drawFootprint(Canvas canvas, double cx, double cy,
      double scaleX, double scaleY, Pose? pose) {
    if (footprint.isEmpty || pose == null) return;

    final pts = <Offset>[];
    for (int i = 0; i < footprint.length; i++) {
      pts.add(Offset(
        cx + (footprint[i].x - pose.x) * scaleX,
        cy - (footprint[i].y - pose.y) * scaleY,
      ));
    }

    final path = Path()..moveTo(pts[0].dx, pts[0].dy);
    for (int i = 1; i < pts.length; i++) {
      path.lineTo(pts[i].dx, pts[i].dy);
    }
    path.close();

    // Semi-transparent fill
    canvas.drawPath(path, Paint()..color = const Color(0xFF64B5F6).withOpacity(0.25));
    // Bright solid border
    canvas.drawPath(path,
        Paint()
          ..color = const Color(0xFF29B6F6)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.0
          ..strokeJoin = StrokeJoin.miter);
    // Corner markers
    for (final p in pts) {
      canvas.drawCircle(p, 4, Paint()..color = const Color(0xFF29B6F6));
    }
  }

  @override
  bool shouldRepaint(LocalPlanningPainter old) =>
      trajectory != old.trajectory ||
      globalPath != old.globalPath ||
      footprint != old.footprint ||
      gridInfo != old.gridInfo ||
      odomPose != old.odomPose ||
      showTrajectory != old.showTrajectory ||
      showGlobalPath != old.showGlobalPath ||
      navTargetPose != old.navTargetPose ||
      showGrid != old.showGrid ||
      showFootprint != old.showFootprint;
}
