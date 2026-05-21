import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/models.dart';
import '../core/providers.dart';

const _kBenchmarkAccent = Color(0xFF7B61FF);

class BenchmarkTab extends ConsumerWidget {
  const BenchmarkTab({super.key});

  Future<void> _post(BuildContext context, WidgetRef ref, String path) async {
    try {
      await ref.read(dioProvider).post(path);
    } on DioException catch (e) {
      if (!context.mounted) return;
      final msg = e.response?.data?.toString() ?? e.message ?? 'Request failed';
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final status = ref.watch(benchmarkStreamProvider).valueOrNull;
    final planning = ref.watch(planningStreamProvider).valueOrNull;
    final running = status?.running == true || status?.state == 'running';

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _HeaderCard(status: status),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: running ? null : () => _post(context, ref, '/benchmark/start'),
                icon: const Icon(Icons.play_arrow_rounded),
                label: const Text('Start'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: () => _post(context, ref, '/benchmark/restart'),
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Restart'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: running ? () => _post(context, ref, '/benchmark/stop') : null,
                icon: const Icon(Icons.stop_rounded),
                label: const Text('Stop'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        _ResultCard(result: status?.result),
        const SizedBox(height: 12),
        _VisualizationCard(planning: planning),
        const SizedBox(height: 12),
        const _NotesCard(),
      ],
    );
  }
}

class _HeaderCard extends StatelessWidget {
  final BenchmarkStatus? status;
  const _HeaderCard({this.status});

  @override
  Widget build(BuildContext context) {
    final state = status?.state ?? 'idle';
    final percent = status?.percent ?? 0.0;
    final progress = status?.progressM ?? 0.0;
    final total = status?.totalM ?? 0.0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: _kBenchmarkAccent.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: const Icon(Icons.analytics_outlined, color: _kBenchmarkAccent),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('PNC Benchmark', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800)),
                      Text('Figure-eight tracking · $state', style: const TextStyle(color: Colors.black54)),
                    ],
                  ),
                ),
                Text('${percent.toStringAsFixed(1)}%', style: const TextStyle(fontWeight: FontWeight.w800)),
              ],
            ),
            const SizedBox(height: 14),
            LinearProgressIndicator(value: total > 0 ? (percent / 100.0).clamp(0.0, 1.0).toDouble() : null),
            const SizedBox(height: 8),
            Text('${progress.toStringAsFixed(2)} m / ${total.toStringAsFixed(2)} m'),
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final BenchmarkResult? result;
  const _ResultCard({this.result});

  @override
  Widget build(BuildContext context) {
    final r = result;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: r == null
            ? const Text('No result yet. Start a benchmark run to collect score, RMSE, and completion.')
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(r.score.toStringAsFixed(1), style: const TextStyle(fontSize: 38, fontWeight: FontWeight.w900)),
                      const SizedBox(width: 8),
                      const Text('/ 100', style: TextStyle(color: Colors.black45, fontWeight: FontWeight.w700)),
                      const Spacer(),
                      Chip(label: Text(r.state)),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 10,
                    runSpacing: 8,
                    children: [
                      _Metric(label: 'RMSE', value: _meters(r.rmseM)),
                      _Metric(label: 'Mean err', value: _meters(r.meanErrorM)),
                      _Metric(label: 'Max err', value: _meters(r.maxErrorM)),
                      _Metric(label: 'Completion', value: '${r.completionPercent.toStringAsFixed(1)}%'),
                      _Metric(label: 'Samples', value: '${r.samples}'),
                      if (r.durationS != null) _Metric(label: 'Duration', value: '${r.durationS!.toStringAsFixed(1)}s'),
                    ],
                  ),
                ],
              ),
      ),
    );
  }

  static String _meters(double? v) => v == null ? '-' : '${v.toStringAsFixed(3)} m';
}

class _Metric extends StatelessWidget {
  final String label;
  final String value;
  const _Metric({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F6F8),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label, style: const TextStyle(fontSize: 11, color: Colors.black45)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }
}

class _VisualizationCard extends StatelessWidget {
  final PlanningState? planning;
  const _VisualizationCard({this.planning});

  @override
  Widget build(BuildContext context) {
    final p = planning;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(4, 2, 4, 10),
              child: Text('Live path visualization', style: TextStyle(fontWeight: FontWeight.w800)),
            ),
            AspectRatio(
              aspectRatio: 1.35,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: const Color(0xFF0F1621),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(14),
                  child: p == null
                      ? const Center(child: Text('Waiting for planning data…', style: TextStyle(color: Colors.white54)))
                      : CustomPaint(
                          painter: _BenchmarkPainter(
                            globalPath: p.globalPath,
                            trajectory: p.trajectory,
                            currentPose: p.odomPose,
                            targetPose: p.navTargetPose,
                          ),
                        ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NotesCard extends StatelessWidget {
  const _NotesCard();

  @override
  Widget build(BuildContext context) {
    return const Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Text(
          'Benchmark node anchors an 8-shaped global path at the robot initial odom pose, publishes a rolling /control/target_pose for local planning, records odom during the run, then scores path tracking quality.',
          style: TextStyle(color: Colors.black54, height: 1.35),
        ),
      ),
    );
  }
}

class _BenchmarkPainter extends CustomPainter {
  final List<TrajPoint> globalPath;
  final List<TrajPoint> trajectory;
  final Pose? currentPose;
  final TrajPoint? targetPose;

  const _BenchmarkPainter({
    required this.globalPath,
    required this.trajectory,
    this.currentPose,
    this.targetPose,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final points = <Offset>[
      ...globalPath.map((p) => Offset(p.x, p.y)),
      ...trajectory.map((p) => Offset(p.x, p.y)),
      if (currentPose != null) Offset(currentPose!.x, currentPose!.y),
      if (targetPose != null) Offset(targetPose!.x, targetPose!.y),
    ];
    if (points.isEmpty) return;

    var minX = points.first.dx, maxX = points.first.dx;
    var minY = points.first.dy, maxY = points.first.dy;
    for (final p in points) {
      if (p.dx < minX) minX = p.dx;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy < minY) minY = p.dy;
      if (p.dy > maxY) maxY = p.dy;
    }
    const pad = 24.0;
    final worldW = (maxX - minX).abs().clamp(1.0, double.infinity).toDouble();
    final worldH = (maxY - minY).abs().clamp(1.0, double.infinity).toDouble();
    final scaleX = ((size.width - pad * 2) / worldW).clamp(1.0, double.infinity).toDouble();
    final scaleY = ((size.height - pad * 2) / worldH).clamp(1.0, double.infinity).toDouble();
    final scale = scaleX < scaleY ? scaleX : scaleY;
    final cx = (minX + maxX) / 2.0;
    final cy = (minY + maxY) / 2.0;
    Offset toCanvas(Offset p) => Offset(
          size.width / 2 + (p.dx - cx) * scale,
          size.height / 2 - (p.dy - cy) * scale,
        );

    void drawPath(List<TrajPoint> pts, Color color, double width) {
      if (pts.length < 2) return;
      final path = Path();
      for (var i = 0; i < pts.length; i++) {
        final c = toCanvas(Offset(pts[i].x, pts[i].y));
        if (i == 0) {
          path.moveTo(c.dx, c.dy);
        } else {
          path.lineTo(c.dx, c.dy);
        }
      }
      canvas.drawPath(
        path,
        Paint()
          ..color = color
          ..strokeWidth = width
          ..style = PaintingStyle.stroke
          ..strokeCap = StrokeCap.round
          ..strokeJoin = StrokeJoin.round,
      );
    }

    drawPath(globalPath, const Color(0xFF69F0AE), 3.0);
    drawPath(trajectory, Colors.cyanAccent, 2.0);

    if (targetPose != null) {
      final c = toCanvas(Offset(targetPose!.x, targetPose!.y));
      canvas.drawCircle(c, 7, Paint()..color = const Color(0xFFFF6D00));
      canvas.drawCircle(
        c,
        11,
        Paint()
          ..color = const Color(0xFFFF6D00)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }
    if (currentPose != null) {
      final c = toCanvas(Offset(currentPose!.x, currentPose!.y));
      canvas.drawCircle(c, 6, Paint()..color = Colors.white);
      canvas.drawCircle(
        c,
        8,
        Paint()
          ..color = Colors.black45
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }
  }

  @override
  bool shouldRepaint(_BenchmarkPainter old) {
    if (old.currentPose != currentPose || old.targetPose != targetPose) return true;
    if (old.trajectory.length != trajectory.length) return true;
    if (old.globalPath.length != globalPath.length) return true;
    if (trajectory.isNotEmpty &&
        (old.trajectory.last.x != trajectory.last.x ||
            old.trajectory.last.y != trajectory.last.y)) return true;
    return false;
  }
}
