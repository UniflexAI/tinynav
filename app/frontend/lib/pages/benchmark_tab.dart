import 'dart:math' as math;

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/models.dart';
import '../core/providers.dart';
import 'planning_painter.dart';

const _kBenchmarkAccent = Color(0xFF7B61FF);

class BenchmarkTab extends ConsumerStatefulWidget {
  const BenchmarkTab({super.key});

  @override
  ConsumerState<BenchmarkTab> createState() => _BenchmarkTabState();
}

class _BenchmarkTabState extends ConsumerState<BenchmarkTab> {
  String _mode = 'figure8';
  double _scale = 1.0;
  double _sineAmplitude = 0.3;
  double _sineFrequency = 1.0;
  double _sineDuration = 20.0;
  List<SisoTracePoint> _lastSisoTrace = const [];

  Future<void> _post(BuildContext context, WidgetRef ref, String path,
      {Map<String, dynamic>? data}) async {
    try {
      if ((path == '/benchmark/start' || path == '/benchmark/restart') &&
          (data?['mode'] == 'siso_vx_sine')) {
        setState(() => _lastSisoTrace = const []);
      }
      await ref.read(dioProvider).post(path, data: data);
    } on DioException catch (e) {
      if (!context.mounted) return;
      final msg = e.response?.data?.toString() ?? e.message ?? 'Request failed';
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final status = ref.watch(benchmarkStreamProvider).valueOrNull;
    final planning = ref.watch(planningStreamProvider).valueOrNull;
    // Cache the latest non-empty siso trace so it survives stop/completion
    if (status != null &&
        status.mode == 'siso_vx_sine' &&
        status.sisoTrace.isNotEmpty) {
      _lastSisoTrace = status.sisoTrace;
    }
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
                onPressed: running
                    ? null
                    : () => _post(context, ref, '/benchmark/start',
                        data: _benchmarkPayload),
                icon: const Icon(Icons.play_arrow_rounded),
                label: const Text('Start'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: () => _post(context, ref, '/benchmark/restart',
                    data: _benchmarkPayload),
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Restart'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: running
                    ? () => _post(context, ref, '/benchmark/stop')
                    : null,
                icon: const Icon(Icons.stop_rounded),
                label: const Text('Stop'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        _BenchmarkConfigCard(
          mode: _mode,
          scale: _scale,
          sineAmplitude: _sineAmplitude,
          sineFrequency: _sineFrequency,
          sineDuration: _sineDuration,
          enabled: !running,
          onModeChanged: (v) => setState(() => _mode = v),
          onScaleChanged: (v) => setState(() => _scale = v),
          onSineAmplitudeChanged: (v) => setState(() => _sineAmplitude = v),
          onSineFrequencyChanged: (v) => setState(() => _sineFrequency = v),
          onSineDurationChanged: (v) => setState(() => _sineDuration = v),
        ),
        if (_mode == 'figure8') ...[
          const SizedBox(height: 12),
          _ResultCard(result: status?.result),
        ],
        const SizedBox(height: 12),
        _VisualizationCard(
          planning: planning,
          status: status,
          selectedMode: _mode,
          cachedSisoTrace: _lastSisoTrace,
        ),
        const SizedBox(height: 12),
        const _NotesCard(),
      ],
    );
  }

  Map<String, dynamic> get _benchmarkPayload => {
        'mode': _mode,
        'scale': _scale,
        'length_m': 4.0 * _scale,
        'width_m': 2.0 * _scale,
        'sine_amplitude_mps': _sineAmplitude,
        'sine_frequency_hz': _sineFrequency,
        'sine_duration_s': _sineDuration,
      };
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
                  child: const Icon(Icons.analytics_outlined,
                      color: _kBenchmarkAccent),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('PNC Benchmark',
                          style: TextStyle(
                              fontSize: 17, fontWeight: FontWeight.w800)),
                      Text('Figure-eight tracking · $state',
                          style: const TextStyle(color: Colors.black54)),
                    ],
                  ),
                ),
                Text('${percent.toStringAsFixed(1)}%',
                    style: const TextStyle(fontWeight: FontWeight.w800)),
              ],
            ),
            const SizedBox(height: 14),
            LinearProgressIndicator(
                value: total > 0
                    ? (percent / 100.0).clamp(0.0, 1.0).toDouble()
                    : null),
            const SizedBox(height: 8),
            Text(
                '${progress.toStringAsFixed(2)} m / ${total.toStringAsFixed(2)} m'),
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
            ? const Text(
                'No result yet. Start a benchmark run to collect score, RMSE/correlation, and completion.')
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(r.score.toStringAsFixed(1),
                          style: const TextStyle(
                              fontSize: 38, fontWeight: FontWeight.w900)),
                      const SizedBox(width: 8),
                      const Text('/ 100',
                          style: TextStyle(
                              color: Colors.black45,
                              fontWeight: FontWeight.w700)),
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
                      _Metric(
                          label: 'Completion',
                          value: '${r.completionPercent.toStringAsFixed(1)}%'),
                      _Metric(label: 'Samples', value: '${r.samples}'),
                      if (r.durationS != null)
                        _Metric(
                            label: 'Duration',
                            value: '${r.durationS!.toStringAsFixed(1)}s'),
                    ],
                  ),
                ],
              ),
      ),
    );
  }

  static String _meters(double? v) =>
      v == null ? '-' : '${v.toStringAsFixed(3)} m';
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
          Text(label,
              style: const TextStyle(fontSize: 11, color: Colors.black45)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }
}

class _BenchmarkConfigCard extends StatelessWidget {
  final String mode;
  final double scale;
  final double sineAmplitude;
  final double sineFrequency;
  final double sineDuration;
  final bool enabled;
  final ValueChanged<String> onModeChanged;
  final ValueChanged<double> onScaleChanged;
  final ValueChanged<double> onSineAmplitudeChanged;
  final ValueChanged<double> onSineFrequencyChanged;
  final ValueChanged<double> onSineDurationChanged;

  const _BenchmarkConfigCard({
    required this.mode,
    required this.scale,
    required this.sineAmplitude,
    required this.sineFrequency,
    required this.sineDuration,
    required this.enabled,
    required this.onModeChanged,
    required this.onScaleChanged,
    required this.onSineAmplitudeChanged,
    required this.onSineFrequencyChanged,
    required this.onSineDurationChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Benchmark resource',
                style: TextStyle(fontWeight: FontWeight.w800)),
            const SizedBox(height: 10),
            SegmentedButton<String>(
              segments: const [
                ButtonSegment(
                    value: 'figure8',
                    label: Text('PNC 8-shape'),
                    icon: Icon(Icons.route_rounded)),
                ButtonSegment(
                    value: 'siso_vx_sine',
                    label: Text('SISO vx sine'),
                    icon: Icon(Icons.ssid_chart_rounded)),
              ],
              selected: {mode},
              onSelectionChanged:
                  enabled ? (v) => onModeChanged(v.first) : null,
            ),
            const SizedBox(height: 12),
            if (mode == 'figure8') ...[
              Row(
                children: [
                  const Expanded(child: Text('Figure-eight size')),
                  Text(
                      '${scale.toStringAsFixed(2)}× · ${(4.0 * scale).toStringAsFixed(1)}m × ${(2.0 * scale).toStringAsFixed(1)}m'),
                ],
              ),
              Slider(
                value: scale,
                min: 0.25,
                max: 1.5,
                divisions: 25,
                label: '${scale.toStringAsFixed(2)}×',
                onChanged: enabled ? onScaleChanged : null,
              ),
              const Text(
                'Use a smaller scale when the site cannot fit the default 4m × 2m figure-eight.',
                style: TextStyle(color: Colors.black54, fontSize: 12),
              ),
            ] else ...[
              _SliderRow(
                label: 'Amplitude',
                valueText: '${sineAmplitude.toStringAsFixed(2)} m/s',
                value: sineAmplitude,
                min: 0.0,
                max: 2.0,
                divisions: 40,
                onChanged: enabled ? onSineAmplitudeChanged : null,
              ),
              _SliderRow(
                label: 'Frequency',
                valueText: '${sineFrequency.toStringAsFixed(2)} Hz',
                value: sineFrequency,
                min: 0.1,
                max: 20.0,
                divisions: 199,
                onChanged: enabled ? onSineFrequencyChanged : null,
              ),
              _SliderRow(
                label: 'Duration',
                valueText: '${sineDuration.toStringAsFixed(0)} s',
                value: sineDuration,
                min: 5,
                max: 60,
                divisions: 11,
                onChanged: enabled ? onSineDurationChanged : null,
              ),
              const Text(
                'Directly publishes /cmd_vel linear.x = A·sin(2πft), records odom, and scores velocity/position tracking.',
                style: TextStyle(color: Colors.black54, fontSize: 12),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _SliderRow extends StatelessWidget {
  final String label;
  final String valueText;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final ValueChanged<double>? onChanged;

  const _SliderRow({
    required this.label,
    required this.valueText,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(children: [Expanded(child: Text(label)), Text(valueText)]),
        Slider(
            value: value,
            min: min,
            max: max,
            divisions: divisions,
            label: valueText,
            onChanged: onChanged),
      ],
    );
  }
}

class _VisualizationCard extends StatelessWidget {
  final PlanningState? planning;
  final BenchmarkStatus? status;
  final String selectedMode;
  final List<SisoTracePoint> cachedSisoTrace;
  const _VisualizationCard({
    this.planning,
    this.status,
    required this.selectedMode,
    this.cachedSisoTrace = const [],
  });

  @override
  Widget build(BuildContext context) {
    // UI selection is authoritative: after a SISO run stops, keep showing the
    // PCA trace instead of falling back to the PNC local-planning view.
    final mode = selectedMode;
    if (mode == 'siso_vx_sine') {
      final liveTrace = status?.sisoTrace ?? const <SisoTracePoint>[];
      return _SisoTraceCard(
        trace: liveTrace.isNotEmpty ? liveTrace : cachedSisoTrace,
      );
    }
    final p = planning;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(4, 2, 4, 10),
              child: Text('Live local planning view',
                  style: TextStyle(fontWeight: FontWeight.w800)),
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
                      ? const Center(
                          child: Text('Waiting for planning data…',
                              style: TextStyle(color: Colors.white54)))
                      : Stack(
                          fit: StackFit.expand,
                          children: [
                            const ColoredBox(color: Color(0xFF0F1621)),
                            if (p.esdfImage != null)
                              Opacity(
                                  opacity: 0.85,
                                  child: Image.memory(p.esdfImage!,
                                      fit: BoxFit.fill, gaplessPlayback: true)),
                            if (p.obstacleImage != null)
                              Opacity(
                                  opacity: 0.45,
                                  child: Image.memory(p.obstacleImage!,
                                      fit: BoxFit.fill, gaplessPlayback: true)),
                            CustomPaint(
                              painter: LocalPlanningPainter(
                                trajectory: p.trajectory,
                                centerline: p.centerline,
                                globalPath: p.globalPath,
                                footprint: p.footprint,
                                gridInfo: p.gridInfo,
                                odomPose: p.odomPose,
                                navTargetPose: p.navTargetPose,
                              ),
                            ),
                          ],
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

class _SisoTraceCard extends StatelessWidget {
  final List<SisoTracePoint> trace;
  const _SisoTraceCard({required this.trace});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(4, 2, 4, 10),
              child: Row(
                children: const [
                  Text('SISO trace',
                      style: TextStyle(fontWeight: FontWeight.w800)),
                  SizedBox(width: 12),
                  _Legend(color: Colors.cyanAccent, label: 'x (actual)'),
                  SizedBox(width: 8),
                  _Legend(color: Color(0xFFFFC107), label: 'x (cmd integral)'),
                  SizedBox(width: 8),
                  _Legend(color: Color(0xFF4CAF50), label: 'y'),
                  SizedBox(width: 8),
                  _Legend(color: Color(0xFFFF5722), label: 'yaw'),
                  SizedBox(width: 8),
                  _Legend(color: Color(0xFF9C27B0), label: 'z'),
                ],
              ),
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
                  child: trace.length < 2
                      ? const Center(
                          child: Text('Waiting for SISO pose samples…',
                              style: TextStyle(color: Colors.white54)))
                      : CustomPaint(painter: _SisoTracePainter(trace)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  final Color color;
  final String label;
  const _Legend({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(width: 10, height: 3, color: color),
        const SizedBox(width: 4),
        Text(label,
            style: const TextStyle(fontSize: 11, color: Colors.black54)),
      ],
    );
  }
}

class _SisoTracePainter extends CustomPainter {
  final List<SisoTracePoint> trace;
  const _SisoTracePainter(this.trace);

  @override
  void paint(Canvas canvas, Size size) {
    const padL = 42.0, padR = 14.0, padT = 16.0, padB = 28.0;
    final plot =
        Rect.fromLTRB(padL, padT, size.width - padR, size.height - padB);
    final gridPaint = Paint()
      ..color = Colors.white10
      ..strokeWidth = 1;
    for (var i = 0; i <= 4; i++) {
      final y = plot.top + plot.height * i / 4;
      canvas.drawLine(Offset(plot.left, y), Offset(plot.right, y), gridPaint);
    }
    for (var i = 0; i <= 5; i++) {
      final x = plot.left + plot.width * i / 5;
      canvas.drawLine(Offset(x, plot.top), Offset(x, plot.bottom), gridPaint);
    }

    final t0 = trace.first.t;
    final t1 = trace.last.t <= t0 ? t0 + 1.0 : trace.last.t;
    final xActual = trace.map((p) => p.xRel).toList();
    final yActual = trace.map((p) => p.yRel).toList();
    final yawActual = trace.map((p) => p.yawRel).toList();
    final zActual = trace.map((p) => p.zRel).toList();
    final expected = _integratedCommand(trace);
    final all = [...xActual, ...expected, ...yActual, ...yawActual, ...zActual];
    var minY = all.reduce(math.min);
    var maxY = all.reduce(math.max);
    if ((maxY - minY).abs() < 1e-6) {
      minY -= 0.5;
      maxY += 0.5;
    }
    final margin = (maxY - minY) * 0.12;
    minY -= margin;
    maxY += margin;

    Offset map(int i, double y) {
      final tx = (trace[i].t - t0) / (t1 - t0);
      final ty = (y - minY) / (maxY - minY);
      return Offset(
          plot.left + tx * plot.width, plot.bottom - ty * plot.height);
    }

    void drawSeries(List<double> values, Color color, double width) {
      if (values.length < 2) return;
      final path = Path()..moveTo(map(0, values[0]).dx, map(0, values[0]).dy);
      for (var i = 1; i < values.length; i++) {
        final p = map(i, values[i]);
        path.lineTo(p.dx, p.dy);
      }
      canvas.drawPath(
          path,
          Paint()
            ..color = color
            ..strokeWidth = width
            ..style = PaintingStyle.stroke
            ..strokeCap = StrokeCap.round);
    }

    drawSeries(zActual, const Color(0xFF9C27B0), 1.5);
    drawSeries(yActual, const Color(0xFF4CAF50), 1.5);
    drawSeries(yawActual, const Color(0xFFFF5722), 1.5);
    drawSeries(expected, const Color(0xFFFFC107), 2.0);
    drawSeries(xActual, Colors.cyanAccent, 2.5);

    final textPainter = TextPainter(textDirection: TextDirection.ltr);
    void label(String text, Offset at) {
      textPainter.text = TextSpan(
          text: text,
          style: const TextStyle(color: Colors.white54, fontSize: 10));
      textPainter.layout();
      textPainter.paint(canvas, at);
    }

    label('${maxY.toStringAsFixed(2)} m', Offset(4, plot.top - 4));
    label('${minY.toStringAsFixed(2)} m', Offset(4, plot.bottom - 10));
    label('${(t1 - t0).toStringAsFixed(1)} s',
        Offset(plot.right - 34, plot.bottom + 6));
  }

  List<double> _integratedCommand(List<SisoTracePoint> pts) {
    final out = <double>[0.0];
    for (var i = 1; i < pts.length; i++) {
      final dt = math.max(0.0, pts[i].t - pts[i - 1].t);
      out.add(out.last + pts[i - 1].cmdVx * dt);
    }
    return out;
  }

  @override
  bool shouldRepaint(_SisoTracePainter old) =>
      old.trace.length != trace.length ||
      (trace.isNotEmpty &&
          old.trace.isNotEmpty &&
          old.trace.last.t != trace.last.t);
}

class _NotesCard extends StatelessWidget {
  const _NotesCard();

  @override
  Widget build(BuildContext context) {
    return const Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Text(
          'PNC 8-shape measures the full planning-control loop. SISO vx sine bypasses planning, directly sends a sine-wave /cmd_vel linear.x, records odom, and checks whether the velocity/position response matches the commanded waveform.',
          style: TextStyle(color: Colors.black54, height: 1.35),
        ),
      ),
    );
  }
}
