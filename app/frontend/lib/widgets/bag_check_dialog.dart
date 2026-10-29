import 'package:dio/dio.dart';
import 'package:flutter/material.dart';

Future<void> showBagCheckDialog(
  BuildContext context,
  Dio dio,
  String bagName,
) =>
    showDialog<void>(
      context: context,
      builder: (_) => BagCheckDialog(dio: dio, bagName: bagName),
    );

class BagCheckDialog extends StatefulWidget {
  final Dio dio;
  final String bagName;

  const BagCheckDialog({super.key, required this.dio, required this.bagName});

  @override
  State<BagCheckDialog> createState() => _BagCheckDialogState();
}

class _BagCheckDialogState extends State<BagCheckDialog> {
  CancelToken _cancel = CancelToken();
  Map<String, dynamic>? _report;
  String _text = '';
  String? _error;
  bool _checking = true;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    _cancel = CancelToken();
    setState(() {
      _checking = true;
      _report = null;
      _error = null;
    });
    try {
      final response = await widget.dio.post<Map<String, dynamic>>(
        '/files/bags/${Uri.encodeComponent(widget.bagName)}/check',
        cancelToken: _cancel,
        options: Options(receiveTimeout: const Duration(minutes: 16)),
      );
      if (!mounted) return;
      final data = response.data!;
      setState(() {
        _report = Map<String, dynamic>.from(data['report'] as Map);
        _text = data['text'] as String? ?? '';
        _checking = false;
      });
    } catch (error) {
      if (!mounted || (error is DioException && CancelToken.isCancel(error))) {
        return;
      }
      String message = 'Could not complete the check. Please try again.';
      if (error is DioException && error.response?.data is Map) {
        message = '${error.response!.data['detail'] ?? message}';
      }
      setState(() {
        _error = message;
        _checking = false;
      });
    }
  }

  @override
  void dispose() {
    _cancel.cancel('Bag check window closed');
    super.dispose();
  }

  static const _modeNames = {
    'looper': 'Looper',
    'realsense': 'RealSense',
    'unknown': 'Unknown sensor',
  };

  static String _label(String topic) {
    const names = {
      'infra1/image_rect_raw': 'Infrared left',
      'infra2/image_rect_raw': 'Infrared right',
      'depth/image_rect_raw': 'Depth',
      'vio_image': 'Pose',
      'imu': 'IMU',
      'color/image_rect_raw/compressed': 'Colour',
      'color/image_raw': 'Colour',
    };
    for (final entry in names.entries) {
      if (topic.endsWith('/${entry.key}')) return entry.value;
    }
    return topic;
  }

  Widget _streams(Map<String, dynamic> report) {
    final topics = report['topics'] as Map? ?? {};
    final missing = (report['missing'] as List? ?? []).toSet();
    final pairing = report['pairing'] as Map?;
    final color = report['color'] as Map?;
    final lines = <Widget>[];
    double worstGap = 0;
    for (final name in report['streams'] as List? ?? []) {
      final topic = topics[name] as Map? ?? {};
      final hz = (topic['hz'] as num?)?.toStringAsFixed(1) ?? '0';
      lines.add(Text(
        missing.contains(name)
            ? '${_label('$name')}: missing'
            : '${_label('$name')}: ${topic['count'] ?? 0} messages, $hz Hz',
        style: missing.contains(name)
            ? TextStyle(color: Theme.of(context).colorScheme.error)
            : null,
      ));
      for (final gap in topic['gaps'] as List? ?? []) {
        if (gap['boundary'] == 'whole_bag') continue;
        final duration = (gap['duration_s'] as num).toDouble();
        if (duration > worstGap) worstGap = duration;
      }
    }
    String pct(Object? v) => (v as num?)?.toStringAsFixed(1) ?? '0';
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 6),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ...lines,
            const SizedBox(height: 6),
            if (pairing != null)
              Text('Synchronized frames: ${pairing['paired']}/'
                  '${pairing['base_count']} (${pct(pairing['pct'])}%)'),
            if (color != null)
              Text('Colour within ${color['slop_ms']} ms of infrared: '
                  '${color['matched']}/${color['base_count']} (${pct(color['pct'])}%)'),
            Text(worstGap > 0
                ? 'Longest missing interval: ${worstGap.toStringAsFixed(2)} s'
                : 'No receive gap over 250 ms'),
          ],
        ),
      ),
    );
  }

  List<Widget> _results(Map<String, dynamic> report) {
    final warnings = report['warnings'] as List? ?? [];
    final missing = report['missing'] as List? ?? [];
    final tf = report['tf_count'] as Map? ?? {};
    final seconds = (report['duration_s'] as num).toStringAsFixed(1);
    final size = ((report['bytes'] as num) / 1e9).toStringAsFixed(2);
    final mode = '${report['mode']}';
    return [
      Row(
        children: [
          Icon(
              warnings.isEmpty
                  ? Icons.check_circle_outline
                  : Icons.warning_amber,
              color: warnings.isEmpty ? Colors.green : Colors.orange),
          const SizedBox(width: 8),
          Expanded(
              child: Text(warnings.isEmpty
                  ? 'No bag anomalies found'
                  : 'Needs attention')),
        ],
      ),
      const SizedBox(height: 8),
      Text('Sensor: ${_modeNames[mode] ?? mode}',
          style: const TextStyle(fontWeight: FontWeight.bold)),
      Text('${report['mode_reason'] ?? ''}',
          style: const TextStyle(fontSize: 12, color: Colors.grey)),
      const SizedBox(height: 8),
      Text('$seconds seconds · $size GB · Database integrity passed'),
      if (missing.isNotEmpty)
        Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Text(
              'Map building needs: ${missing.map((t) => _label('$t')).join(', ')}',
              style: TextStyle(color: Theme.of(context).colorScheme.error)),
        ),
      if (report['streams'] is List && (report['streams'] as List).isNotEmpty)
        _streams(report),
      const SizedBox(height: 8),
      Text('Static transforms: ${tf['/tf_static'] ?? 0} messages'
          '${(tf['/tf'] ?? 0) > 0 ? ' · /tf: ${tf['/tf']} messages' : ''}'),
      const SizedBox(height: 8),
      const Text(
          'Pairing covers received frames only. Entirely missing intervals are checked separately.'),
      if (warnings.isNotEmpty)
        ExpansionTile(
          title: Text('Warnings (${warnings.length})'),
          children: [
            for (final warning in warnings)
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                child: Align(
                    alignment: Alignment.centerLeft, child: Text('$warning')),
              ),
          ],
        ),
      ExpansionTile(
        title: const Text('Detailed report'),
        children: [
          Padding(
              padding: const EdgeInsets.all(8), child: SelectableText(_text)),
        ],
      ),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Check bag'),
      content: SizedBox(
        width: 600,
        child: ConstrainedBox(
          constraints: BoxConstraints(
              maxHeight: MediaQuery.sizeOf(context).height * .65),
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SelectableText(widget.bagName),
                const SizedBox(height: 16),
                if (_checking) ...[
                  const Center(child: CircularProgressIndicator()),
                  const SizedBox(height: 16),
                  const Text('Checking bag data…'),
                  const Text(
                      'Large bags may take several minutes. Cancel stops this check.'),
                ] else if (_error != null)
                  Text(_error!,
                      style:
                          TextStyle(color: Theme.of(context).colorScheme.error))
                else
                  ..._results(_report!),
              ],
            ),
          ),
        ),
      ),
      actions: [
        if (_error != null)
          TextButton(onPressed: _run, child: const Text('Retry')),
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(_checking ? 'Cancel' : 'Close'),
        ),
      ],
    );
  }
}
