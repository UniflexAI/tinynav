import 'dart:async';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';

class FieldRecordBar extends ConsumerStatefulWidget {
  final bool simulated;
  const FieldRecordBar({super.key, required this.simulated});

  @override
  ConsumerState<FieldRecordBar> createState() => _FieldRecordBarState();
}

class _FieldRecordBarState extends ConsumerState<FieldRecordBar> {
  Timer? _timer;
  Map<String, dynamic> _info = {};
  bool _busy = false, _polling = false, _ready = false;
  String? _error;
  int _epoch = 0;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  @override
  void didUpdateWidget(FieldRecordBar oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.simulated != widget.simulated) _connect();
  }

  void _connect() {
    _timer?.cancel();
    _epoch++;
    _ready = false;
    if (widget.simulated) return;
    _poll();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) => _poll());
  }

  Future<void> _poll() async {
    if (_busy || _polling || widget.simulated) return;
    _polling = true;
    final epoch = _epoch;
    try {
      final r = await ref.read(dioProvider).get('/field-record/info');
      if (mounted && !widget.simulated && epoch == _epoch) {
        setState(() {
          if (!_ready) _error = null;
          _info = Map<String, dynamic>.from(r.data as Map);
          _ready = true;
        });
      }
    } on DioException catch (e) {
      if (mounted && !widget.simulated && epoch == _epoch) {
        setState(() { _ready = false; _error = e.response?.data?['detail']?.toString() ?? 'Recorder unavailable'; });
      }
    } finally { _polling = false; }
  }

  Future<void> _toggle() async {
    final recording = _info['recording'] == true;
    _epoch++;
    setState(() { _busy = true; _error = null; });
    try {
      final r = await ref.read(dioProvider).post(
        recording ? '/field-record/stop' : '/field-record/start',
        options: Options(receiveTimeout: const Duration(seconds: 45)),
      );
      if (mounted) setState(() => _info = Map<String, dynamic>.from(r.data as Map));
      if (mounted && !recording) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Recording started. Start your stair task when ready.')));
      }
    } on DioException catch (e) {
      if (mounted) setState(() => _error = e.response?.data?['detail']?.toString() ?? e.message ?? 'Recording request failed');
    } finally {
      if (mounted) { setState(() => _busy = false); _poll(); }
    }
  }

  Future<void> _details() async {
    final path = _info['path'] as String?;
    await showDialog<void>(context: context, builder: (ctx) => AlertDialog(
      title: const Text('Recording files'),
      content: SingleChildScrollView(child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Files are saved on the robot/backend host.'),
        const SizedBox(height: 12),
        SelectableText(path ?? 'No recording yet'),
        const SizedBox(height: 12),
        const Text('Return the complete folder: metadata.yaml, all bag data files and info.json. Recording continues if you leave Operate.'),
        if (_info['message_count'] != null) Text('Recorded messages: ${_info['message_count']}'),
        if ((_info['missing_topics'] as List? ?? []).isNotEmpty) ...[
          const SizedBox(height: 12),
          const Text('Input topics without messages:', style: TextStyle(color: Colors.orangeAccent)),
          ...(_info['missing_topics'] as List).map((t) => SelectableText('$t')),
        ],
      ])),
      actions: [
        if (path != null) TextButton(onPressed: () async {
          await Clipboard.setData(ClipboardData(text: path));
          if (ctx.mounted) Navigator.pop(ctx);
        }, child: const Text('Copy path')),
        TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Close')),
      ],
    ));
  }

  @override
  void dispose() { _timer?.cancel(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final recording = !widget.simulated && _info['recording'] == true;
    final saving = _info['saving'] == true || (_busy && recording);
    final secs = ((_info['seconds'] as num?) ?? 0).floor();
    final clock = '${(secs ~/ 60).toString().padLeft(2, '0')}:${(secs % 60).toString().padLeft(2, '0')}';
    final error = _error ?? _info['error'] as String?;
    final verified = _info['verified'] == true;
    final missing = (_info['missing_topics'] as List? ?? []).isNotEmpty;
    final label = widget.simulated ? 'Bag recording' : saving ? 'Saving...' : recording ? 'Stop & save' : _busy ? 'Starting...' : 'Record bag';
    final detail = widget.simulated ? 'Simulation traces are saved automatically'
      : error ?? (saving ? 'Closing and checking bag' : recording ? 'REC $clock' : verified ? 'Saved · ${_info['message_count']} messages${missing ? ' · Check inputs' : ''}' : _ready ? 'Record before starting a task' : 'Connecting recorder...');
    return Container(
      color: const Color(0xFF101C28),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      child: Row(children: [
        FilledButton.icon(
          key: const Key('field-record-toggle'),
          onPressed: widget.simulated || !_ready || _busy || saving ? null : _toggle,
          style: FilledButton.styleFrom(foregroundColor: Colors.white, backgroundColor: recording ? Colors.red.shade700 : const Color(0xFF245778)),
          icon: _busy || saving ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
            : Icon(recording ? Icons.stop_rounded : Icons.fiber_manual_record, size: 16),
          label: Text(label),
        ),
        const SizedBox(width: 10),
        Expanded(child: Text(detail, maxLines: 2, overflow: TextOverflow.ellipsis,
          style: TextStyle(fontSize: 11, color: widget.simulated ? Colors.white60 : (error != null || (verified && missing)) ? Colors.orangeAccent : recording ? Colors.redAccent : Colors.white70))),
        if (!widget.simulated) IconButton(tooltip: 'Recording files', onPressed: _details, icon: const Icon(Icons.folder_open, size: 20)),
      ]),
    );
  }
}
