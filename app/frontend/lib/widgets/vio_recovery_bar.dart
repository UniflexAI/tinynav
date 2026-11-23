import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/providers.dart';

class VioRecoveryBar extends ConsumerStatefulWidget {
  final Map<String, dynamic> safety;
  const VioRecoveryBar({super.key, required this.safety});
  @override
  ConsumerState<VioRecoveryBar> createState() => _VioRecoveryBarState();
}

class _VioRecoveryBarState extends ConsumerState<VioRecoveryBar> {
  bool _busy = false;

  Future<void> _confirm() async {
    final faultId = widget.safety['fault_id'];
    final agreed = await showDialog<bool>(context: context, builder: (ctx) => AlertDialog(
      title: const Text('Confirm VIO recovery'),
      content: const Text('VIO and depth have recovered. Check that the robot is stable and its surroundings are safe. Confirm to rebuild local perception and let the current task continue. A separately paused task stays paused.'),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Keep stopped')),
        FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Confirm recovery')),
      ],
    ));
    if (agreed != true || !mounted) return;
    setState(() => _busy = true);
    try {
      await ref.read(dioProvider).post('/nav/vio/confirm', data: {'fault_id': faultId});
    } on DioException catch (e) {
      final data = e.response?.data;
      final detail = data is Map ? data['detail']?.toString() : null;
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(detail ?? 'Recovery confirmation failed'),
        backgroundColor: Colors.red));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final safety = widget.safety;
    if (safety['blocked'] != true) return const SizedBox.shrink();
    final locked = safety['locked'] == true;
    final ready = safety['ready'] == true;
    const reasons = {
      'meter_jump': 'Position estimate jumped', 'pose_timeout': 'Odometry stream lost',
      'depth_timeout': 'Depth stream lost', 'vio_restarting': 'VIO restarted',
      'vio_not_inited': 'VIO is initializing', 'supervisor_restart': 'Safety supervisor restarted',
      'supervisor_unavailable': 'Safety supervisor unavailable',
    };
    final reason = reasons[safety['reason']] ?? 'Position estimate unavailable';
    return Container(
      width: double.infinity, color: const Color(0xFF6A2525),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      child: Wrap(spacing: 12, runSpacing: 6, crossAxisAlignment: WrapCrossAlignment.center, children: [
        Text(locked ? 'VIO safety stop · $reason' : 'Waiting for fresh VIO and depth',
          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        if (locked) Text(ready ? 'Recovered · manual confirmation required' : 'Waiting for stable inputs',
          style: const TextStyle(color: Colors.white70, fontSize: 12)),
        if (locked) FilledButton.icon(
          key: const Key('vio-confirm'),
          onPressed: ready && !_busy ? _confirm : null,
          icon: _busy ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
            : const Icon(Icons.verified_user_outlined, size: 16),
          label: const Text('Confirm VIO recovery')),
      ]),
    );
  }
}
