import 'package:flutter/material.dart';

class VioRecoveryBar extends StatelessWidget {
  final Map<String, dynamic> safety;
  const VioRecoveryBar({super.key, required this.safety});

  @override
  Widget build(BuildContext context) {
    if (safety['blocked'] != true) return const SizedBox.shrink();
    const reasons = {
      'meter_jump': 'Position estimate jumped', 'pose_timeout': 'Odometry stream lost',
      'depth_timeout': 'Depth stream lost', 'vio_restarting': 'VIO restarted',
      'vio_not_inited': 'VIO is initializing', 'supervisor_restart': 'Safety supervisor restarted',
      'supervisor_unavailable': 'Safety supervisor unavailable',
    };
    final reason = reasons[safety['reason']] ?? 'Waiting for fresh VIO and depth';
    return Container(
      width: double.infinity, color: const Color(0xFF6A2525),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      child: Wrap(spacing: 12, runSpacing: 6, crossAxisAlignment: WrapCrossAlignment.center, children: [
        Text('VIO stopped · $reason',
          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        const Text('Automatic recovery · waiting for stable inputs',
          style: TextStyle(color: Colors.white70, fontSize: 12)),
      ]),
    );
  }
}
