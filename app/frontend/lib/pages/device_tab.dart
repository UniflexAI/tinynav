import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/models.dart';
import '../core/providers.dart';

class DeviceTab extends ConsumerWidget {
  const DeviceTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statusAsync = ref.watch(deviceStatusProvider);

    return RefreshIndicator(
      onRefresh: () async => ref.invalidate(deviceStatusProvider),
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          statusAsync.when(
            data: (s) => _StatusCard(status: s),
            loading: () => const _LoadingCard(),
            error: (e, _) => _ErrorCard(message: '$e'),
          ),
          const SizedBox(height: 12),
          statusAsync.when(
            data: (s) => _BagCard(status: s),
            loading: () => const SizedBox.shrink(),
            error: (_, __) => const SizedBox.shrink(),
          ),
          const SizedBox(height: 12),
          statusAsync.when(
            data: (s) => _MapBuildCard(status: s),
            loading: () => const SizedBox.shrink(),
            error: (_, __) => const SizedBox.shrink(),
          ),
        ],
      ),
    );
  }
}

// ── Device status ──────────────────────────────────────────────────────────

class _StatusCard extends StatelessWidget {
  final DeviceStatus status;
  const _StatusCard({required this.status});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          _CardHeader(Icons.device_hub_outlined, 'Device',
              trailing: _StatusChip(status.online ? 'Online' : 'Offline', status.online)),
          const Divider(height: 24),
          _Row('State', status.rawState),
          if (status.battery != null)
            _Row('Battery', '${(status.battery! * 100).toStringAsFixed(0)}%'),
        ]),
      ),
    );
  }
}

// ── Bag control ─────────────────────────────────────────────────────────────

class _BagCard extends ConsumerStatefulWidget {
  final DeviceStatus status;
  const _BagCard({required this.status});

  @override
  ConsumerState<_BagCard> createState() => _BagCardState();
}

class _BagCardState extends ConsumerState<_BagCard> {
  bool _busy = false;

  Future<void> _call(String path) async {
    setState(() => _busy = true);
    try {
      await ref.read(dioProvider).post(path);
    } on DioException catch (e) {
      if (mounted) _snack(context, e.response?.data?['detail'] ?? e.message ?? 'Error');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.status;
    final isRecording = s.rawState == 'realsense_bag_record';
    final canStart = s.online && s.rawState == 'idle';
    final canStop = s.online && isRecording;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          _CardHeader(
            Icons.videocam_outlined,
            'Bag Recording',
            trailing: isRecording
                ? const Row(mainAxisSize: MainAxisSize.min, children: [
                    Icon(Icons.circle, size: 10, color: Colors.red),
                    SizedBox(width: 4),
                    Text('REC', style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
                  ])
                : null,
          ),
          const Divider(height: 24),
          _Row('Status', s.bagStatus),
          _Row('Bag ready', s.bagFileReady ? 'Yes' : 'No'),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: canStart && !_busy ? () => _call('/bag/start') : null,
                icon: const Icon(Icons.fiber_manual_record, size: 16),
                label: const Text('Start'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: canStop && !_busy ? () => _call('/bag/stop') : null,
                icon: const Icon(Icons.stop),
                label: const Text('Stop'),
              ),
            ),
          ]),
        ]),
      ),
    );
  }
}

// ── Map build ───────────────────────────────────────────────────────────────

class _MapBuildCard extends ConsumerStatefulWidget {
  final DeviceStatus status;
  const _MapBuildCard({required this.status});

  @override
  ConsumerState<_MapBuildCard> createState() => _MapBuildCardState();
}

class _MapBuildCardState extends ConsumerState<_MapBuildCard> {
  bool _busy = false;

  Future<void> _buildMap() async {
    setState(() => _busy = true);
    try {
      await ref.read(dioProvider).post('/map/build');
    } on DioException catch (e) {
      if (mounted) _snack(context, e.response?.data?['detail'] ?? e.message ?? 'Error');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.status;
    final isBuilding = s.rawState == 'rosbag_build_map';
    final canBuild = s.online && s.bagFileReady && s.rawState == 'idle';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const _CardHeader(Icons.map_outlined, 'Map Building'),
          const Divider(height: 24),
          _Row('Status', s.mapStatus),
          if (isBuilding) ...[
            const SizedBox(height: 8),
            LinearProgressIndicator(value: s.mappingPercent > 0 ? s.mappingPercent / 100 : null),
            const SizedBox(height: 4),
            Text('${s.mappingPercent.toStringAsFixed(1)}%',
                style: const TextStyle(fontSize: 12, color: Colors.grey)),
          ],
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: canBuild && !_busy ? _buildMap : null,
              icon: _busy
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.construction),
              label: const Text('Build Map'),
            ),
          ),
        ]),
      ),
    );
  }
}

// ── Shared helpers ──────────────────────────────────────────────────────────

class _CardHeader extends StatelessWidget {
  final IconData icon;
  final String title;
  final Widget? trailing;
  const _CardHeader(this.icon, this.title, {this.trailing});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Icon(icon, size: 20),
      const SizedBox(width: 8),
      Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
      if (trailing != null) ...[const Spacer(), trailing!],
    ]);
  }
}

class _StatusChip extends StatelessWidget {
  final String label;
  final bool positive;
  const _StatusChip(this.label, this.positive);

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(label, style: const TextStyle(fontSize: 12)),
      backgroundColor: positive ? Colors.green.shade100 : Colors.red.shade100,
      side: BorderSide.none,
      padding: EdgeInsets.zero,
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  const _Row(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.grey)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}

class _LoadingCard extends StatelessWidget {
  const _LoadingCard();

  @override
  Widget build(BuildContext context) {
    return const Card(
      child: Padding(
        padding: EdgeInsets.all(32),
        child: Center(child: CircularProgressIndicator()),
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  final String message;
  const _ErrorCard({required this.message});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.red.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(children: [
          const Icon(Icons.error_outline, color: Colors.red),
          const SizedBox(width: 8),
          Expanded(child: Text(message, style: const TextStyle(color: Colors.red))),
        ]),
      ),
    );
  }
}

void _snack(BuildContext context, String message) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(content: Text(message), backgroundColor: Colors.red),
  );
}
