import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';

class DeviceTab extends ConsumerWidget {
  const DeviceTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statusAsync = ref.watch(deviceStatusProvider);
    final sensorAsync = ref.watch(sensorModeProvider);
    final ip = ref.watch(deviceIpProvider) ?? '—';

    return RefreshIndicator(
      onRefresh: () async {
        ref.invalidate(deviceStatusProvider);
        ref.invalidate(sensorModeProvider);
      },
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Connection ─────────────────────────────────────────────────
          _SectionCard(
            icon: Icons.wifi_rounded,
            title: 'Connection',
            children: statusAsync.when(
              data: (s) => [
                _InfoRow('Status', s.online ? 'Online' : 'Offline',
                    valueColor: s.online ? const Color(0xFF34C759) : Colors.red),
                _InfoRow('IP', ip),
                _InfoRow('State', s.rawState),
              ],
              loading: () => [const _LoadingRow()],
              error: (e, _) => [_InfoRow('Error', '$e', valueColor: Colors.red)],
            ),
          ),
          const SizedBox(height: 12),
          // ── Sensor ─────────────────────────────────────────────────────
          _SectionCard(
            icon: Icons.sensors_rounded,
            title: 'Sensor',
            children: [
              sensorAsync.when(
                data: (mode) => _InfoRow(
                  'Mode',
                  mode == 'realsense'
                      ? 'RealSense'
                      : mode == 'looper'
                          ? 'Looper'
                          : 'Unknown',
                  valueColor: mode == 'unknown' ? Colors.grey : null,
                ),
                loading: () => const _LoadingRow(),
                error: (_, __) => const _InfoRow('Mode', '—'),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // ── System ─────────────────────────────────────────────────────
          _SectionCard(
            icon: Icons.memory_rounded,
            title: 'System',
            children: [
              statusAsync.when(
                data: (s) => s.battery != null
                    ? _InfoRow(
                        'Battery',
                        '${s.battery!.toStringAsFixed(0)}%',
                        valueColor: s.battery! < 20 ? Colors.red : null,
                      )
                    : const _InfoRow('Battery', '—'),
                loading: () => const _LoadingRow(),
                error: (_, __) => const _InfoRow('Battery', '—'),
              ),
              const _InfoRow('CPU', '—', dimmed: true),
              const _InfoRow('GPU', '—', dimmed: true),
              const _InfoRow('Disk', '—', dimmed: true),
            ],
          ),
        ],
      ),
    );
  }
}

// ── Section card ──────────────────────────────────────────────────────────────

class _SectionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final List<Widget> children;

  const _SectionCard({
    required this.icon,
    required this.title,
    required this.children,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.grey.shade200),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(icon, size: 18, color: const Color(0xFF2B3A42)),
              const SizedBox(width: 8),
              Text(title,
                  style: const TextStyle(
                      fontWeight: FontWeight.w700, fontSize: 14)),
            ]),
            const Divider(height: 20),
            ...children,
          ],
        ),
      ),
    );
  }
}

// ── Info row ──────────────────────────────────────────────────────────────────

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;
  final bool dimmed;

  const _InfoRow(this.label, this.value, {this.valueColor, this.dimmed = false});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(
                  fontSize: 13, color: Color(0xFF9E9E9E))),
          Text(
            value,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: dimmed
                  ? Colors.grey.shade400
                  : (valueColor ?? Colors.black87),
            ),
          ),
        ],
      ),
    );
  }
}

class _LoadingRow extends StatelessWidget {
  const _LoadingRow();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Center(child: SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))),
    );
  }
}
