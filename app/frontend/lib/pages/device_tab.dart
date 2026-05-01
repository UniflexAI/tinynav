import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';

const _bg = Color(0xFF0A0A0A);
const _surface = Color(0xFF111111);
const _border = Color(0xFF1A1A1A);
const _green = Color(0xFF00E676);
const _cyan = Color(0xFF00BCD4);
const _red = Color(0xFFFF5252);
const _text = Color(0xFFE0E0E0);
const _muted = Color(0xFF616161);

class DeviceTab extends ConsumerWidget {
  const DeviceTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statusAsync = ref.watch(deviceStatusProvider);
    final sensorAsync = ref.watch(sensorModeProvider);
    final sysAsync = ref.watch(sysInfoProvider);
    final ip = ref.watch(deviceIpProvider) ?? '—';

    return RefreshIndicator(
      color: _green,
      backgroundColor: _surface,
      onRefresh: () async {
        ref.invalidate(deviceStatusProvider);
        ref.invalidate(sensorModeProvider);
      },
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _SectionCard(
            icon: Icons.wifi_rounded,
            title: 'CONNECTION',
            children: statusAsync.when(
              data: (s) => [
                _InfoRow('Status', s.online ? 'Online' : 'Offline',
                    valueColor: s.online ? _green : _red),
                _InfoRow('IP', ip),
                _InfoRow('State', s.rawState),
              ],
              loading: () => [const _LoadingRow()],
              error: (e, _) => [_InfoRow('Error', '$e', valueColor: _red)],
            ),
          ),
          const SizedBox(height: 12),
          _SectionCard(
            icon: Icons.sensors_rounded,
            title: 'SENSOR',
            children: [
              sensorAsync.when(
                data: (mode) => _InfoRow(
                  'Mode',
                  mode == 'realsense'
                      ? 'RealSense'
                      : mode == 'looper'
                          ? 'Looper'
                          : 'Unknown',
                  valueColor: mode == 'unknown' ? _muted : null,
                ),
                loading: () => const _LoadingRow(),
                error: (_, __) => const _InfoRow('Mode', '—'),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _SectionCard(
            icon: Icons.memory_rounded,
            title: 'SYSTEM',
            children: [
              statusAsync.when(
                data: (s) => s.battery != null
                    ? _InfoRow(
                        'Battery',
                        '${s.battery!.toStringAsFixed(0)}%',
                        valueColor: s.battery! < 20 ? _red : null,
                      )
                    : const _InfoRow('Battery', '—'),
                loading: () => const _LoadingRow(),
                error: (_, __) => const _InfoRow('Battery', '—'),
              ),
              sysAsync.when(
                data: (sys) => Column(
                  children: [
                    _InfoRow('CPU', '${sys.cpuPercent.toStringAsFixed(1)}%',
                        valueColor: sys.cpuPercent > 85 ? _red : null),
                    _InfoRow(
                      'Memory',
                      '${sys.memUsedGb.toStringAsFixed(1)}/${sys.memTotalGb.toStringAsFixed(1)} GB  (${sys.memPercent.toStringAsFixed(0)}%)',
                      valueColor: sys.memPercent > 85 ? _red : null,
                    ),
                    _InfoRow(
                      'Disk',
                      '${sys.diskUsedGb.toStringAsFixed(1)}/${sys.diskTotalGb.toStringAsFixed(1)} GB  (${sys.diskPercent.toStringAsFixed(0)}%)',
                      valueColor: sys.diskPercent > 90 ? _red : null,
                    ),
                    if (sys.gpuPercent != null)
                      _InfoRow('GPU', '${sys.gpuPercent!.toStringAsFixed(1)}%',
                          valueColor: sys.gpuPercent! > 85 ? _red : null),
                  ],
                ),
                loading: () => const _LoadingRow(),
                error: (_, __) => const _InfoRow('System', 'unavailable', dimmed: true),
              ),
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
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _surface,
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: _border, width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(icon, size: 16, color: _cyan),
            const SizedBox(width: 8),
            Text(title,
                style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 11,
                  letterSpacing: 1.5,
                  color: _muted,
                )),
          ]),
          const Divider(height: 20, color: _border),
          ...children,
        ],
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
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(
                fontSize: 12,
                letterSpacing: 0.5,
                color: _muted,
              )),
          Text(
            value,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.3,
              color: dimmed
                  ? _muted
                  : (valueColor ?? _text),
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
      child: Center(child: SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 1.5, color: _green))),
    );
  }
}
