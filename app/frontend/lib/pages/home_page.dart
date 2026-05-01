import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../core/providers.dart';
import 'device_tab.dart';
import 'map_tab.dart';
import 'operate_tab.dart';

// ── HUD style constants ──────────────────────────────────────────────────────

const _bg = Color(0xFF0A0A0A);
const _surface = Color(0xFF111111);
const _border = Color(0xFF1A1A1A);
const _green = Color(0xFF00E676);
const _cyan = Color(0xFF00BCD4);
const _blue = Color(0xFF448AFF);
const _red = Color(0xFFFF5252);
const _text = Color(0xFFE0E0E0);
const _muted = Color(0xFF616161);

// ── Top-level menu ────────────────────────────────────────────────────────────

class HomePage extends ConsumerWidget {
  const HomePage({super.key});

  Future<void> _disconnect(WidgetRef ref) async {
    final prefs = ref.read(sharedPreferencesProvider);
    await prefs.remove('device_ip');
    ref.read(deviceIpProvider.notifier).state = null;
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final ip = ref.watch(deviceIpProvider) ?? '';
    final statusAsync = ref.watch(deviceStatusProvider);
    final status = statusAsync.valueOrNull;
    final isOnline = status?.online ?? false;

    return Scaffold(
      backgroundColor: _bg,
      body: SafeArea(
        child: Column(
          children: [
            // ── Status bar ────────────────────────────────────────────────
            _StatusBar(ip: ip, isOnline: isOnline, onDisconnect: () => _disconnect(ref)),
            // ── Content ───────────────────────────────────────────────────
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
                children: [
                  const _HeroBanner(),
                  const SizedBox(height: 24),
                  // ── Navigation entries ──────────────────────────────────
                  _HudEntry(
                    icon: Icons.memory_rounded,
                    accent: _text,
                    label: 'DEVICE',
                    detail: 'Status · Sensor · System',
                    badge: status?.rawState == 'realsense_bag_record' ? 'REC' : null,
                    badgeColor: _red,
                    onTap: () => _push(context, 'Device', const DeviceTab()),
                  ),
                  const Divider(height: 1, thickness: 1, color: _border),
                  _HudEntry(
                    icon: Icons.folder_outlined,
                    accent: _blue,
                    label: 'MAP',
                    detail: 'Record · Build · Files',
                    badge: status?.rawState == 'rosbag_build_map' ? 'BUILD' : null,
                    badgeColor: _cyan,
                    onTap: () => _push(context, 'Map', const MapTab()),
                  ),
                  const Divider(height: 1, thickness: 1, color: _border),
                  _HudEntry(
                    icon: Icons.sports_esports_outlined,
                    accent: _green,
                    label: 'OPERATE',
                    detail: 'Live Map · Teleop · POI',
                    badge: status?.rawState == 'navigation' ? 'NAV' : null,
                    badgeColor: _green,
                    onTap: () => _push(context, 'Operate', const OperateTab()),
                  ),
                  const Divider(height: 1, thickness: 1, color: _border),
                  if (status != null) ...[
                    const SizedBox(height: 24),
                    _StatusPanel(status: status),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _push(BuildContext context, String title, Widget page) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => _SubPage(title: title, child: page),
      ),
    );
  }
}

// ── Sub-page wrapper ──────────────────────────────────────────────────────────

class _SubPage extends StatelessWidget {
  final String title;
  final Widget child;
  const _SubPage({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      appBar: AppBar(
        backgroundColor: _bg,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 16, color: _muted),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(title),
        bottom: const PreferredSize(
          preferredSize: Size.fromHeight(1),
          child: Divider(height: 1, thickness: 1, color: _border),
        ),
      ),
      body: child,
    );
  }
}

// ── Status bar ────────────────────────────────────────────────────────────────

class _StatusBar extends StatelessWidget {
  final String ip;
  final bool isOnline;
  final VoidCallback onDisconnect;
  const _StatusBar({required this.ip, required this.isOnline, required this.onDisconnect});

  @override
  Widget build(BuildContext context) {
    final color = isOnline ? _green : _red;
    return Container(
      color: _bg,
      padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
      child: Row(
        children: [
          Container(
            width: 6, height: 6,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color,
              boxShadow: [BoxShadow(color: color.withOpacity(0.5), blurRadius: 4)],
            ),
          ),
          const SizedBox(width: 8),
          Text(
            isOnline ? ip : 'OFFLINE',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.8,
              color: color,
            ),
          ),
          const Spacer(),
          IconButton(
            icon: const Icon(Icons.logout_rounded, size: 16, color: _muted),
            tooltip: 'Disconnect',
            onPressed: onDisconnect,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
          ),
          const SizedBox(width: 8),
        ],
      ),
    );
  }
}

// ── Hero banner ───────────────────────────────────────────────────────────────

class _HeroBanner extends StatelessWidget {
  const _HeroBanner();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Column(
        children: [
          Image.asset(
            'assets/images/tinynav_dark.png',
            width: 100,
            height: 100,
          ),
          const SizedBox(height: 12),
          const Text(
            'TINYNAV',
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w300,
              letterSpacing: 4,
              color: _text,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'VISUAL NAVIGATION MODULE',
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 2,
              color: _muted,
            ),
          ),
        ],
      ),
    );
  }
}

// ── HUD menu entry ────────────────────────────────────────────────────────────

class _HudEntry extends StatelessWidget {
  final IconData icon;
  final Color accent;
  final String label;
  final String detail;
  final String? badge;
  final Color? badgeColor;
  final VoidCallback onTap;

  const _HudEntry({
    required this.icon,
    required this.accent,
    required this.label,
    required this.detail,
    this.badge,
    this.badgeColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: _bg,
      child: InkWell(
        onTap: onTap,
        splashColor: accent.withOpacity(0.08),
        highlightColor: accent.withOpacity(0.04),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 16),
          child: Row(
            children: [
              Icon(icon, color: accent, size: 20),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Text(label,
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 13,
                            letterSpacing: 1.5,
                            color: accent,
                          )),
                      if (badge != null) ...[
                        const SizedBox(width: 10),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                          decoration: BoxDecoration(
                            border: Border.all(color: badgeColor ?? _muted, width: 1),
                            borderRadius: BorderRadius.circular(2),
                          ),
                          child: Text(badge!,
                              style: TextStyle(
                                fontSize: 9,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 1,
                                color: badgeColor ?? _muted,
                              )),
                        ),
                      ],
                    ]),
                    const SizedBox(height: 2),
                    Text(detail,
                        style: const TextStyle(
                          fontSize: 11,
                          color: _muted,
                          letterSpacing: 0.3,
                        )),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right_rounded, color: _muted, size: 16),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Status panel ──────────────────────────────────────────────────────────────

class _StatusPanel extends StatelessWidget {
  final dynamic status;
  const _StatusPanel({required this.status});

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
          const Text('STATUS',
              style: TextStyle(
                fontWeight: FontWeight.w700,
                fontSize: 10,
                letterSpacing: 1.5,
                color: _muted,
              )),
          const SizedBox(height: 12),
          Row(
            children: [
              _StatusCell(label: 'STATE', value: status.rawState ?? '—', color: _text),
              const SizedBox(width: 8),
              _StatusCell(label: 'BAG', value: status.bagStatus ?? '—', color: _blue),
              const SizedBox(width: 8),
              _StatusCell(label: 'MAP', value: status.mapStatus ?? '—', color: _green),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatusCell extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _StatusCell({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
        decoration: BoxDecoration(
          border: Border.all(color: color.withOpacity(0.15), width: 1),
          borderRadius: BorderRadius.circular(2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label,
                style: TextStyle(
                  fontSize: 9,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.2,
                  color: color.withOpacity(0.6),
                )),
            const SizedBox(height: 3),
            Text(value,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: color,
                  letterSpacing: 0.3,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis),
          ],
        ),
      ),
    );
  }
}
