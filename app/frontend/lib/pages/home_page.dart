import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';
import 'device_tab.dart';
import 'map_tab.dart';
import 'operate_tab.dart';

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
      backgroundColor: const Color(0xFF0B1118),
      body: SafeArea(
        child: Column(
          children: [
            // ── Thin status bar ────────────────────────────────────────────
            _StatusBar(ip: ip, isOnline: isOnline, onDisconnect: () => _disconnect(ref)),
            // ── Menu cards ─────────────────────────────────────────────────
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 28),
                children: [
                  // ── Hero ─────────────────────────────────────────────────
                  const _HeroBanner(),
                  const SizedBox(height: 20),
                  _MenuCard(
                    icon: Icons.memory_rounded,
                    iconColor: const Color(0xFF7BD8FF),
                    title: 'Device',
                    subtitle: 'Status · Sensor · System info',
                    badge: status?.rawState == 'realsense_bag_record' ? 'REC' : null,
                    badgeColor: Colors.red,
                    onTap: () => _push(context, 'Device', const DeviceTab()),
                  ),
                  const SizedBox(height: 12),
                  _MenuCard(
                    icon: Icons.folder_outlined,
                    iconColor: const Color(0xFF7BD8FF),
                    title: 'Map',
                    subtitle: 'Bag recording · Map building · Files',
                    badge: status?.rawState == 'rosbag_build_map' ? 'Building' : null,
                    badgeColor: const Color(0xFF4A90D9),
                    onTap: () => _push(context, 'Map', const MapTab()),
                  ),
                  const SizedBox(height: 12),
                  _MenuCard(
                    icon: Icons.sports_esports_outlined,
                    iconColor: const Color(0xFF7BD8FF),
                    title: 'Operate',
                    subtitle: 'Live map · Camera · Teleop · POI',
                    badge: status?.rawState == 'navigation' ? 'Navigating' : null,
                    badgeColor: const Color(0xFF45C95A),
                    onTap: () => _push(context, 'Operate', const OperateTab()),
                  ),
                  const SizedBox(height: 24),
                  if (status != null) _QuickStatusCard(status: status),
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
      backgroundColor: const Color(0xFF0B1118),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0F1822),
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 18),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(title,
            style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 17)),
        bottom: const PreferredSize(
          preferredSize: Size.fromHeight(1),
          child: Divider(height: 1, thickness: 1, color: Color(0xFF233142)),
        ),
      ),
      body: child,
    );
  }
}

// ── Thin status bar ───────────────────────────────────────────────────────────

class _StatusBar extends StatelessWidget {
  final String ip;
  final bool isOnline;
  final VoidCallback onDisconnect;
  const _StatusBar({required this.ip, required this.isOnline, required this.onDisconnect});

  @override
  Widget build(BuildContext context) {
    const kGreen = Color(0xFF45C95A);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: const Color(0xFF111A24),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFF263647)),
          boxShadow: const [
            BoxShadow(
              color: Color(0x11000000),
              blurRadius: 10,
              offset: Offset(0, 3),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isOnline ? kGreen : Colors.red,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                isOnline ? ip : 'Offline',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: isOnline ? kGreen : Colors.red,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const SizedBox(width: 8),
            InkWell(
              borderRadius: BorderRadius.circular(20),
              onTap: onDisconnect,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: const Color(0xFF1A2532),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.logout_rounded, size: 15, color: Color(0xFF9EB0C3)),
                    SizedBox(width: 4),
                    Text(
                      'Disconnect',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFF9EB0C3),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Hero banner ───────────────────────────────────────────────────────────────

class _HeroBanner extends StatelessWidget {
  const _HeroBanner();

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 8, bottom: 14),
      padding: const EdgeInsets.fromLTRB(18, 18, 18, 16),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF132131), Color(0xFF0E1824)],
        ),
        border: Border.all(color: const Color(0xFF284057)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x12000000),
            blurRadius: 14,
            offset: Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: const Color(0xFF16283A),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Image.asset('assets/images/tinynav.png'),
          ),
          const SizedBox(width: 14),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'TinyNav Console',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    color: Color(0xFFE8F2FF),
                    letterSpacing: -0.2,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  'Visual Navigation Module',
                  style: TextStyle(
                    fontSize: 13,
                    color: Color(0xFF9CB0C5),
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ── Menu card ─────────────────────────────────────────────────────────────────

class _MenuCard extends StatelessWidget {
  final IconData icon;
  final Color iconColor;
  final String title;
  final String subtitle;
  final String? badge;
  final Color? badgeColor;
  final VoidCallback onTap;

  const _MenuCard({
    required this.icon,
    required this.iconColor,
    required this.title,
    required this.subtitle,
    this.badge,
    this.badgeColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: const Color(0xFF111A24),
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Ink(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: const Color(0xFF2A3B4D)),
            boxShadow: const [
              BoxShadow(
                color: Color(0x10000000),
                blurRadius: 10,
                offset: Offset(0, 4),
              ),
            ],
          ),
          child: Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: iconColor.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: iconColor, size: 24),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Text(title,
                          style: const TextStyle(
                              fontWeight: FontWeight.w700, fontSize: 15)),
                      if (badge != null) ...[
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 7, vertical: 2),
                          decoration: BoxDecoration(
                            color: (badgeColor ?? Colors.grey).withOpacity(0.15),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(badge!,
                              style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                  color: badgeColor ?? Colors.grey)),
                        ),
                      ],
                    ]),
                    const SizedBox(height: 3),
                    Text(subtitle,
                        style: const TextStyle(
                            fontSize: 12,
                            color: Color(0xFF9AAFC4),
                            fontWeight: FontWeight.w500)),
                  ],
                ),
              ),
              Container(
                width: 28,
                height: 28,
                decoration: BoxDecoration(
                  color: const Color(0xFF1A2532),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: const Icon(Icons.chevron_right_rounded,
                    color: Color(0xFFA9BDCF), size: 20),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Quick status card ─────────────────────────────────────────────────────────

class _QuickStatusCard extends StatelessWidget {
  final dynamic status;
  const _QuickStatusCard({required this.status});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF111A24),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF2A3B4D)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x10000000),
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Quick Status',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13,
                  color: Color(0xFF9FB0C3))),
          const SizedBox(height: 12),
          Row(
            children: [
              _StatItem(
                label: 'State',
                value: status.rawState ?? '—',
                color: const Color(0xFF7BD8FF),
              ),
              const SizedBox(width: 12),
              _StatItem(
                label: 'Bag',
                value: status.bagStatus ?? '—',
                color: const Color(0xFF7BD8FF),
              ),
              const SizedBox(width: 12),
              _StatItem(
                label: 'Map',
                value: status.mapStatus ?? '—',
                color: const Color(0xFF7BD8FF),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _StatItem({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
        decoration: BoxDecoration(
          color: const Color(0xFF1A2532),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label,
                style: TextStyle(
                    fontSize: 10, color: Color(0xFF9FB0C3), fontWeight: FontWeight.w600)),
            const SizedBox(height: 3),
            Text(value,
                style: const TextStyle(
                    fontSize: 12, fontWeight: FontWeight.w600, color: Color(0xFFE8F2FF),
                    overflow: TextOverflow.ellipsis),
                maxLines: 1),
          ],
        ),
      ),
    );
  }
}
