import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../core/providers.dart';
import 'device_tab.dart';
import 'map_tab.dart';
import 'nav_tab.dart';

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
      backgroundColor: const Color(0xFFF2F3F5),
      body: SafeArea(
        child: Column(
          children: [
            // ── Header ─────────────────────────────────────────────────────
            _Header(ip: ip, isOnline: isOnline, onDisconnect: () => _disconnect(ref)),
            // ── Menu cards ─────────────────────────────────────────────────
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
                children: [
                  _MenuCard(
                    icon: Icons.sensors,
                    iconColor: const Color(0xFFFF6B35),
                    title: 'Device',
                    subtitle: 'Status · Bag recording · Map build',
                    badge: status?.rawState == 'realsense_bag_record' ? 'REC' : null,
                    badgeColor: Colors.red,
                    onTap: () => _push(context, 'Device', const DeviceTab()),
                  ),
                  const SizedBox(height: 12),
                  _MenuCard(
                    icon: Icons.map_outlined,
                    iconColor: const Color(0xFF4A90D9),
                    title: 'Map',
                    subtitle: 'Visualize · POI · Local planning',
                    badge: status?.rawState == 'rosbag_build_map' ? 'Building' : null,
                    badgeColor: const Color(0xFF4A90D9),
                    onTap: () => _push(context, 'Map', const MapTab()),
                  ),
                  const SizedBox(height: 12),
                  _MenuCard(
                    icon: Icons.navigation_outlined,
                    iconColor: const Color(0xFF34C759),
                    title: 'Navigate',
                    subtitle: 'Go to POI · Cancel · Nav status',
                    badge: status?.rawState == 'navigation' ? 'Active' : null,
                    badgeColor: const Color(0xFF34C759),
                    onTap: () => _push(context, 'Navigate', const NavTab()),
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
      backgroundColor: const Color(0xFFF2F3F5),
      appBar: AppBar(
        backgroundColor: Colors.white,
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
          child: Divider(height: 1, thickness: 1, color: Color(0xFFEEEEEE)),
        ),
      ),
      body: child,
    );
  }
}

// ── Header ────────────────────────────────────────────────────────────────────

class _Header extends StatelessWidget {
  final String ip;
  final bool isOnline;
  final VoidCallback onDisconnect;
  const _Header({required this.ip, required this.isOnline, required this.onDisconnect});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(20, 16, 12, 16),
      child: Row(
        children: [
          // Logo / brand
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: const Color(0xFFFF6B35),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.navigation_rounded, color: Colors.white, size: 22),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('TinyNav',
                    style: TextStyle(fontWeight: FontWeight.w800, fontSize: 17)),
                Row(children: [
                  Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: isOnline ? const Color(0xFF34C759) : Colors.red,
                    ),
                  ),
                  const SizedBox(width: 5),
                  Text(
                    isOnline ? ip : 'Offline',
                    style: TextStyle(
                      fontSize: 12,
                      color: isOnline ? const Color(0xFF34C759) : Colors.red,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ]),
              ],
            ),
          ),
          IconButton(
            icon: const Icon(Icons.logout_rounded, size: 20, color: Colors.black54),
            tooltip: 'Disconnect',
            onPressed: onDisconnect,
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
      color: Colors.white,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
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
                            color: Color(0xFF9E9E9E),
                            fontWeight: FontWeight.w400)),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right_rounded,
                  color: Color(0xFFBDBDBD), size: 22),
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
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Quick Status',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13,
                  color: Color(0xFF9E9E9E))),
          const SizedBox(height: 12),
          Row(
            children: [
              _StatItem(
                label: 'State',
                value: status.rawState ?? '—',
                color: const Color(0xFFFF6B35),
              ),
              const SizedBox(width: 12),
              _StatItem(
                label: 'Bag',
                value: status.bagStatus ?? '—',
                color: const Color(0xFF4A90D9),
              ),
              const SizedBox(width: 12),
              _StatItem(
                label: 'Map',
                value: status.mapStatus ?? '—',
                color: const Color(0xFF34C759),
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
          color: color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label,
                style: TextStyle(
                    fontSize: 10, color: color, fontWeight: FontWeight.w600)),
            const SizedBox(height: 3),
            Text(value,
                style: const TextStyle(
                    fontSize: 12, fontWeight: FontWeight.w600,
                    overflow: TextOverflow.ellipsis),
                maxLines: 1),
          ],
        ),
      ),
    );
  }
}
