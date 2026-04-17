import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';
import 'device_tab.dart';
import 'map_tab.dart';
import 'nav_tab.dart';

class HomePage extends ConsumerStatefulWidget {
  const HomePage({super.key});

  @override
  ConsumerState<HomePage> createState() => _HomePageState();
}

class _HomePageState extends ConsumerState<HomePage> {
  int _currentIndex = 0;

  static const _tabs = [
    DeviceTab(),
    MapTab(),
    NavTab(),
  ];

  Future<void> _disconnect() async {
    final prefs = ref.read(sharedPreferencesProvider);
    await prefs.remove('device_ip');
    ref.read(deviceIpProvider.notifier).state = null;
  }

  @override
  Widget build(BuildContext context) {
    final ip = ref.watch(deviceIpProvider) ?? '';
    final statusAsync = ref.watch(deviceStatusProvider);
    final isOnline = statusAsync.valueOrNull?.online ?? false;

    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 500),
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isOnline ? Colors.green : Colors.red,
            ),
          ),
          const SizedBox(width: 8),
          Text(ip, style: const TextStyle(fontSize: 15)),
        ]),
        actions: [
          IconButton(
            icon: const Icon(Icons.link_off),
            tooltip: 'Disconnect',
            onPressed: _disconnect,
          ),
        ],
      ),
      body: _tabs[_currentIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.device_hub_outlined), label: 'Device'),
          NavigationDestination(icon: Icon(Icons.map_outlined), label: 'Map'),
          NavigationDestination(icon: Icon(Icons.navigation_outlined), label: 'Navigate'),
        ],
      ),
    );
  }
}
