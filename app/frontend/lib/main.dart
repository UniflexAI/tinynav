import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'core/providers.dart';
import 'pages/home_page.dart';
import 'pages/setup_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final prefs = await SharedPreferences.getInstance();
  final savedIp = prefs.getString('device_ip');

  runApp(
    ProviderScope(
      overrides: [
        sharedPreferencesProvider.overrideWithValue(prefs),
        if (savedIp != null) deviceIpProvider.overrideWith((ref) => savedIp),
      ],
      child: const TinyNavApp(),
    ),
  );
}

class TinyNavApp extends ConsumerWidget {
  const TinyNavApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final ip = ref.watch(deviceIpProvider);
    return MaterialApp(
      title: 'TinyNav',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1565C0)),
        useMaterial3: true,
      ),
      // Switches automatically when deviceIpProvider changes.
      home: ip == null ? const SetupPage() : const HomePage(),
    );
  }
}
