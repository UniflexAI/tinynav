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
        scaffoldBackgroundColor: const Color(0xFFF4F6F8),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(shape: const StadiumBorder()),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(shape: const StadiumBorder()),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: const BorderSide(color: Color(0xFFE0E0E0)),
          ),
        ),
      ),
      // Switches automatically when deviceIpProvider changes.
      home: ip == null ? const SetupPage() : const HomePage(),
    );
  }
}
