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
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF45C95A),
          primary: const Color(0xFF45C95A),
        ),
        useMaterial3: true,
        fontFamily: 'RobotoLocal',
        scaffoldBackgroundColor: const Color(0xFFF2F3F5),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          foregroundColor: Color(0xFF2B3A42),
          elevation: 0,
          surfaceTintColor: Colors.transparent,
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFF45C95A),
            foregroundColor: Colors.white,
            shape: const StadiumBorder(),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFF2B3A42),
            side: const BorderSide(color: Color(0xFF2B3A42)),
            shape: const StadiumBorder(),
          ),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      // Switches automatically when deviceIpProvider changes.
      home: ip == null ? const SetupPage() : const HomePage(),
    );
  }
}
