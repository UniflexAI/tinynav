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
        brightness: Brightness.dark,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00E676),        // fluorescent green
          secondary: Color(0xFF00BCD4),      // cyan
          surface: Color(0xFF0A0A0A),        // near-black
          onSurface: Color(0xFFE0E0E0),      // light gray text
          error: Color(0xFFFF5252),
        ),
        useMaterial3: true,
        fontFamily: 'RobotoLocal',
        scaffoldBackgroundColor: const Color(0xFF0A0A0A),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF0A0A0A),
          foregroundColor: Color(0xFF00E676),
          elevation: 0,
          surfaceTintColor: Colors.transparent,
          titleTextStyle: TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 15,
            color: Color(0xFF00E676),
            letterSpacing: 0.5,
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFF00E676),
            foregroundColor: const Color(0xFF0A0A0A),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(4),
            ),
            textStyle: const TextStyle(
              fontWeight: FontWeight.w700,
              letterSpacing: 0.5,
            ),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFF00E676),
            side: const BorderSide(color: Color(0xFF00E676), width: 1),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(4),
            ),
          ),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          color: const Color(0xFF111111),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(4),
            side: const BorderSide(color: Color(0xFF1A1A1A), width: 1),
          ),
        ),
        dividerTheme: const DividerThemeData(
          color: Color(0xFF1A1A1A),
          thickness: 1,
        ),
        iconTheme: const IconThemeData(
          color: Color(0xFF9E9E9E),
          size: 20,
        ),
        textTheme: const TextTheme(
          headlineLarge: TextStyle(
            fontWeight: FontWeight.w300,
            fontSize: 28,
            letterSpacing: -0.5,
            color: Color(0xFFE0E0E0),
          ),
          titleLarge: TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 15,
            letterSpacing: 0.3,
            color: Color(0xFFE0E0E0),
          ),
          bodyMedium: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w400,
            color: Color(0xFF9E9E9E),
          ),
          labelSmall: TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w600,
            letterSpacing: 1.0,
            color: Color(0xFF616161),
          ),
        ),
      ),
      // Switches automatically when deviceIpProvider changes.
      home: ip == null ? const SetupPage() : const HomePage(),
    );
  }
}
