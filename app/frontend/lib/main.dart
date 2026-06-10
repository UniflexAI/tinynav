import 'package:flutter/foundation.dart';
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
          seedColor: const Color(0xFF38D3FF),
          primary: const Color(0xFF38D3FF),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
        fontFamily: 'RobotoLocal',
        scaffoldBackgroundColor: const Color(0xFF0B1118),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF0F1822),
          foregroundColor: Color(0xFFE6EEF7),
          elevation: 0,
          surfaceTintColor: Colors.transparent,
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFF38D3FF),
            foregroundColor: const Color(0xFF04131B),
            shape: const StadiumBorder(),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFFB8C8D9),
            side: const BorderSide(color: Color(0xFF2D3E50)),
            shape: const StadiumBorder(),
          ),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          color: const Color(0xFF111A24),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      builder: (context, child) {
        if (!kIsWeb || child == null) return child ?? const SizedBox.shrink();

        return LayoutBuilder(
          builder: (context, constraints) {
            final viewportAspect = constraints.maxWidth / constraints.maxHeight;

            // Portrait (< 1:1): keep default stretch/fill behavior.
            if (viewportAspect < 1.0) {
              return child;
            }

            // Landscape (>= 1:1): lock app content to 1:1, top-aligned.
            final side = constraints.maxHeight;
            return ColoredBox(
              color: const Color(0xFF11161C),
              child: Align(
                alignment: Alignment.topCenter,
                child: SizedBox(
                  width: side,
                  height: side,
                  child: ClipRect(child: child),
                ),
              ),
            );
          },
        );
      },
      // Switches automatically when deviceIpProvider changes.
      home: ip == null ? const SetupPage() : const HomePage(),
    );
  }
}
