import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';

const _bg = Color(0xFF0A0A0A);
const _green = Color(0xFF00E676);
const _red = Color(0xFFFF5252);
const _muted = Color(0xFF616161);
const _text = Color(0xFFE0E0E0);

class SetupPage extends ConsumerStatefulWidget {
  const SetupPage({super.key});

  @override
  ConsumerState<SetupPage> createState() => _SetupPageState();
}

class _SetupPageState extends ConsumerState<SetupPage> {
  final _ipController = TextEditingController();
  String? _testResult;
  bool _testOk = false;
  bool _testing = false;

  @override
  void dispose() {
    _ipController.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    final ip = _ipController.text.trim();
    if (ip.isEmpty) return;
    setState(() {
      _testing = true;
      _testResult = null;
    });
    try {
      final dio = Dio(BaseOptions(
        baseUrl: 'http://$ip:8000',
        connectTimeout: const Duration(seconds: 5),
      ));
      final resp = await dio.get('/device/info');
      final data = resp.data as Map<String, dynamic>;
      setState(() {
        _testOk = true;
        _testResult = 'Connected: ${data['deviceId']}  v${data['firmwareVersion']}';
      });
    } catch (e) {
      setState(() {
        _testOk = false;
        _testResult = 'Failed: $e';
      });
    } finally {
      setState(() => _testing = false);
    }
  }

  Future<void> _connect() async {
    final ip = _ipController.text.trim();
    if (ip.isEmpty) return;
    final prefs = ref.read(sharedPreferencesProvider);
    await prefs.setString('device_ip', ip);
    ref.read(deviceIpProvider.notifier).state = ip;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(32),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 400),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Image.asset('assets/images/tinynav_dark.png', width: 80, height: 80),
                const SizedBox(height: 16),
                const Text('TINYNAV',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w300,
                      letterSpacing: 4,
                      color: _text,
                    )),
                const SizedBox(height: 4),
                const Text('DEVICE CONNECT',
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 2,
                      color: _muted,
                    )),
                const SizedBox(height: 32),
                TextField(
                  controller: _ipController,
                  style: const TextStyle(
                    fontFamily: 'RobotoLocal',
                    fontWeight: FontWeight.w600,
                    letterSpacing: 1,
                    color: _text,
                  ),
                  decoration: InputDecoration(
                    labelText: 'DEVICE IP',
                    labelStyle: const TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.5,
                      color: _muted,
                    ),
                    hintText: '192.168.1.100',
                    hintStyle: TextStyle(color: _muted.withOpacity(0.3)),
                    prefixIcon: const Icon(Icons.router_outlined, color: _muted, size: 18),
                    enabledBorder: const OutlineInputBorder(
                      borderSide: BorderSide(color: Color(0xFF2A2A2A), width: 1),
                      borderRadius: BorderRadius.zero,
                    ),
                    focusedBorder: const OutlineInputBorder(
                      borderSide: BorderSide(color: _green, width: 1),
                      borderRadius: BorderRadius.zero,
                    ),
                  ),
                  keyboardType: TextInputType.number,
                  onSubmitted: (_) => _testConnection(),
                ),
                const SizedBox(height: 16),
                if (_testResult != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 16),
                    child: Row(children: [
                      Icon(
                        _testOk ? Icons.check_circle : Icons.error_outline,
                        color: _testOk ? _green : _red,
                        size: 14,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _testResult!,
                          style: TextStyle(
                            fontSize: 12,
                            color: _testOk ? _green : _red,
                            fontFamily: 'RobotoLocal',
                          ),
                        ),
                      ),
                    ]),
                  ),
                Row(children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: _testing ? null : _testConnection,
                      child: _testing
                          ? const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(strokeWidth: 1.5, color: _green),
                            )
                          : const Text('TEST'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 2,
                    child: FilledButton(
                      onPressed: _connect,
                      child: const Text('CONNECT'),
                    ),
                  ),
                ]),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
