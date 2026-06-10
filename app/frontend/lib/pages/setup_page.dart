import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';

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
    // Updating this provider switches MaterialApp.home to HomePage automatically.
    ref.read(deviceIpProvider.notifier).state = ip;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(32),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 400),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.navigation_outlined, size: 72, color: Color(0xFF1565C0)),
                const SizedBox(height: 16),
                const Text('TinyNav',
                    style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold)),
                const SizedBox(height: 6),
                const Text('Enter the device IP address',
                    style: TextStyle(color: Colors.grey)),
                const SizedBox(height: 32),
                TextField(
                  controller: _ipController,
                  decoration: const InputDecoration(
                    labelText: 'Device IP',
                    hintText: '192.168.1.100',
                    prefixIcon: Icon(Icons.router_outlined),
                    border: OutlineInputBorder(),
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
                        color: _testOk ? Colors.green : Colors.red,
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _testResult!,
                          style: TextStyle(color: _testOk ? Colors.green : Colors.red),
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
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Text('Test'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 2,
                    child: FilledButton(
                      onPressed: _connect,
                      child: const Text('Connect'),
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
