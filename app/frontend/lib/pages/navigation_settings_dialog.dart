import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/providers.dart';

class NavigationSettingsDialog extends ConsumerStatefulWidget {
  const NavigationSettingsDialog({super.key});
  @override
  ConsumerState<NavigationSettingsDialog> createState() =>
      _NavigationSettingsDialogState();
}

class _NavigationSettingsDialogState
    extends ConsumerState<NavigationSettingsDialog> {
  final _form = GlobalKey<FormState>();
  final _inputs = <String, TextEditingController>{};
  Map<String, dynamic> _presets = {};
  String? _robot;
  String? _error;
  bool _loading = true, _saving = false;
  static const _labels = {
    'safety_radius': 'Safety radius (m)',
    'dilation_cells': 'Dilation (cells)',
    'decay_factor': 'Decay factor (0–1)',
    'max_linear_vel': 'Maximum linear speed (m/s)',
    'max_angular_vel': 'Maximum angular speed (rad/s)',
    'robot_z_bottom': 'Z bottom relative to camera (m)',
    'robot_z_top': 'Z top relative to camera (m)',
  };

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    for (final c in _inputs.values) {
      c.dispose();
    }
    super.dispose();
  }

  void _fill(Map<String, dynamic> values) {
    _robot = values['robot_type'] as String;
    for (final key in _labels.keys) {
      (_inputs[key] ??= TextEditingController()).text = values[key].toString();
    }
  }

  void _usePreset(String? name) {
    if (name == null) return;
    setState(() => _fill(Map<String, dynamic>.from(_presets[name] as Map)));
  }

  String _message(Object error) {
    if (error is DioException) {
      final data = error.response?.data;
      return data is Map
          ? '${data['detail'] ?? error.message}'
          : (error.message ?? 'Request failed');
    }
    return error.toString();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final response = await ref.read(dioProvider).get('/nav/settings');
      if (!mounted) return;
      setState(() {
        _presets = Map<String, dynamic>.from(response.data['presets'] as Map);
        _fill(Map<String, dynamic>.from(response.data['settings'] as Map));
        _loading = false;
      });
    } catch (error) {
      if (mounted) {
        setState(() {
          _error = _message(error);
          _loading = false;
        });
      }
    }
  }

  String? _validate(String key, String? text) {
    final value = double.tryParse(text ?? '');
    if (value == null || !value.isFinite) return 'Enter a finite number';
    if (key == 'dilation_cells' && (int.tryParse(text!) == null || value < 0)) {
      return 'Enter a non-negative integer';
    }
    if (key == 'safety_radius' && value < 0) return 'Must be non-negative';
    if (key == 'decay_factor' && (value < 0 || value > 1)) {
      return 'Must be between 0 and 1';
    }
    if (key.startsWith('max_') && value <= 0) return 'Must be positive';
    if (key == 'robot_z_top') {
      final bottom = double.tryParse(_inputs['robot_z_bottom']!.text);
      if (bottom != null && value <= bottom) return 'Must be above Z bottom';
    }
    return null;
  }

  Future<void> _save() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _saving = true;
      _error = null;
    });
    final values = <String, dynamic>{'robot_type': _robot};
    for (final key in _labels.keys) {
      values[key] = key == 'dilation_cells'
          ? int.parse(_inputs[key]!.text)
          : double.parse(_inputs[key]!.text);
    }
    try {
      final response =
          await ref.read(dioProvider).put('/nav/settings', data: values);
      if (!mounted) return;
      if (response.data['changed'] == true) {
        ref.read(activeNavPoisProvider.notifier).state = const [];
      }
      Navigator.of(context).pop();
    } catch (error) {
      if (mounted) {
        setState(() {
          _saving = false;
          _error = _message(error);
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) => PopScope(
        canPop: !_saving,
        child: AlertDialog(
          title: const Text('Navigation settings'),
          content: SizedBox(
            width: 440,
            child: SingleChildScrollView(
              child: _loading
                  ? const Center(child: CircularProgressIndicator())
                  : _robot == null
                      ? Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(_error ?? 'Unable to load settings'),
                            TextButton(
                                onPressed: _load, child: const Text('Retry')),
                          ],
                        )
                      : Form(
                          key: _form,
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              const Text(
                                'Saving stops current navigation and applies the new settings.',
                              ),
                              const SizedBox(height: 16),
                              DropdownButtonFormField<String>(
                                initialValue: _robot,
                                decoration: const InputDecoration(
                                  labelText: 'Robot type',
                                ),
                                items: _presets.keys
                                    .map(
                                      (name) => DropdownMenuItem(
                                        value: name,
                                        child: Text(name.toUpperCase()),
                                      ),
                                    )
                                    .toList(),
                                onChanged: _saving ? null : _usePreset,
                              ),
                              for (final entry in _labels.entries)
                                Padding(
                                  padding: const EdgeInsets.only(top: 12),
                                  child: TextFormField(
                                    controller: _inputs[entry.key],
                                    enabled: !_saving,
                                    keyboardType:
                                        const TextInputType.numberWithOptions(
                                      decimal: true,
                                      signed: true,
                                    ),
                                    decoration:
                                        InputDecoration(labelText: entry.value),
                                    validator: (text) =>
                                        _validate(entry.key, text),
                                  ),
                                ),
                              const Padding(
                                padding: EdgeInsets.only(top: 12),
                                child: Text(
                                  'Decay multiplies old occupancy on every update. Default: 0.99. Larger values retain obstacles longer.',
                                ),
                              ),
                              if (_error != null)
                                Padding(
                                  padding: const EdgeInsets.only(top: 12),
                                  child: Text(
                                    _error!,
                                    style: const TextStyle(color: Colors.red),
                                  ),
                                ),
                            ],
                          ),
                        ),
            ),
          ),
          actions: [
            TextButton(
              onPressed:
                  _saving || _robot == null ? null : () => _usePreset(_robot),
              child: const Text('Restore defaults'),
            ),
            TextButton(
              onPressed: _saving ? null : () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: _loading || _saving || _robot == null ? null : _save,
              child: Text(_saving ? 'Saving…' : 'Save'),
            ),
          ],
        ),
      );
}
