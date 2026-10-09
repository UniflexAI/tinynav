import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';

// Stair mode settings, kept on the robot (GET/PUT /nav/stair/settings) and used the next time stair mode
// starts from the Operate page's Stairs button.

class StairTab extends ConsumerStatefulWidget {
  const StairTab({super.key});

  @override
  ConsumerState<StairTab> createState() => _StairTabState();
}

class _StairTabState extends ConsumerState<StairTab> {
  final _cameraHeight = TextEditingController();
  final _floors = TextEditingController();
  final _landingsPerFloor = TextEditingController();
  bool _loading = true;
  bool _saving = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _cameraHeight.dispose();
    _floors.dispose();
    _landingsPerFloor.dispose();
    super.dispose();
  }

  void _fill(Map<String, dynamic> s) {
    final h = s['camera_height'];
    _cameraHeight.text = h == null ? '' : (h as num).toStringAsFixed(2);
    _floors.text = '${s['floors'] ?? 0}';
    _landingsPerFloor.text = '${s['landings_per_floor'] ?? 2}';
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final r = await ref.read(dioProvider).get('/nav/stair/settings');
      _fill(Map<String, dynamic>.from(r.data as Map));
    } on DioException catch (e) {
      _error = e.response?.data?['detail']?.toString() ?? e.message ?? 'Error';
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _snack(String text, {bool error = false}) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(text),
      backgroundColor: error ? Colors.red : const Color(0xFF45C95A),
    ));
  }

  Future<void> _save() async {
    final hText = _cameraHeight.text.trim();
    final h = hText.isEmpty ? null : double.tryParse(hText);
    final floors = int.tryParse(_floors.text.trim());
    final perFloor = int.tryParse(_landingsPerFloor.text.trim());
    if (hText.isNotEmpty && (h == null || h < 0.2 || h > 1.2)) {
      _snack('Camera height: 0.20 - 1.20 m, or empty for the robot default', error: true);
      return;
    }
    if (floors == null || floors < 0 || floors > 50) {
      _snack('Floors: 0 - 50 (0: no limit)', error: true);
      return;
    }
    if (perFloor == null || perFloor < 1 || perFloor > 4) {
      _snack('Landings per floor: 1 - 4', error: true);
      return;
    }
    setState(() => _saving = true);
    try {
      final r = await ref.read(dioProvider).put('/nav/stair/settings', data: {
        'camera_height': h,
        'floors': floors,
        'landings_per_floor': perFloor,
      });
      _fill(Map<String, dynamic>.from(r.data as Map));
      if (mounted) _snack('Saved, used the next time stair mode starts');
    } on DioException catch (e) {
      if (mounted) _snack(e.response?.data?['detail']?.toString() ?? e.message ?? 'Error', error: true);
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final status = ref.watch(deviceStatusProvider).valueOrNull;
    // stairStatus is "<direction> <status> well_side=.. cam_h=.. pitch=.. landings=2/4"
    final parts = (status?.stairStatus ?? '').split(' ');
    String field(String key) =>
        parts.firstWhere((p) => p.startsWith('$key='), orElse: () => '').replaceFirst('$key=', '');
    final active = status?.stairMode != null;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // ── Settings ──────────────────────────────────────────────────
        _Card(
          icon: Icons.tune_rounded,
          title: 'Settings',
          children: _loading
              ? [const Center(child: Padding(padding: EdgeInsets.all(8), child: CircularProgressIndicator(strokeWidth: 2)))]
              : [
                  if (_error != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Text(_error!, style: const TextStyle(color: Colors.red)),
                    ),
                  _NumberField(
                    controller: _cameraHeight,
                    label: 'Camera height (m)',
                    hint: 'empty: robot default (go2 0.45)',
                    decimal: true,
                  ),
                  const _Help('Camera above the ground. Refined on flat ground within ±25% of this value.'),
                  const SizedBox(height: 12),
                  _NumberField(controller: _floors, label: 'Floors', hint: '0: no limit'),
                  const _Help('Floors to go, then stop on that landing ("arrived") and leave stair mode.'),
                  const SizedBox(height: 12),
                  _NumberField(controller: _landingsPerFloor, label: 'Landings per floor', hint: '2'),
                  const _Help('2 in a U-shaped stairwell. Stops on landing number floors × this.'),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: _saving ? null : _save,
                          icon: _saving
                              ? const SizedBox(
                                  width: 14,
                                  height: 14,
                                  child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                                )
                              : const Icon(Icons.save_rounded, size: 16),
                          label: const Text('Save'),
                        ),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton(onPressed: _saving ? null : _load, child: const Text('Reload')),
                    ],
                  ),
                  if (active)
                    const Padding(
                      padding: EdgeInsets.only(top: 8),
                      child: _Help('Stair mode is running: changes apply when it is started again.'),
                    ),
                ],
        ),
        const SizedBox(height: 12),
        // ── Live ──────────────────────────────────────────────────────
        _Card(
          icon: Icons.stairs,
          title: 'Live',
          children: [
            _Row('Stair mode', active ? '${status!.stairMode}' : 'off'),
            _Row('Status', active && parts.length > 1 ? parts[1] : '—'),
            _Row('Landings', active ? (field('landings').isEmpty ? '—' : field('landings')) : '—'),
            _Row('Camera height', active && field('cam_h').isNotEmpty ? '${field('cam_h')} m' : '—'),
          ],
        ),
      ],
    );
  }
}

// ── Small widgets, styled like the Device page ────────────────────────────────

class _Card extends StatelessWidget {
  final IconData icon;
  final String title;
  final List<Widget> children;
  const _Card({required this.icon, required this.title, required this.children});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: const Color(0xFF111A24),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: Color(0xFF2A3B4D)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(icon, size: 18, color: const Color(0xFF9EC7E8)),
              const SizedBox(width: 8),
              Text(title,
                  style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 14, color: Color(0xFFE8F2FF))),
            ]),
            const Divider(height: 20, color: Color(0xFF243446)),
            ...children,
          ],
        ),
      ),
    );
  }
}

class _NumberField extends StatelessWidget {
  final TextEditingController controller;
  final String label;
  final String hint;
  final bool decimal;
  const _NumberField({required this.controller, required this.label, required this.hint, this.decimal = false});

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      keyboardType: TextInputType.numberWithOptions(decimal: decimal),
      style: const TextStyle(color: Color(0xFFE6EEF7)),
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        border: const OutlineInputBorder(),
        isDense: true,
      ),
    );
  }
}

class _Help extends StatelessWidget {
  final String text;
  const _Help(this.text);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Text(text, style: const TextStyle(fontSize: 12, color: Color(0xFF77889A))),
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  const _Row(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
            child: Text(label,
                style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: Color(0xFF9FB0C3))),
          ),
          Text(value, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFFE6EEF7))),
        ],
      ),
    );
  }
}
