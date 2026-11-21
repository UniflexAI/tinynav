import 'dart:async';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/providers.dart';
import '../widgets/simulation_view.dart';

class SimulationPage extends ConsumerStatefulWidget {
  const SimulationPage({super.key});
  @override
  ConsumerState<SimulationPage> createState() => _SimulationPageState();
}

class _SimulationPageState extends ConsumerState<SimulationPage> {
  Timer? _timer;
  Map<String, dynamic>? _state;
  String? _viewerUrl, _error;
  bool _busy = false, _polling = false, _connected = false, _settingsLoaded = false;
  int _floors = 9, _speed = 1;
  double _depth = 1.2;
  bool _noise = true;

  @override
  void initState() {
    super.initState();
    _connect();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) => _poll());
  }
  @override
  void dispose() { _timer?.cancel(); super.dispose(); }

  Future<void> _connect() async {
    try {
      final dio = ref.read(dioProvider);
      final result = await dio.get('/simulation');
      final host = Uri.parse(ref.read(baseUrlProvider)!).host;
      if (!mounted) return;
      setState(() => _viewerUrl = Uri(scheme: 'http', host: host,
        port: result.data['viewer_port'] as int, path: '/').toString());
      await _poll();
    } catch (e) { if (mounted) setState(() => _error = '$e'); }
  }

  Future<void> _poll() async {
    if (_polling || !mounted) return;
    _polling = true;
    try {
      final result = await ref.read(dioProvider).get('/simulation/state');
      if (mounted) setState(() {
        _state = Map<String, dynamic>.from(result.data);
        _connected = _state!['ready'] == true;
        if (!_settingsLoaded && _connected) {
          _floors = (_state!['floors'] as num).toInt();
          _depth = (_state!['landing_depth'] as num).toDouble();
          _speed = (_state!['speed'] as num).toInt();
          _noise = (_state!['odom_noise'] as Map)['mode'] != 'none';
          _settingsLoaded = true;
        }
      });
    } catch (_) { if (mounted) setState(() => _connected = false); }
    finally { _polling = false; }
  }

  Map<String, dynamic> _config(String direction) => {
    'direction': direction, 'floors': _floors, 'landing_depth': _depth,
    'noise_mode': _noise ? 'combined' : 'none', 'noise_scale': 0.5,
    'noise_seed': 7, 'speed': _speed,
  };

  Future<void> _command(String action, [Map<String, dynamic>? data]) async {
    setState(() { _busy = true; _error = null; });
    try {
      final result = await ref.read(dioProvider).post('/simulation/$action', data: data ?? {},
        options: Options(receiveTimeout: const Duration(seconds: 70)));
      if (mounted) setState(() { _state = Map<String, dynamic>.from(result.data); _connected = true; });
    } on DioException catch (e) {
      if (mounted) setState(() => _error = e.response?.data?['detail']?.toString() ?? e.message);
    } finally { if (mounted) setState(() => _busy = false); }
  }

  @override
  Widget build(BuildContext context) {
    final s = _state ?? {};
    final ended = s['validation_passed'] == true || s['cancelled'] == true || (s['validation_error'] ?? '') != '';
    final disabled = _busy || !_connected;
    final direction = s['direction'] as String? ?? 'down';
    final noise = s['odom_noise'] as Map? ?? {};
    final status = s['validation_passed'] == true ? '已到达 · 停稳检查通过'
      : (s['validation_error'] ?? '') != '' ? '安全断言中止：${s['validation_error']}'
      : s['cancelled'] == true ? '任务已停止' : s['paused'] == true ? '已暂停' : '运行中';
    return Scaffold(
      appBar: AppBar(title: const Text('TinyNav × MuJoCo'), actions: [
        Center(child: Text(_connected ? '● 仿真已连接  ' : '● 仿真连接中断  ',
          style: TextStyle(color: _connected ? Colors.greenAccent : Colors.orangeAccent))),
      ]),
      body: LayoutBuilder(builder: (context, constraints) {
        final controls = SizedBox(width: 278, child: ListView(padding: const EdgeInsets.all(16), children: [
          const Text('楼梯任务', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text('10 个整层平台 · 楼梯净宽 1.2 m\n每层 2 段楼梯', style: TextStyle(color: Colors.white60)),
          const SizedBox(height: 16),
          _select<int>('跨越层数', _floors, List.generate(9, (i) => i + 1), (v) => setState(() => _floors = v)),
          _select<double>('平台进深（m）', _depth, [1.2, 1.75], (v) => setState(() => _depth = v)),
          _select<int>('仿真速度', _speed, [1, 2, 4], (v) => setState(() => _speed = v)),
          SwitchListTile(contentPadding: EdgeInsets.zero, title: const Text('Odom 噪声'),
            subtitle: const Text('combined · 0.5× · seed 7'), value: _noise,
            onChanged: _busy ? null : (v) => setState(() => _noise = v)),
          const Text('发起任务时重置环境，配置生效。', style: TextStyle(fontSize: 12, color: Colors.white54)),
          const SizedBox(height: 16),
          FilledButton.icon(onPressed: disabled ? null : () => _command('start', _config('up')),
            icon: const Icon(Icons.stairs), label: const Text('发起上楼任务')),
          const SizedBox(height: 8),
          FilledButton.icon(onPressed: disabled ? null : () => _command('start', _config('down')),
            icon: const Icon(Icons.arrow_downward), label: const Text('发起下楼任务')),
          const SizedBox(height: 8),
          OutlinedButton(onPressed: disabled || ended ? null : () => _command(s['paused'] == true ? 'resume' : 'pause'),
            child: Text(s['paused'] == true ? '继续仿真' : '暂停仿真')),
          OutlinedButton(onPressed: disabled ? null : () => _command('reset', _config(direction)),
            child: const Text('重置环境（暂停）')),
          OutlinedButton(onPressed: disabled || ended ? null : () => _command('stop'), child: const Text('停止任务')),
          const Divider(height: 28),
          const Text('第三方视角', style: TextStyle(fontWeight: FontWeight.bold)),
          Wrap(spacing: 4, children: [
            for (final entry in {'自由视角': 'Free orbit', '跟随机器狗': 'Follow robot', '全景': 'Overview'}.entries)
              TextButton(onPressed: disabled ? null : () => _command('camera', {'mode': entry.value}), child: Text(entry.key)),
          ]),
          const Text('自由视角：鼠标拖动旋转，滚轮缩放，右键平移。', style: TextStyle(fontSize: 12, color: Colors.white54)),
          const Divider(height: 28),
          Text(status, style: TextStyle(color: s['validation_error'] == '' ? Colors.greenAccent : Colors.orangeAccent)),
          Text('${direction.toUpperCase()} · ${s['landings'] ?? 0} / ${(s['floors'] as num? ?? 0) * 2} 段\n仿真时间 ${s['time'] ?? 0} s'),
          Text('生效噪声：${noise['mode'] ?? '—'} · ${noise['scale'] ?? '—'}× · seed ${noise['seed'] ?? '—'}',
            style: const TextStyle(fontSize: 12, color: Colors.white60)),
          const SizedBox(height: 8),
          const Text('TinyNav 目标 → 局部规划 → 速度跟踪\nGo2 策略 → MuJoCo 物理仿真\n安全断言保持启用', style: TextStyle(fontSize: 12, color: Colors.white54)),
          if (_busy) const Padding(padding: EdgeInsets.all(12), child: LinearProgressIndicator()),
          if (_error != null) SelectableText(_error!, style: const TextStyle(color: Colors.redAccent)),
        ]));
        final view = _viewerUrl == null ? const Center(child: CircularProgressIndicator())
          : SimulationView(key: ValueKey(_viewerUrl), url: _viewerUrl!);
        if (constraints.maxWidth < 760) {
          return Column(children: [SizedBox(height: 260, child: Center(child: controls)), Expanded(child: view)]);
        }
        return Row(children: [controls, const VerticalDivider(width: 1), Expanded(child: view)]);
      }),
    );
  }

  Widget _select<T>(String label, T value, List<T> options, ValueChanged<T> onChanged) => Padding(
    padding: const EdgeInsets.only(bottom: 10),
    child: DropdownButtonFormField<T>(key: ValueKey('$label:$value'), initialValue: value, decoration: InputDecoration(labelText: label),
      items: options.map((v) => DropdownMenuItem(value: v, child: Text('$v'))).toList(),
      onChanged: _busy ? null : (v) { if (v != null) onChanged(v); }),
  );
}
