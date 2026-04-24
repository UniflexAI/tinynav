import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../core/models.dart';
import '../core/providers.dart';
import 'map_painter.dart';
import 'planning_painter.dart';

const double _maxLinear = 0.5;   // m/s
const double _maxAngular = 1.0;  // rad/s

// ── Main widget ───────────────────────────────────────────────────────────────

class OperateTab extends ConsumerStatefulWidget {
  const OperateTab({super.key});

  @override
  ConsumerState<OperateTab> createState() => _OperateTabState();
}

class _OperateTabState extends ConsumerState<OperateTab> {
  WebSocketChannel? _teleopChannel;
  double _linearX = 0, _linearY = 0, _angularZ = 0;

  bool _showObstacle = true;
  bool _showEsdf = true;
  bool _showTrajectory = true;
  bool _showGlobalPath = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _connectTeleop());
  }

  void _connectTeleop() {
    final ip = ref.read(deviceIpProvider);
    if (ip == null) return;
    try {
      _teleopChannel = WebSocketChannel.connect(
        Uri.parse('ws://$ip:8000/ws/teleop'),
      );
    } catch (_) {}
  }

  void _sendVelocity() {
    try {
      _teleopChannel?.sink.add(jsonEncode({
        'linear_x': _linearX,
        'linear_y': _linearY,
        'angular_z': _angularZ,
      }));
    } catch (_) {}
  }

  void _onLeftJoystick(double x, double y) {
    _linearX = -y * _maxLinear;
    _linearY = -x * _maxLinear;
    _sendVelocity();
  }

  void _onRightJoystick(double x, double y) {
    _angularZ = -x * _maxAngular;
    _sendVelocity();
  }

  Future<void> _emergencyStop() async {
    _linearX = 0; _linearY = 0; _angularZ = 0;
    _sendVelocity();
    try { await ref.read(dioProvider).post('/nav/restart'); } catch (_) {}
  }

  @override
  void dispose() {
    _teleopChannel?.sink.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final poisAsync = ref.watch(poisProvider);
    final poseAsync = ref.watch(poseStreamProvider);
    final planningAsync = ref.watch(planningStreamProvider);
    final planning = planningAsync.valueOrNull;

    return Column(
      children: [
        // ── Camera (1/4) ──────────────────────────────────────────────
        const Expanded(flex: 2, child: _CameraPanel()),
        const Divider(height: 1, thickness: 1, color: Color(0xFFE0E0E0)),
        // ── Map / planning view (3/8) ─────────────────────────────────
        Expanded(
          flex: 3,
          child: Stack(
            children: [
              Positioned.fill(
                child: _LocalPlanningView(
                  planning: planning,
                  showObstacle: _showObstacle,
                  showEsdf: _showEsdf,
                  showTrajectory: _showTrajectory,
                  showGlobalPath: _showGlobalPath,
                ),
              ),
              if (planning != null)
                Positioned(
                  top: 8,
                  left: 8,
                  child: _LocalizationChip(localized: planning.localized),
                ),
              Positioned(
                top: 8,
                right: 8,
                child: _LayerTogglePanel(
                  showObstacle: _showObstacle,
                  showEsdf: _showEsdf,
                  showTrajectory: _showTrajectory,
                  showGlobalPath: _showGlobalPath,
                  onChanged: (obs, esdf, traj, gp) => setState(() {
                    _showObstacle = obs;
                    _showEsdf = esdf;
                    _showTrajectory = traj;
                    _showGlobalPath = gp;
                  }),
                ),
              ),
              Positioned(
                bottom: 10,
                left: 10,
                child: _PoiButton(
                  poisAsync: poisAsync,
                  statusAsync: ref.watch(deviceStatusProvider),
                  pose: poseAsync.valueOrNull,
                ),
              ),
              Positioned(
                bottom: 10,
                right: 10,
                child: _NavNodesButton(statusAsync: ref.watch(deviceStatusProvider)),
              ),
            ],
          ),
        ),
        const Divider(height: 1, thickness: 1, color: Color(0xFFE0E0E0)),
        // ── Joystick panel (1/4) ──────────────────────────────────────
        Expanded(
          flex: 2,
          child: _JoystickPanel(
            onLeft: _onLeftJoystick,
            onRight: _onRightJoystick,
            onStop: _emergencyStop,
          ),
        ),
      ],
    );
  }
}

// ── Global map view (nav path in map frame on SLAM map PNG) ──────────────────

class _GlobalMapView extends StatelessWidget {
  final MapInfo mapInfo;
  final String baseUrl;
  final PlanningState? planning;
  final List<Poi> pois;

  const _GlobalMapView({
    required this.mapInfo,
    required this.baseUrl,
    this.planning,
    this.pois = const [],
  });

  @override
  Widget build(BuildContext context) {
    final p = planning;
    return Stack(
      fit: StackFit.expand,
      children: [
        Container(color: const Color(0xFF0D1117)),
        Center(
          child: AspectRatio(
            aspectRatio: mapInfo.width / mapInfo.height,
            child: InteractiveViewer(
              minScale: 0.5,
              maxScale: 8.0,
              boundaryMargin: const EdgeInsets.all(double.infinity),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.network(
                    '$baseUrl${mapInfo.imageUrl}',
                    fit: BoxFit.fill,
                    gaplessPlayback: true,
                    errorBuilder: (_, __, ___) => const ColoredBox(color: Color(0xFF1A1A2E)),
                  ),
                  if (p != null)
                    CustomPaint(
                      painter: MapOverlayPainter(
                        mapInfo: mapInfo,
                        pose: p.mapPose,
                        pois: pois,
                        globalPath: p.globalPath,
                        showGlobalPath: true,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ── Local planning view ───────────────────────────────────────────────────────

class _LocalPlanningView extends StatelessWidget {
  final PlanningState? planning;
  final bool showObstacle;
  final bool showEsdf;
  final bool showTrajectory;
  final bool showGlobalPath;

  const _LocalPlanningView({
    this.planning,
    this.showObstacle = true,
    this.showEsdf = false,
    this.showTrajectory = false,
    this.showGlobalPath = true,
  });

  @override
  Widget build(BuildContext context) {
    final p = planning;
    return Stack(
      fit: StackFit.expand,
      children: [
        Container(color: const Color(0xFF0D1117)),
        Center(
          child: AspectRatio(
            aspectRatio: 1.0,
            child: InteractiveViewer(
              minScale: 0.5,
              maxScale: 8.0,
              boundaryMargin: const EdgeInsets.all(double.infinity),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (showEsdf && p?.esdfImage != null)
                    Opacity(
                      opacity: 0.85,
                      child: Image.memory(p!.esdfImage!, fit: BoxFit.fill, gaplessPlayback: true),
                    ),
                  if (showObstacle && p?.obstacleImage != null)
                    Opacity(
                      opacity: 0.45,
                      child: Image.memory(p!.obstacleImage!, fit: BoxFit.fill, gaplessPlayback: true),
                    ),
                  if (p != null)
                    CustomPaint(
                      painter: LocalPlanningPainter(
                        trajectory: p.trajectory,
                        globalPath: p.globalPath,
                        gridInfo: p.gridInfo,
                        odomPose: p.odomPose,
                        showTrajectory: showTrajectory,
                        showGlobalPath: showGlobalPath,
                        navTargetPose: p.navTargetPose,
                      ),
                    )
                  else
                    const Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.map_outlined, size: 48, color: Colors.white24),
                          SizedBox(height: 8),
                          Text('Waiting for planning data…',
                              style: TextStyle(color: Colors.white38, fontSize: 13)),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _LayerTogglePanel extends StatefulWidget {
  final bool showObstacle;
  final bool showEsdf;
  final bool showTrajectory;
  final bool showGlobalPath;
  final void Function(bool obs, bool esdf, bool traj, bool gp) onChanged;

  const _LayerTogglePanel({
    required this.showObstacle,
    required this.showEsdf,
    required this.showTrajectory,
    required this.showGlobalPath,
    required this.onChanged,
  });

  @override
  State<_LayerTogglePanel> createState() => _LayerTogglePanelState();
}

class _LayerTogglePanelState extends State<_LayerTogglePanel> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      mainAxisSize: MainAxisSize.min,
      children: [
        GestureDetector(
          onTap: () => setState(() => _expanded = !_expanded),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.layers_outlined, color: Colors.white70, size: 14),
                const SizedBox(width: 4),
                const Text('Layers', style: TextStyle(color: Colors.white70, fontSize: 12)),
                const SizedBox(width: 4),
                Icon(_expanded ? Icons.expand_less : Icons.expand_more,
                    color: Colors.white54, size: 14),
              ],
            ),
          ),
        ),
        if (_expanded) ...[
          const SizedBox(height: 4),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                _LayerRow('Obstacle', widget.showObstacle,
                    (v) => widget.onChanged(v, widget.showEsdf, widget.showTrajectory, widget.showGlobalPath)),
                _LayerRow('ESDF', widget.showEsdf,
                    (v) => widget.onChanged(widget.showObstacle, v, widget.showTrajectory, widget.showGlobalPath)),
                _LayerRow('Trajectory', widget.showTrajectory,
                    (v) => widget.onChanged(widget.showObstacle, widget.showEsdf, v, widget.showGlobalPath)),
                _LayerRow('Global Path', widget.showGlobalPath,
                    (v) => widget.onChanged(widget.showObstacle, widget.showEsdf, widget.showTrajectory, v)),
              ],
            ),
          ),
        ],
      ],
    );
  }
}

class _LayerRow extends StatelessWidget {
  final String label;
  final bool value;
  final ValueChanged<bool> onChanged;
  const _LayerRow(this.label, this.value, this.onChanged);

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: 28,
          height: 28,
          child: Transform.scale(
            scale: 0.75,
            child: Switch(
              value: value,
              onChanged: onChanged,
              activeColor: const Color(0xFF45C95A),
            ),
          ),
        ),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }
}

// ── Localization chip ─────────────────────────────────────────────────────────

class _LocalizationChip extends StatelessWidget {
  final bool localized;
  const _LocalizationChip({required this.localized});

  @override
  Widget build(BuildContext context) {
    final dotColor = localized ? const Color(0xFF69F0AE) : Colors.redAccent;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.65),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7, height: 7,
            decoration: BoxDecoration(shape: BoxShape.circle, color: dotColor),
          ),
          const SizedBox(width: 6),
          Text(
            localized ? 'Localized' : 'Not Localized',
            style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }
}

// ── POI button + bottom sheet ─────────────────────────────────────────────────

class _PoiButton extends ConsumerWidget {
  final AsyncValue<List<Poi>> poisAsync;
  final AsyncValue<DeviceStatus> statusAsync;
  final Pose? pose;

  const _PoiButton({
    required this.poisAsync,
    required this.statusAsync,
    this.pose,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final count = poisAsync.valueOrNull?.length ?? 0;
    final isNavigating = statusAsync.valueOrNull?.rawState == 'navigation';

    return FilledButton.icon(
      onPressed: () => showModalBottomSheet(
        context: context,
        isScrollControlled: true,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        builder: (_) => _PoiSheet(pose: pose),
      ),
      style: FilledButton.styleFrom(
        backgroundColor: isNavigating
            ? const Color(0xFF34C759).withOpacity(0.9)
            : Colors.black87,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      ),
      icon: Icon(
        isNavigating ? Icons.navigation_rounded : Icons.place_outlined,
        size: 16,
      ),
      label: Text(isNavigating
          ? 'Navigating'
          : 'POIs${count > 0 ? ' ($count)' : ''}'),
    );
  }
}

class _PoiSheet extends ConsumerStatefulWidget {
  final Pose? pose;
  const _PoiSheet({this.pose});

  @override
  ConsumerState<_PoiSheet> createState() => _PoiSheetState();
}

class _PoiSheetState extends ConsumerState<_PoiSheet> {
  bool _canceling = false;

  Future<void> _addPoi() async {
    final pose = widget.pose;
    if (pose == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No pose — robot must be localized first')),
      );
      return;
    }
    final ctrl = TextEditingController();
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('New POI'),
        content: TextField(
          controller: ctrl,
          decoration: const InputDecoration(labelText: 'Name', hintText: 'e.g. Entrance'),
          autofocus: true,
          textCapitalization: TextCapitalization.sentences,
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Create')),
        ],
      ),
    );
    if (ok != true || ctrl.text.trim().isEmpty) return;
    try {
      await ref.read(dioProvider).post('/map/pois', data: {
        'name': ctrl.text.trim(),
        'position': [pose.x, pose.y, 0.0],
      });
      ref.invalidate(poisProvider);
    } on DioException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
          backgroundColor: Colors.red,
        ));
      }
    }
  }

  Future<void> _cancelNav() async {
    setState(() => _canceling = true);
    try {
      await ref.read(dioProvider).post('/nav/cancel');
    } on DioException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
          backgroundColor: Colors.red,
        ));
      }
    } finally {
      if (mounted) setState(() => _canceling = false);
    }
  }

  Future<void> _deletePoi(Poi poi) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete POI'),
        content: Text('Delete "${poi.name}"?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    try {
      await ref.read(dioProvider).delete('/poi/${poi.id}');
      ref.invalidate(poisProvider);
    } on DioException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
          backgroundColor: Colors.red,
        ));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final poisAsync = ref.watch(poisProvider);
    final statusAsync = ref.watch(deviceStatusProvider);
    final status = statusAsync.valueOrNull;
    final isNavigating = status?.rawState == 'navigation';
    final canGo = status != null && status.online && status.rawState == 'idle';

    return Padding(
      padding: EdgeInsets.fromLTRB(
          16, 12, 16, 24 + MediaQuery.of(context).viewInsets.bottom),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 36, height: 4,
              margin: const EdgeInsets.only(bottom: 14),
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          // ── Header ──────────────────────────────────────────────────
          Row(children: [
            const Icon(Icons.place_outlined, size: 20),
            const SizedBox(width: 8),
            const Text('POIs', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const Spacer(),
            if (isNavigating)
              OutlinedButton.icon(
                onPressed: _canceling ? null : _cancelNav,
                style: OutlinedButton.styleFrom(foregroundColor: Colors.red),
                icon: _canceling
                    ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.cancel_outlined, size: 16),
                label: const Text('Cancel Nav'),
              )
            else
              TextButton.icon(
                onPressed: _addPoi,
                icon: const Icon(Icons.add_location_alt_outlined, size: 18),
                label: const Text('Add here'),
              ),
          ]),
          const Divider(height: 20),
          // ── POI list ────────────────────────────────────────────────
          poisAsync.when(
            data: (pois) => pois.isEmpty
                ? const Padding(
                    padding: EdgeInsets.symmetric(vertical: 20),
                    child: Center(
                      child: Text('No POIs yet', style: TextStyle(color: Colors.grey)),
                    ),
                  )
                : Column(
                    children: pois
                        .map((poi) => _PoiTile(
                              poi: poi,
                              canGo: canGo,
                              onDelete: () => _deletePoi(poi),
                            ))
                        .toList(),
                  ),
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (e, _) => Text('$e', style: const TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }
}

class _PoiTile extends ConsumerStatefulWidget {
  final Poi poi;
  final bool canGo;
  final VoidCallback onDelete;

  const _PoiTile({required this.poi, required this.canGo, required this.onDelete});

  @override
  ConsumerState<_PoiTile> createState() => _PoiTileState();
}

class _PoiTileState extends ConsumerState<_PoiTile> {
  bool _loading = false;

  Future<void> _go() async {
    setState(() => _loading = true);
    try {
      await ref.read(dioProvider).post('/nav/go-to-poi', data: {'poi_id': widget.poi.id});
    } on DioException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
          backgroundColor: Colors.red,
        ));
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: const Icon(Icons.place, color: Colors.amber),
      title: Text(widget.poi.name),
      subtitle: Text(
        '(${widget.poi.x.toStringAsFixed(2)}, ${widget.poi.y.toStringAsFixed(2)})',
        style: const TextStyle(fontSize: 12),
      ),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _loading
              ? const SizedBox(width: 28, height: 28, child: CircularProgressIndicator(strokeWidth: 2))
              : FilledButton(
                  onPressed: widget.canGo ? _go : null,
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    minimumSize: Size.zero,
                  ),
                  child: const Text('Go', style: TextStyle(fontSize: 12)),
                ),
          const SizedBox(width: 4),
          IconButton(
            icon: const Icon(Icons.delete_outline, color: Colors.red, size: 18),
            onPressed: widget.onDelete,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
          ),
        ],
      ),
      dense: true,
      contentPadding: const EdgeInsets.symmetric(horizontal: 4),
    );
  }
}

// ── Nav nodes toggle button ───────────────────────────────────────────────────

class _NavNodesButton extends ConsumerStatefulWidget {
  final AsyncValue<DeviceStatus> statusAsync;
  const _NavNodesButton({required this.statusAsync});

  @override
  ConsumerState<_NavNodesButton> createState() => _NavNodesButtonState();
}

class _NavNodesButtonState extends ConsumerState<_NavNodesButton> {
  bool _loading = false;

  Future<void> _toggle(bool running) async {
    setState(() => _loading = true);
    try {
      await ref.read(dioProvider).post(
        running ? '/nav/nodes/disable' : '/nav/nodes/enable',
      );
    } on DioException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
          backgroundColor: Colors.red,
        ));
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final status = widget.statusAsync.valueOrNull;
    final running = status?.navNodesRunning ?? false;

    return FilledButton.icon(
      onPressed: _loading ? null : () => _toggle(running),
      style: FilledButton.styleFrom(
        backgroundColor: running
            ? const Color(0xFF45C95A).withOpacity(0.9)
            : Colors.black87,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      ),
      icon: _loading
          ? const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
            )
          : Icon(
              running ? Icons.sensors_rounded : Icons.sensors_off_rounded,
              size: 16,
            ),
      label: Text(running ? 'Nav ON' : 'Nav'),
    );
  }
}

// ── Camera panel ──────────────────────────────────────────────────────────────

class _CameraPanel extends ConsumerStatefulWidget {
  const _CameraPanel();

  @override
  ConsumerState<_CameraPanel> createState() => _CameraPanelState();
}

class _CameraPanelState extends ConsumerState<_CameraPanel> {
  Uint8List? _latestFrame;

  void _showFullscreen(BuildContext context) {
    final topic = ref.read(selectedPreviewTopicProvider);
    if (topic == null) return;
    showDialog(
      context: context,
      builder: (_) => _FullscreenPreview(topic: topic),
    );
  }

  @override
  Widget build(BuildContext context) {
    final topicsAsync = ref.watch(imageTopicsProvider);
    final selectedTopic = ref.watch(selectedPreviewTopicProvider);
    final topics = topicsAsync.valueOrNull ?? [];
    final baseUrl = ref.watch(baseUrlProvider);
    final mapInfo = ref.watch(mapInfoProvider).valueOrNull;
    final planning = ref.watch(planningStreamProvider).valueOrNull;

    // Auto-select color topic on first load
    ref.listen<AsyncValue<List<String>>>(imageTopicsProvider, (_, next) {
      if (next.valueOrNull != null &&
          ref.read(selectedPreviewTopicProvider) == null) {
        const colorTopic = '/camera/camera/color/image_raw';
        if (next.value!.contains(colorTopic)) {
          ref.read(selectedPreviewTopicProvider.notifier).state = colorTopic;
        }
      }
    });

    if (selectedTopic != null) {
      ref.listen<AsyncValue<Uint8List>>(
        previewStreamProvider(selectedTopic),
        (_, next) {
          if (next case AsyncData(:final value)) {
            if (mounted) setState(() => _latestFrame = value);
          }
        },
      );
    }

    return Container(
      color: Colors.black,
      child: Stack(
        fit: StackFit.expand,
        children: [
          if (selectedTopic != null && _latestFrame != null)
            GestureDetector(
              onTap: () => _showFullscreen(context),
              child: Image.memory(_latestFrame!, fit: BoxFit.contain, gaplessPlayback: true),
            )
          else
            Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.videocam_off_outlined, color: Colors.white24, size: 32),
                  const SizedBox(height: 6),
                  Text(
                    selectedTopic == null ? 'Select a camera topic' : 'Waiting for stream…',
                    style: const TextStyle(color: Colors.white38, fontSize: 12),
                  ),
                ],
              ),
            ),
          // ── Map PiP ──────────────────────────────────────────────────
          if (mapInfo != null && planning != null &&
              planning.localized && baseUrl != null)
            Positioned(
              top: 8, left: 8,
              child: _MapPip(mapInfo: mapInfo, planning: planning, baseUrl: baseUrl),
            ),
          // ── Topic selector ───────────────────────────────────────────
          Positioned(
            top: 8, right: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.videocam_outlined, color: Colors.white70, size: 14),
                  const SizedBox(width: 6),
                  DropdownButton<String?>(
                    value: selectedTopic,
                    hint: const Text('Off', style: TextStyle(color: Colors.white54, fontSize: 12)),
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                    dropdownColor: Colors.black87,
                    underline: const SizedBox(),
                    isDense: true,
                    items: [
                      const DropdownMenuItem<String?>(
                        value: null,
                        child: Text('Off', style: TextStyle(color: Colors.white54, fontSize: 12)),
                      ),
                      ...topics.map((t) {
                        const labels = {
                          '/camera/camera/color/image_raw': 'color',
                          '/camera/camera/infra1/image_rect_raw': 'left',
                          '/camera/camera/infra2/image_rect_raw': 'right',
                          '/slam/depth': 'depth',
                        };
                        final label = labels[t] ?? t.split('/').last;
                        return DropdownMenuItem<String?>(
                          value: t,
                          child: Text(label),
                        );
                      }),
                    ],
                    onChanged: (v) {
                      ref.read(selectedPreviewTopicProvider.notifier).state = v;
                      if (v == null) setState(() => _latestFrame = null);
                    },
                  ),
                ],
              ),
            ),
          ),
          if (selectedTopic != null && _latestFrame != null)
            Positioned(
              bottom: 8, right: 8,
              child: GestureDetector(
                onTap: () => _showFullscreen(context),
                child: Container(
                  padding: const EdgeInsets.all(4),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: const Icon(Icons.fullscreen, color: Colors.white, size: 20),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _FullscreenPreview extends ConsumerStatefulWidget {
  final String topic;
  const _FullscreenPreview({required this.topic});

  @override
  ConsumerState<_FullscreenPreview> createState() => _FullscreenPreviewState();
}

class _FullscreenPreviewState extends ConsumerState<_FullscreenPreview> {
  Uint8List? _frame;

  @override
  Widget build(BuildContext context) {
    ref.listen<AsyncValue<Uint8List>>(
      previewStreamProvider(widget.topic),
      (_, next) {
        if (next case AsyncData(:final value)) {
          if (mounted) setState(() => _frame = value);
        }
      },
    );

    return Dialog(
      backgroundColor: Colors.black,
      insetPadding: const EdgeInsets.all(12),
      child: Stack(
        children: [
          Center(
            child: _frame != null
                ? Image.memory(_frame!, fit: BoxFit.contain, gaplessPlayback: true)
                : const CircularProgressIndicator(color: Colors.white54),
          ),
          Positioned(
            top: 8, right: 8,
            child: IconButton(
              icon: const Icon(Icons.close, color: Colors.white),
              onPressed: () => Navigator.pop(context),
            ),
          ),
        ],
      ),
    );
  }
}

class _MapPip extends StatelessWidget {
  final MapInfo mapInfo;
  final PlanningState planning;
  final String baseUrl;

  const _MapPip({required this.mapInfo, required this.planning, required this.baseUrl});

  @override
  Widget build(BuildContext context) {
    const pipSize = 120.0;
    return Container(
      width: pipSize,
      height: pipSize,
      decoration: BoxDecoration(
        border: Border.all(color: Colors.white30, width: 1),
        borderRadius: BorderRadius.circular(8),
        color: const Color(0xFF1A1A2E),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(7),
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.network(
              '$baseUrl${mapInfo.imageUrl}',
              fit: BoxFit.fill,
              gaplessPlayback: true,
              errorBuilder: (_, __, ___) => const SizedBox(),
            ),
            CustomPaint(
              painter: MapOverlayPainter(
                mapInfo: mapInfo,
                pose: planning.mapPose,
                globalPath: planning.globalPath,
                showGlobalPath: true,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Joystick panel ────────────────────────────────────────────────────────────

class _JoystickPanel extends ConsumerWidget {
  final void Function(double x, double y) onLeft;
  final void Function(double x, double y) onRight;
  final Future<void> Function() onStop;

  const _JoystickPanel({required this.onLeft, required this.onRight, required this.onStop});

  Future<void> _sendAction(WidgetRef ref, BuildContext context, String command) async {
    try {
      await ref.read(dioProvider).post('/action/command', data: {'command': command});
    } on DioException catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
          backgroundColor: Colors.red,
        ));
      }
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      color: const Color(0xFFF5F5F5),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          // ── Left joystick (Move) ────────────────────────────────────
          Expanded(
            child: Column(
              children: [
                const Text('Move', style: TextStyle(fontSize: 10, color: Colors.grey, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Expanded(child: _JoystickPad(onChange: onLeft)),
              ],
            ),
          ),
          const SizedBox(width: 12),
          // ── Center: STOP + Sit / Stand ───────────────────────────────
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _ActionButton(
                icon: Icons.airline_seat_recline_extra_rounded,
                label: 'Sit',
                onTap: () => _sendAction(ref, context, 'sit'),
              ),
              const SizedBox(height: 8),
              _EStopButton(onStop: onStop),
              const SizedBox(height: 8),
              _ActionButton(
                icon: Icons.directions_walk_rounded,
                label: 'Stand',
                onTap: () => _sendAction(ref, context, 'stand'),
              ),
            ],
          ),
          const SizedBox(width: 12),
          // ── Right joystick (Rotate) ─────────────────────────────────
          Expanded(
            child: Column(
              children: [
                const Text('Rotate', style: TextStyle(fontSize: 10, color: Colors.grey, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Expanded(child: _JoystickPad(onChange: onRight, axisOnly: Axis.horizontal)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ActionButton({required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 56,
        padding: const EdgeInsets.symmetric(vertical: 8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFFE0E0E0)),
          boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 3, offset: Offset(0, 1))],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 20, color: const Color(0xFF2B3A42)),
            const SizedBox(height: 3),
            Text(label, style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: Color(0xFF2B3A42))),
          ],
        ),
      ),
    );
  }
}

class _EStopButton extends StatefulWidget {
  final Future<void> Function() onStop;
  const _EStopButton({required this.onStop});

  @override
  State<_EStopButton> createState() => _EStopButtonState();
}

class _EStopButtonState extends State<_EStopButton> {
  bool _pressing = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _pressing = true),
      onTapUp: (_) async {
        setState(() => _pressing = false);
        await widget.onStop();
      },
      onTapCancel: () => setState(() => _pressing = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 80),
        width: 56,
        height: 56,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: _pressing ? const Color(0xFFB71C1C) : const Color(0xFFE53935),
          boxShadow: [
            BoxShadow(
              color: const Color(0xFFE53935).withOpacity(0.45),
              blurRadius: _pressing ? 4 : 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: const Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.pan_tool_rounded, color: Colors.white, size: 18),
            SizedBox(height: 2),
            Text('STOP', style: TextStyle(color: Colors.white, fontSize: 9, fontWeight: FontWeight.w800, letterSpacing: 0.5)),
          ],
        ),
      ),
    );
  }
}

class _JoystickPad extends StatefulWidget {
  final void Function(double x, double y) onChange;
  final Axis? axisOnly;

  const _JoystickPad({required this.onChange, this.axisOnly});

  @override
  State<_JoystickPad> createState() => _JoystickPadState();
}

class _JoystickPadState extends State<_JoystickPad> {
  Offset _thumb = Offset.zero;

  void _update(Offset local, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = min(size.width, size.height) / 2 * 0.85;
    var delta = local - center;
    if (widget.axisOnly == Axis.horizontal) delta = Offset(delta.dx, 0);
    if (widget.axisOnly == Axis.vertical) delta = Offset(0, delta.dy);
    if (delta.distance > radius) delta = delta / delta.distance * radius;
    final norm = radius > 0 ? delta / radius : Offset.zero;
    setState(() => _thumb = norm);
    widget.onChange(norm.dx, norm.dy);
  }

  void _reset() {
    setState(() => _thumb = Offset.zero);
    widget.onChange(0, 0);
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (ctx, constraints) {
      final size = Size(constraints.maxWidth, constraints.maxHeight);
      final radius = min(size.width, size.height) / 2;
      final padRadius = radius * 0.85;
      final thumbOffset = Offset(
        size.width / 2 + _thumb.dx * padRadius,
        size.height / 2 + _thumb.dy * padRadius,
      );

      return GestureDetector(
        onPanStart: (d) => _update(d.localPosition, size),
        onPanUpdate: (d) => _update(d.localPosition, size),
        onPanEnd: (_) => _reset(),
        onPanCancel: () => _reset(),
        child: CustomPaint(
          painter: _JoystickPainter(
            thumbOffset: thumbOffset,
            padRadius: padRadius,
            size: size,
            axisOnly: widget.axisOnly,
          ),
          size: Size(constraints.maxWidth, constraints.maxHeight),
        ),
      );
    });
  }
}

class _JoystickPainter extends CustomPainter {
  final Offset thumbOffset;
  final double padRadius;
  final Size size;
  final Axis? axisOnly;

  const _JoystickPainter({
    required this.thumbOffset,
    required this.padRadius,
    required this.size,
    this.axisOnly,
  });

  @override
  void paint(Canvas canvas, Size _) {
    final center = Offset(size.width / 2, size.height / 2);

    // Background circle
    canvas.drawCircle(
      center,
      padRadius,
      Paint()..color = const Color(0xFFE0E0E0),
    );

    // Cross-hair lines
    final linePaint = Paint()
      ..color = const Color(0xFFBDBDBD)
      ..strokeWidth = 1;
    canvas.drawLine(
      Offset(center.dx - padRadius, center.dy),
      Offset(center.dx + padRadius, center.dy),
      linePaint,
    );
    canvas.drawLine(
      Offset(center.dx, center.dy - padRadius),
      Offset(center.dx, center.dy + padRadius),
      linePaint,
    );

    // Thumb
    canvas.drawCircle(
      thumbOffset,
      padRadius * 0.3,
      Paint()..color = const Color(0xFF45C95A),
    );
  }

  @override
  bool shouldRepaint(_JoystickPainter old) =>
      old.thumbOffset != thumbOffset || old.padRadius != padRadius;
}
