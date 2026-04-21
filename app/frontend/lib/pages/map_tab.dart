import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/models.dart';
import '../core/providers.dart';
import 'map_painter.dart';
import 'planning_painter.dart';

class MapTab extends ConsumerWidget {
  const MapTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final mapAsync = ref.watch(mapInfoProvider);
    final poisAsync = ref.watch(poisProvider);
    final poseAsync = ref.watch(poseStreamProvider);
    final planningAsync = ref.watch(planningStreamProvider);
    final baseUrl = ref.watch(baseUrlProvider);
    final planning = planningAsync.valueOrNull;

    return Column(
      children: [
        // ── Camera panel: 1/3 ─────────────────────────────────────────────
        const Expanded(flex: 1, child: _CameraPanel()),
        const Divider(height: 1, thickness: 1, color: Color(0xFFE0E0E0)),
        // ── Map: 2/3 ──────────────────────────────────────────────────────
        Expanded(
          flex: 2,
          child: Stack(
            children: [
              Positioned.fill(
                child: mapAsync.when(
                  data: (mapInfo) => mapInfo == null
                      ? _LocalPlanningView(planning: planning)
                      : _MapView(
                          mapInfo: mapInfo,
                          imageUrl: '${baseUrl!}${mapInfo.imageUrl}',
                          pose: poseAsync.valueOrNull,
                          pois: poisAsync.valueOrNull ?? [],
                          planning: planning,
                        ),
                  loading: () => const Center(child: CircularProgressIndicator()),
                  error: (e, _) => Center(
                    child: Text('$e', style: const TextStyle(color: Colors.red)),
                  ),
                ),
              ),
              if (planning != null)
                Positioned(
                  top: 8,
                  left: 8,
                  child: _LocalizationChip(localized: planning.localized),
                ),
              Positioned(
                bottom: 12,
                left: 12,
                child: _PoiFloatingButton(
                  poisAsync: poisAsync,
                  pose: poseAsync.valueOrNull,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// ── Map image + overlay ──────────────────────────────────────────────────────

class _MapView extends StatelessWidget {
  final MapInfo mapInfo;
  final String imageUrl;
  final Pose? pose;
  final List<Poi> pois;
  final PlanningState? planning;

  const _MapView({
    required this.mapInfo,
    required this.imageUrl,
    required this.pois,
    this.pose,
    this.planning,
  });

  @override
  Widget build(BuildContext context) {
    final aspect = mapInfo.width / mapInfo.height;
    return Center(
      child: AspectRatio(
        aspectRatio: aspect > 0 ? aspect : 1.0,
        child: InteractiveViewer(
      minScale: 0.5,
      maxScale: 8.0,
      boundaryMargin: const EdgeInsets.all(double.infinity),
      child: LayoutBuilder(
        builder: (ctx, constraints) {
          final canvasW = constraints.maxWidth;
          final canvasH = constraints.maxHeight;

          Positioned? esdfOverlay;
          final p = planning;
          if (p != null &&
              p.localized &&
              p.esdfImage != null &&
              p.mapPose != null &&
              p.gridInfo != null) {
            final gi = p.gridInfo!;
            final mp = p.mapPose!;
            final pxPerMeter = canvasW / (mapInfo.width * mapInfo.resolution);
            final gridW_m = gi.width * gi.resolution;
            final gridH_m = gi.height * gi.resolution;
            final left = (mp.x - gridW_m / 2 - mapInfo.originX) * pxPerMeter;
            final top = canvasH -
                (mp.y - gridH_m / 2 - mapInfo.originY) * pxPerMeter -
                gridH_m * pxPerMeter;
            final width = gridW_m * pxPerMeter;
            final height = gridH_m * pxPerMeter;

            esdfOverlay = Positioned(
              left: left,
              top: top,
              width: width,
              height: height,
              child: Opacity(
                opacity: 0.5,
                child: Image.memory(
                  p.esdfImage!,
                  fit: BoxFit.fill,
                  gaplessPlayback: true,
                ),
              ),
            );
          }

          return Stack(
            fit: StackFit.expand,
            children: [
              Image.network(
                imageUrl,
                fit: BoxFit.fill,
                loadingBuilder: (ctx2, child, progress) => progress == null
                    ? child
                    : Center(
                        child: CircularProgressIndicator(
                          value: progress.expectedTotalBytes != null
                              ? progress.cumulativeBytesLoaded /
                                  progress.expectedTotalBytes!
                              : null,
                        ),
                      ),
                errorBuilder: (_, e, __) => Center(
                  child: Text('Image error: $e',
                      style: const TextStyle(color: Colors.red)),
                ),
              ),
              if (esdfOverlay != null) esdfOverlay,
              CustomPaint(
                painter: MapOverlayPainter(
                  mapInfo: mapInfo,
                  pose: pose,
                  pois: pois,
                ),
              ),
            ],
          );
        },
      ),
        ),
      ),
    );
  }
}

// ── Local planning view (Phase 1, no global map) ─────────────────────────────

class _LocalPlanningView extends StatelessWidget {
  final PlanningState? planning;
  const _LocalPlanningView({this.planning});

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
          if (p?.esdfImage != null)
            Opacity(
              opacity: 0.85,
              child: Image.memory(p!.esdfImage!, fit: BoxFit.fill, gaplessPlayback: true),
            ),
          if (p?.obstacleImage != null)
            Opacity(
              opacity: 0.45,
              child: Image.memory(p!.obstacleImage!, fit: BoxFit.fill, gaplessPlayback: true),
            ),
          if (p != null)
            CustomPaint(
              painter: LocalPlanningPainter(
                trajectory: p.trajectory,
                gridInfo: p.gridInfo,
                odomPose: p.odomPose,
              ),
            )
          else
            const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.map_outlined, size: 52, color: Colors.white24),
                  SizedBox(height: 8),
                  Text('Waiting for planning data…',
                      style: TextStyle(color: Colors.white38, fontSize: 13)),
                  SizedBox(height: 4),
                  Text('Build a map or start navigation to see the map here',
                      style: TextStyle(color: Colors.white24, fontSize: 11)),
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
            width: 7,
            height: 7,
            decoration: BoxDecoration(shape: BoxShape.circle, color: dotColor),
          ),
          const SizedBox(width: 6),
          Text(
            localized ? 'Localized' : 'Not Localized',
            style: const TextStyle(
                color: Colors.white, fontSize: 12, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }
}

// ── POI floating button + bottom sheet ──────────────────────────────────────

class _PoiFloatingButton extends ConsumerWidget {
  final AsyncValue<List<Poi>> poisAsync;
  final Pose? pose;
  const _PoiFloatingButton({required this.poisAsync, this.pose});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final count = poisAsync.valueOrNull?.length ?? 0;
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
        backgroundColor: Colors.black87,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      ),
      icon: const Icon(Icons.place_outlined, size: 18),
      label: Text('POIs${count > 0 ? ' ($count)' : ''}'),
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
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
            backgroundColor: Colors.red,
          ),
        );
      }
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
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final poisAsync = ref.watch(poisProvider);
    return Padding(
      padding: EdgeInsets.fromLTRB(
          16, 12, 16, 24 + MediaQuery.of(context).viewInsets.bottom),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 36,
              height: 4,
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          Row(children: [
            const Icon(Icons.place_outlined, size: 20),
            const SizedBox(width: 8),
            const Text('POIs', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const Spacer(),
            TextButton.icon(
              onPressed: _addPoi,
              icon: const Icon(Icons.add_location_alt_outlined, size: 18),
              label: const Text('Add here'),
            ),
          ]),
          const Divider(height: 20),
          poisAsync.when(
            data: (pois) => pois.isEmpty
                ? const Padding(
                    padding: EdgeInsets.symmetric(vertical: 24),
                    child: Center(
                      child: Text('No POIs yet', style: TextStyle(color: Colors.grey)),
                    ),
                  )
                : Column(
                    children: pois
                        .map((poi) => ListTile(
                              leading: const Icon(Icons.place, color: Colors.amber),
                              title: Text(poi.name),
                              subtitle: Text(
                                '(${poi.x.toStringAsFixed(2)}, ${poi.y.toStringAsFixed(2)})',
                                style: const TextStyle(fontSize: 12),
                              ),
                              trailing: IconButton(
                                icon: const Icon(Icons.delete_outline, color: Colors.red),
                                onPressed: () => _deletePoi(poi),
                              ),
                              dense: true,
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

// ── Camera panel (top 1/3) ────────────────────────────────────────────────────

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
          // ── Video frame ────────────────────────────────────────────────
          if (selectedTopic != null && _latestFrame != null)
            GestureDetector(
              onTap: () => _showFullscreen(context),
              child: Image.memory(
                _latestFrame!,
                fit: BoxFit.contain,
                gaplessPlayback: true,
              ),
            )
          else
            Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.videocam_off_outlined,
                      color: Colors.white24, size: 36),
                  const SizedBox(height: 8),
                  Text(
                    selectedTopic == null ? 'Select a camera topic' : 'Waiting for stream…',
                    style: const TextStyle(color: Colors.white38, fontSize: 12),
                  ),
                ],
              ),
            ),
          // ── Topic selector (top-right) ──────────────────────────────────
          Positioned(
            top: 8,
            right: 8,
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
          // ── Fullscreen button (bottom-right) ───────────────────────────
          if (selectedTopic != null && _latestFrame != null)
            Positioned(
              bottom: 8,
              right: 8,
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

// ── Fullscreen preview dialog ─────────────────────────────────────────────────

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
            top: 8,
            right: 8,
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
