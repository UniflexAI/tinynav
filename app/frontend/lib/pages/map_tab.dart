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

    return Stack(
      children: [
        RefreshIndicator(
          onRefresh: () async {
            ref.invalidate(mapInfoProvider);
            ref.invalidate(poisProvider);
          },
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              mapAsync.when(
                data: (mapInfo) => mapInfo == null
                    ? _LocalPlanningView(planning: planning)
                    : _MapView(
                        mapInfo: mapInfo,
                        imageUrl: '${baseUrl!}${mapInfo.imageUrl}',
                        pose: poseAsync.valueOrNull,
                        pois: poisAsync.valueOrNull ?? [],
                        planning: planning,
                      ),
                loading: () => const Card(
                  child: Padding(padding: EdgeInsets.all(48), child: Center(child: CircularProgressIndicator())),
                ),
                error: (e, _) => Card(
                  color: Colors.red.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text('$e', style: const TextStyle(color: Colors.red)),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              _PoiCard(poisAsync: poisAsync, pose: poseAsync.valueOrNull),
            ],
          ),
        ),
        if (planning != null)
          Positioned(
            top: 8,
            left: 12,
            child: _LocalizationChip(localized: planning.localized),
          ),
        const Positioned(
          right: 12,
          bottom: 16,
          child: _CameraPreviewPip(),
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
    return Card(
      clipBehavior: Clip.antiAlias,
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

            // Compute ESDF overlay rect in canvas coords when localized.
            Positioned? esdfOverlay;
            final p = planning;
            if (p != null &&
                p.localized &&
                p.esdfImage != null &&
                p.mapPose != null &&
                p.gridInfo != null) {
              final gi = p.gridInfo!;
              final mp = p.mapPose!;
              // canvas pixels per world meter
              final pxPerMeter = canvasW / (mapInfo.width * mapInfo.resolution);
              final gridW_m = gi.width * gi.resolution;
              final gridH_m = gi.height * gi.resolution;
              // Grid is centered on robot in map frame (approx — grid rolls with odom).
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

/// Phase 1 view: odom-centric local planning canvas shown when no global map.
class _LocalPlanningView extends StatelessWidget {
  final PlanningState? planning;
  const _LocalPlanningView({this.planning});

  @override
  Widget build(BuildContext context) {
    final p = planning;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: AspectRatio(
        aspectRatio: 1.0,
        child: InteractiveViewer(
          minScale: 0.5,
          maxScale: 8.0,
          boundaryMargin: const EdgeInsets.all(double.infinity),
          child: Stack(
            fit: StackFit.expand,
          children: [
            Container(color: const Color(0xFF0D1117)),
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
    );
  }
}

class _LocalizationChip extends StatelessWidget {
  final bool localized;
  const _LocalizationChip({required this.localized});

  @override
  Widget build(BuildContext context) {
    final color = localized ? Colors.greenAccent : Colors.redAccent;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color, width: 1.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(localized ? Icons.gps_fixed : Icons.gps_off, size: 14, color: color),
          const SizedBox(width: 4),
          Text(
            localized ? 'Localized' : 'Not Localized',
            style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

// ── Camera PiP ───────────────────────────────────────────────────────────────

class _CameraPreviewPip extends ConsumerStatefulWidget {
  const _CameraPreviewPip();

  @override
  ConsumerState<_CameraPreviewPip> createState() => _CameraPreviewPipState();
}

class _CameraPreviewPipState extends ConsumerState<_CameraPreviewPip> {
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

    // Listen to preview stream when a topic is selected.
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

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        // Topic selector row
        Container(
          decoration: BoxDecoration(
            color: Colors.black87,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.videocam_outlined, color: Colors.white70, size: 14),
              const SizedBox(width: 4),
              DropdownButton<String?>(
                value: selectedTopic,
                hint: const Text('Camera', style: TextStyle(color: Colors.white54, fontSize: 12)),
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
                      child: Text(label, style: const TextStyle(fontSize: 12)),
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
        // Preview frame
        if (selectedTopic != null)
          GestureDetector(
            onTap: _latestFrame != null ? () => _showFullscreen(context) : null,
            child: ClipRRect(
              borderRadius: const BorderRadius.vertical(bottom: Radius.circular(8)),
              child: SizedBox(
                width: 176,
                height: 132,
                child: _latestFrame != null
                    ? Stack(
                        fit: StackFit.expand,
                        children: [
                          Image.memory(
                            _latestFrame!,
                            fit: BoxFit.cover,
                            gaplessPlayback: true,
                          ),
                          Align(
                            alignment: Alignment.bottomRight,
                            child: Container(
                              margin: const EdgeInsets.all(4),
                              decoration: BoxDecoration(
                                color: Colors.black54,
                                borderRadius: BorderRadius.circular(4),
                              ),
                              padding: const EdgeInsets.all(2),
                              child: const Icon(Icons.fullscreen, color: Colors.white, size: 20),
                            ),
                          ),
                        ],
                      )
                    : Container(
                        color: Colors.black,
                        child: const Center(
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white54,
                          ),
                        ),
                      ),
              ),
            ),
          ),
      ],
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

// ── POI list ─────────────────────────────────────────────────────────────────

class _PoiCard extends ConsumerStatefulWidget {
  final AsyncValue<List<Poi>> poisAsync;
  final Pose? pose;

  const _PoiCard({required this.poisAsync, this.pose});

  @override
  ConsumerState<_PoiCard> createState() => _PoiCardState();
}

class _PoiCardState extends ConsumerState<_PoiCard> {
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
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            const Icon(Icons.place_outlined, size: 20),
            const SizedBox(width: 8),
            const Text('POIs', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const Spacer(),
            TextButton.icon(
              onPressed: _addPoi,
              icon: const Icon(Icons.add_location_alt_outlined, size: 18),
              label: const Text('Add at current pose'),
            ),
          ]),
          const Divider(height: 20),
          widget.poisAsync.when(
            data: (pois) => pois.isEmpty
                ? const Center(
                    child: Padding(
                      padding: EdgeInsets.all(16),
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
        ]),
      ),
    );
  }
}
