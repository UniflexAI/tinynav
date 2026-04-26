import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/models.dart';
import '../core/providers.dart';
import 'map_painter.dart';
import 'map_preview_page.dart';
import 'planning_painter.dart';

class MapTab extends ConsumerWidget {
  const MapTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statusAsync = ref.watch(deviceStatusProvider);

    return RefreshIndicator(
      onRefresh: () async {
        ref.invalidate(bagFilesProvider);
        ref.invalidate(mapFilesProvider);
      },
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Bag recording ─────────────────────────────────────────────
          statusAsync.when(
            data: (s) => _BagRecordCard(status: s),
            loading: () => const _LoadingCard(),
            error: (e, _) => _ErrorCard('$e'),
          ),
          const SizedBox(height: 12),
          _BagFileListCard(
            onRefresh: () => ref.invalidate(bagFilesProvider),
          ),
          const SizedBox(height: 20),
          // ── Map building ──────────────────────────────────────────────
          statusAsync.when(
            data: (s) => _MapBuildCard(status: s),
            loading: () => const _LoadingCard(),
            error: (_, __) => const SizedBox.shrink(),
          ),
          const SizedBox(height: 12),
          _FileListCard(
            title: 'Map Files',
            icon: Icons.map_outlined,
            provider: mapFilesProvider,
            onRefresh: () => ref.invalidate(mapFilesProvider),
            onTapFile: (f) => Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) => MapPreviewPage(mapName: f.name),
              ),
            ),
          ),
          const SizedBox(height: 24),
        ],
      ),
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

class _LocalPlanningView extends StatefulWidget {
  final PlanningState? planning;
  const _LocalPlanningView({this.planning});

  @override
  State<_LocalPlanningView> createState() => _LocalPlanningViewState();
}

class _LocalPlanningViewState extends State<_LocalPlanningView> {
  bool _showObstacle   = true;
  bool _showEsdf       = false;
  bool _showTrajectory = false;

  @override
  Widget build(BuildContext context) {
    final p = widget.planning;
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
                  if (_showEsdf && p?.esdfImage != null)
                    Opacity(
                      opacity: 0.85,
                      child: Image.memory(p!.esdfImage!, fit: BoxFit.fill, gaplessPlayback: true),
                    ),
                  if (_showObstacle && p?.obstacleImage != null)
                    Opacity(
                      opacity: 0.45,
                      child: Image.memory(p!.obstacleImage!, fit: BoxFit.fill, gaplessPlayback: true),
                    ),
                  if (p != null)
                    CustomPaint(
                      painter: LocalPlanningPainter(
                        trajectory: _showTrajectory ? p.trajectory : const [],
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
        // ── Layer toggle (top-right) ────────────────────────────────────
        Positioned(
          top: 8,
          right: 8,
          child: _LayerTogglePanel(
            showObstacle:   _showObstacle,
            showEsdf:       _showEsdf,
            showTrajectory: _showTrajectory,
            onChanged: (obs, esdf, traj) => setState(() {
              _showObstacle   = obs;
              _showEsdf       = esdf;
              _showTrajectory = traj;
            }),
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
  final void Function(bool obs, bool esdf, bool traj) onChanged;

  const _LayerTogglePanel({
    required this.showObstacle,
    required this.showEsdf,
    required this.showTrajectory,
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
                _LayerRow('Obstacle',   widget.showObstacle,
                    (v) => widget.onChanged(v, widget.showEsdf, widget.showTrajectory)),
                _LayerRow('ESDF',       widget.showEsdf,
                    (v) => widget.onChanged(widget.showObstacle, v, widget.showTrajectory)),
                _LayerRow('Trajectory', widget.showTrajectory,
                    (v) => widget.onChanged(widget.showObstacle, widget.showEsdf, v)),
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
              activeColor: const Color(0xFFFF6B35),
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
          Icon(Icons.circle, size: 8, color: dotColor),
          const SizedBox(width: 6),
          Text(
            localized ? 'Localized' : 'Not localized',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

// ── Bag recording card ────────────────────────────────────────────────────────

class _BagRecordCard extends ConsumerStatefulWidget {
  final dynamic status;
  const _BagRecordCard({required this.status});

  @override
  ConsumerState<_BagRecordCard> createState() => _BagRecordCardState();
}

class _BagRecordCardState extends ConsumerState<_BagRecordCard> {
  bool _busy = false;

  Future<void> _call(String path) async {
    setState(() => _busy = true);
    try {
      await ref.read(dioProvider).post(path);
    } on DioException catch (e) {
      if (mounted) _snack(context, e.response?.data?['detail'] ?? e.message ?? 'Error');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.status;
    final isRecording = s.rawState == 'realsense_bag_record';
    final canStart = s.online && s.rawState == 'idle';
    final canStop = s.online && isRecording;

    return _SectionCard(
      icon: Icons.videocam_outlined,
      iconColor: Colors.red,
      title: 'Bag Recording',
      badge: isRecording ? 'REC' : null,
      badgeColor: Colors.red,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _InfoRow('Status', s.bagStatus),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: canStart && !_busy ? () => _call('/bag/start') : null,
                icon: const Icon(Icons.fiber_manual_record, size: 16),
                label: const Text('Start'),
                style: FilledButton.styleFrom(backgroundColor: Colors.red),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: canStop && !_busy ? () => _call('/bag/stop') : null,
                icon: const Icon(Icons.stop),
                label: const Text('Stop'),
              ),
            ),
          ]),
        ],
      ),
    );
  }
}

// ── Map build card ────────────────────────────────────────────────────────────

class _MapBuildCard extends ConsumerStatefulWidget {
  final dynamic status;
  const _MapBuildCard({required this.status});

  @override
  ConsumerState<_MapBuildCard> createState() => _MapBuildCardState();
}

class _MapBuildCardState extends ConsumerState<_MapBuildCard> {
  bool _busy = false;

  Future<void> _buildMap() async {
    setState(() => _busy = true);
    final selectedBag = ref.read(selectedBagProvider);
    try {
      await ref.read(dioProvider).post(
        '/map/build',
        data: selectedBag != null ? {'bag_name': selectedBag} : null,
      );
    } on DioException catch (e) {
      if (mounted) _snack(context, e.response?.data?['detail'] ?? e.message ?? 'Error');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.status;
    final isBuilding = s.rawState == 'rosbag_build_map';
    final selectedBag = ref.watch(selectedBagProvider);
    final canBuild = s.online && (s.bagFileReady || selectedBag != null) && s.rawState == 'idle';

    return _SectionCard(
      icon: Icons.construction_rounded,
      iconColor: const Color(0xFF4A90D9),
      title: 'Map Building',
      badge: isBuilding ? 'Building' : null,
      badgeColor: const Color(0xFF4A90D9),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _InfoRow('Status', s.mapStatus),
          if (selectedBag != null) ...[
            const SizedBox(height: 6),
            Row(children: [
              const Icon(Icons.folder_rounded, size: 13, color: Color(0xFFFFB300)),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  selectedBag,
                  style: const TextStyle(fontSize: 12, color: Color(0xFF4A90D9),
                      fontWeight: FontWeight.w500),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              GestureDetector(
                onTap: () => ref.read(selectedBagProvider.notifier).state = null,
                child: const Icon(Icons.close_rounded, size: 14, color: Colors.grey),
              ),
            ]),
          ],
          if (isBuilding) ...[
            const SizedBox(height: 8),
            LinearProgressIndicator(
              value: s.mappingPercent > 0 ? s.mappingPercent / 100 : null,
              backgroundColor: Colors.grey.shade200,
            ),
            const SizedBox(height: 4),
            Text('${s.mappingPercent.toStringAsFixed(1)}%',
                style: const TextStyle(fontSize: 12, color: Colors.grey)),
          ],
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: canBuild && !_busy ? _buildMap : null,
              icon: _busy
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.construction),
              label: const Text('Build Map'),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Bag file list card (with selection) ──────────────────────────────────────

class _BagFileListCard extends ConsumerWidget {
  final VoidCallback onRefresh;
  const _BagFileListCard({required this.onRefresh});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final filesAsync = ref.watch(bagFilesProvider);
    final selected = ref.watch(selectedBagProvider);

    return _SectionCard(
      icon: Icons.folder_outlined,
      iconColor: Colors.grey.shade600,
      title: 'Bag Files',
      trailing: IconButton(
        icon: const Icon(Icons.refresh_rounded, size: 18),
        onPressed: onRefresh,
        padding: EdgeInsets.zero,
        constraints: const BoxConstraints(),
        tooltip: 'Refresh',
      ),
      child: filesAsync.when(
        data: (files) => files.isEmpty
            ? const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Center(
                  child: Text('No bags', style: TextStyle(color: Colors.grey, fontSize: 13)),
                ),
              )
            : Column(
                children: files.map((f) {
                  final isSelected = selected == f.name;
                  return _BagFileRow(
                    file: f,
                    isSelected: isSelected,
                    onTap: () {
                      ref.read(selectedBagProvider.notifier).state =
                          isSelected ? null : f.name;
                    },
                  );
                }).toList(),
              ),
        loading: () => const Padding(
          padding: EdgeInsets.symmetric(vertical: 12),
          child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
        ),
        error: (e, _) => Text('$e', style: const TextStyle(color: Colors.red, fontSize: 12)),
      ),
    );
  }
}

class _BagFileRow extends StatelessWidget {
  final FileEntry file;
  final bool isSelected;
  final VoidCallback onTap;
  const _BagFileRow({required this.file, required this.isSelected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final dt = DateTime.fromMillisecondsSinceEpoch((file.mtime * 1000).toInt());
    final dateStr =
        '${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 6),
        decoration: isSelected
            ? BoxDecoration(
                color: const Color(0xFF4A90D9).withOpacity(0.08),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: const Color(0xFF4A90D9).withOpacity(0.4)),
              )
            : null,
        child: Row(
          children: [
            Icon(Icons.folder_rounded, size: 16,
                color: isSelected ? const Color(0xFF4A90D9) : const Color(0xFFFFB300)),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                file.name,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: isSelected ? const Color(0xFF4A90D9) : null,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const SizedBox(width: 8),
            Text('${file.sizeLabel}  $dateStr',
                style: const TextStyle(fontSize: 11, color: Color(0xFF9E9E9E))),
            const SizedBox(width: 4),
            Icon(
              isSelected ? Icons.check_circle_rounded : Icons.radio_button_unchecked_rounded,
              size: 16,
              color: isSelected ? const Color(0xFF4A90D9) : Colors.grey.shade400,
            ),
          ],
        ),
      ),
    );
  }
}

// ── File list card ────────────────────────────────────────────────────────────

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

// ── Generic file list card ────────────────────────────────────────────────────

class _FileListCard extends ConsumerWidget {
  final String title;
  final IconData icon;
  final ProviderListenable<AsyncValue<List<FileEntry>>> provider;
  final VoidCallback onRefresh;
  final void Function(FileEntry)? onTapFile;

  const _FileListCard({
    required this.title,
    required this.icon,
    required this.provider,
    required this.onRefresh,
    this.onTapFile,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final filesAsync = ref.watch(provider);

    return _SectionCard(
      icon: icon,
      iconColor: Colors.grey.shade600,
      title: title,
      trailing: IconButton(
        icon: const Icon(Icons.refresh_rounded, size: 18),
        onPressed: onRefresh,
        padding: EdgeInsets.zero,
        constraints: const BoxConstraints(),
        tooltip: 'Refresh',
      ),
      child: filesAsync.when(
        data: (files) => files.isEmpty
            ? const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Center(
                  child: Text('No files', style: TextStyle(color: Colors.grey, fontSize: 13)),
                ),
              )
            : Column(
                children: files
                    .map((f) => _FileRow(file: f, onTap: onTapFile != null ? () => onTapFile!(f) : null))
                    .toList(),
              ),
        loading: () => const Padding(
          padding: EdgeInsets.symmetric(vertical: 12),
          child: Center(child: CircularProgressIndicator(strokeWidth: 2)),
        ),
        error: (e, _) => Text('$e', style: const TextStyle(color: Colors.red, fontSize: 12)),
      ),
    );
  }
}

// ── POI floating button ───────────────────────────────────────────────────────

class _PoiFloatingButton extends ConsumerStatefulWidget {
  final AsyncValue<List<Poi>> poisAsync;
  final Pose? pose;

  const _PoiFloatingButton({required this.poisAsync, this.pose});

  @override
  ConsumerState<_PoiFloatingButton> createState() => _PoiFloatingButtonState();
}

class _PoiFloatingButtonState extends ConsumerState<_PoiFloatingButton> {
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

class _FileRow extends StatelessWidget {
  final FileEntry file;
  final VoidCallback? onTap;
  const _FileRow({required this.file, this.onTap});

  @override
  Widget build(BuildContext context) {
    final dt = DateTime.fromMillisecondsSinceEpoch((file.mtime * 1000).toInt());
    final dateStr =
        '${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';

    final row = Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Icon(
            file.isDir ? Icons.folder_rounded : Icons.insert_drive_file_outlined,
            size: 16,
            color: file.isDir ? const Color(0xFFFFB300) : Colors.grey.shade500,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              file.name,
              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            '${file.sizeLabel}  $dateStr',
            style: const TextStyle(fontSize: 11, color: Color(0xFF9E9E9E)),
          ),
          if (onTap != null) ...[
            const SizedBox(width: 4),
            const Icon(Icons.chevron_right_rounded, size: 16, color: Color(0xFFBDBDBD)),
          ],
        ],
      ),
    );

    if (onTap == null) return row;
    return InkWell(onTap: onTap, borderRadius: BorderRadius.circular(8), child: row);
  }
}

// ── Shared section card ───────────────────────────────────────────────────────

class _SectionCard extends StatelessWidget {
  final IconData icon;
  final Color? iconColor;
  final String title;
  final String? badge;
  final Color? badgeColor;
  final Widget? trailing;
  final Widget child;

  const _SectionCard({
    required this.icon,
    this.iconColor,
    required this.title,
    this.badge,
    this.badgeColor,
    this.trailing,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.grey.shade200),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(icon, size: 18, color: iconColor ?? const Color(0xFF2B3A42)),
              const SizedBox(width: 8),
              Text(title,
                  style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 14)),
              if (badge != null) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                  decoration: BoxDecoration(
                    color: (badgeColor ?? Colors.grey).withOpacity(0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(badge!,
                      style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          color: badgeColor ?? Colors.grey)),
                ),
              ],
              const Spacer(),
              if (trailing != null) trailing!,
            ]),
            const Divider(height: 20),
            child,
          ],
        ),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;
  const _InfoRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 13, color: Color(0xFF9E9E9E))),
          Text(value, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}

class _LoadingCard extends StatelessWidget {
  const _LoadingCard();

  @override
  Widget build(BuildContext context) {
    return const Card(
      child: Padding(
        padding: EdgeInsets.all(32),
        child: Center(child: CircularProgressIndicator()),
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  final String message;
  const _ErrorCard(this.message);

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.red.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(children: [
          const Icon(Icons.error_outline, color: Colors.red),
          const SizedBox(width: 8),
          Expanded(child: Text(message, style: const TextStyle(color: Colors.red))),
        ]),
      ),
    );
  }
}

void _snack(BuildContext context, String message) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(content: Text(message), backgroundColor: Colors.red),
  );
}
