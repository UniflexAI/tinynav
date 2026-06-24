import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/providers.dart';

const _floorProbTopic = '/segmentation/floor_prob';
const _floorOverlayTopic = '/segmentation/floor_overlay';

class SegmentationTab extends ConsumerWidget {
  const SegmentationTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final topics = ref.watch(imageTopicsProvider).valueOrNull ?? const <String>[];
    final quality = ref.watch(previewQualityProvider);
    final hasProb = topics.contains(_floorProbTopic);
    final hasOverlay = topics.contains(_floorOverlayTopic);

    return Container(
      color: const Color(0xFF0B1118),
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
          Row(
            children: [
              const Icon(Icons.layers_outlined, color: Color(0xFF7BD8FF), size: 18),
              const SizedBox(width: 8),
              const Expanded(
                child: Text(
                  'Segmentation',
                  style: TextStyle(
                    color: Color(0xFFE8F2FF),
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              DropdownButton<PreviewQuality>(
                value: quality,
                dropdownColor: const Color(0xFF111A24),
                style: const TextStyle(color: Color(0xFFE8F2FF), fontSize: 12),
                underline: const SizedBox(),
                isDense: true,
                items: PreviewQuality.values
                    .map((q) => DropdownMenuItem(value: q, child: Text(q.label)))
                    .toList(),
                onChanged: (q) {
                  if (q != null) {
                    ref.read(previewQualityProvider.notifier).state = q;
                  }
                },
              ),
            ],
          ),
          const SizedBox(height: 10),
          Expanded(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final horizontal = constraints.maxWidth >= 760;
                final panels = [
                  Expanded(
                    child: _SegmentationPreviewPanel(
                      title: 'Overlay',
                      topic: _floorOverlayTopic,
                      available: hasOverlay,
                      quality: quality,
                      fit: BoxFit.contain,
                    ),
                  ),
                  const SizedBox(width: 10, height: 10),
                  Expanded(
                    child: _SegmentationPreviewPanel(
                      title: 'Floor probability',
                      topic: _floorProbTopic,
                      available: hasProb,
                      quality: quality,
                      fit: BoxFit.contain,
                    ),
                  ),
                ];
                return horizontal
                    ? Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: panels)
                    : Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: panels);
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _SegmentationPreviewPanel extends ConsumerStatefulWidget {
  final String title;
  final String topic;
  final bool available;
  final PreviewQuality quality;
  final BoxFit fit;

  const _SegmentationPreviewPanel({
    required this.title,
    required this.topic,
    required this.available,
    required this.quality,
    required this.fit,
  });

  @override
  ConsumerState<_SegmentationPreviewPanel> createState() => _SegmentationPreviewPanelState();
}

class _SegmentationPreviewPanelState extends ConsumerState<_SegmentationPreviewPanel> {
  Uint8List? _latestFrame;

  @override
  Widget build(BuildContext context) {
    if (widget.available) {
      ref.listen<AsyncValue<Uint8List>>(
        previewStreamProvider((topic: widget.topic, quality: widget.quality)),
        (_, next) {
          if (next case AsyncData(:final value)) {
            if (mounted) setState(() => _latestFrame = value);
          }
        },
      );
    }

    return Container(
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF263647)),
      ),
      child: Stack(
        fit: StackFit.expand,
        children: [
          if (widget.available && _latestFrame != null)
            Image.memory(_latestFrame!, fit: widget.fit, gaplessPlayback: true)
          else
            Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    widget.available ? Icons.image_not_supported_outlined : Icons.link_off_rounded,
                    color: Colors.white24,
                    size: 32,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    widget.available ? 'Waiting for stream' : 'Topic not available',
                    style: const TextStyle(color: Colors.white38, fontSize: 12),
                  ),
                ],
              ),
            ),
          Positioned(
            left: 8,
            top: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                widget.title,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ),
          Positioned(
            left: 8,
            right: 8,
            bottom: 8,
            child: Text(
              widget.topic,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: Colors.white54, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }
}
