import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/models.dart';
import '../core/providers.dart';

class MapPreviewPage extends ConsumerWidget {
  final String mapName;
  const MapPreviewPage({super.key, required this.mapName});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final infoAsync = ref.watch(mapFileInfoProvider(mapName));
    final baseUrl = ref.watch(baseUrlProvider) ?? '';

    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF16213E),
        foregroundColor: Colors.white,
        elevation: 0,
        title: Text(
          mapName,
          style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
          overflow: TextOverflow.ellipsis,
        ),
      ),
      body: infoAsync.when(
        data: (info) => _MapViewer(info: info, baseUrl: baseUrl),
        loading: () => const Center(
          child: CircularProgressIndicator(color: Colors.white54),
        ),
        error: (e, _) => Center(
          child: Text('Failed to load map:\n$e',
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red)),
        ),
      ),
    );
  }
}

// ── Map viewer ─────────────────────────────────────────────────────────────────

class _MapViewer extends StatelessWidget {
  final MapFileInfo info;
  final String baseUrl;

  const _MapViewer({required this.info, required this.baseUrl});

  Offset _worldToPixel(double wx, double wy) {
    final px = (wx - info.originX) / info.resolution;
    final py = (info.height - 1) - (wy - info.originY) / info.resolution;
    return Offset(px, py);
  }

  @override
  Widget build(BuildContext context) {
    final imageUrl = '$baseUrl${info.imageUrl}';

    return Stack(
      children: [
        InteractiveViewer(
          minScale: 0.3,
          maxScale: 8.0,
          boundaryMargin: const EdgeInsets.all(80),
          child: Center(
            child: SizedBox(
              width: info.width.toDouble(),
              height: info.height.toDouble(),
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  // ── Occupancy map ──────────────────────────────────────────
                  Image.network(
                    imageUrl,
                    width: info.width.toDouble(),
                    height: info.height.toDouble(),
                    fit: BoxFit.fill,
                    loadingBuilder: (_, child, progress) => progress == null
                        ? child
                        : Container(
                            color: const Color(0xFF2A2A3E),
                            child: const Center(
                              child: CircularProgressIndicator(
                                  color: Colors.white54, strokeWidth: 2),
                            ),
                          ),
                    errorBuilder: (_, __, ___) => Container(
                      color: const Color(0xFF2A2A3E),
                      child: const Center(
                        child: Icon(Icons.broken_image_outlined,
                            color: Colors.white38, size: 48),
                      ),
                    ),
                  ),
                  // ── POI markers ────────────────────────────────────────────
                  ...info.pois.map((poi) {
                    final px = _worldToPixel(poi.x, poi.y);
                    return Positioned(
                      left: px.dx - 6,
                      top: px.dy - 6,
                      child: _PoiMarker(label: poi.name),
                    );
                  }),
                ],
              ),
            ),
          ),
        ),
        // ── Info bar ──────────────────────────────────────────────────────────
        Positioned(
          bottom: 16,
          left: 16,
          right: 16,
          child: _InfoBar(info: info),
        ),
      ],
    );
  }
}

// ── POI marker ─────────────────────────────────────────────────────────────────

class _PoiMarker extends StatelessWidget {
  final String label;
  const _PoiMarker({required this.label});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(
            color: const Color(0xFF4A90D9),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white, width: 1.5),
            boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 3)],
          ),
        ),
        const SizedBox(height: 2),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.65),
            borderRadius: BorderRadius.circular(4),
          ),
          child: Text(
            label,
            style: const TextStyle(
                color: Colors.white, fontSize: 8, fontWeight: FontWeight.w600),
          ),
        ),
      ],
    );
  }
}

// ── Info bar ──────────────────────────────────────────────────────────────────

class _InfoBar extends StatelessWidget {
  final MapFileInfo info;
  const _InfoBar({required this.info});

  @override
  Widget build(BuildContext context) {
    final sizeM = '${(info.width * info.resolution).toStringAsFixed(1)} × '
        '${(info.height * info.resolution).toStringAsFixed(1)} m';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.7),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          const Icon(Icons.map_outlined, color: Colors.white54, size: 14),
          const SizedBox(width: 8),
          Text(
            'Size: $sizeM  •  ${info.resolution * 100} cm/px  •  ${info.pois.length} POI',
            style: const TextStyle(
                color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }
}
