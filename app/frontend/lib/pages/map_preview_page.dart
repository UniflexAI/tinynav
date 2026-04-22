import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/models.dart';
import '../core/providers.dart';

class MapPreviewPage extends ConsumerStatefulWidget {
  final String mapName;
  const MapPreviewPage({super.key, required this.mapName});

  @override
  ConsumerState<MapPreviewPage> createState() => _MapPreviewPageState();
}

class _MapPreviewPageState extends ConsumerState<MapPreviewPage> {
  bool _setting = false;

  Future<void> _setAsNavMap() async {
    setState(() => _setting = true);
    try {
      await ref.read(dioProvider).post('/map/set-active/${widget.mapName}');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('${widget.mapName} set as navigation map'),
            backgroundColor: const Color(0xFF45C95A),
          ),
        );
      }
    } on DioException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _setting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final infoAsync = ref.watch(mapFileInfoProvider(widget.mapName));
    final baseUrl = ref.watch(baseUrlProvider) ?? '';

    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF16213E),
        foregroundColor: Colors.white,
        elevation: 0,
        title: Text(
          widget.mapName,
          style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
          overflow: TextOverflow.ellipsis,
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: FilledButton.icon(
              onPressed: _setting ? null : _setAsNavMap,
              style: FilledButton.styleFrom(
                backgroundColor: const Color(0xFF45C95A),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                minimumSize: Size.zero,
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
              icon: _setting
                  ? const SizedBox(
                      width: 14, height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.navigation_rounded, size: 16),
              label: const Text('Set as Nav Map', style: TextStyle(fontSize: 13)),
            ),
          ),
        ],
      ),
      body: infoAsync.when(
        data: (info) => _MapViewer(
          info: info,
          baseUrl: baseUrl,
          mapName: widget.mapName,
        ),
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

// Gesture detection must be OUTSIDE InteractiveViewer: on Flutter Web the
// InteractiveViewer's internal scale recognizer consumes child pointer events
// before child GestureDetectors can fire.  We hold a TransformationController
// and convert viewport coordinates to image-pixel coordinates manually.

class _MapViewer extends ConsumerStatefulWidget {
  final MapFileInfo info;
  final String baseUrl;
  final String mapName;

  const _MapViewer({
    required this.info,
    required this.baseUrl,
    required this.mapName,
  });

  @override
  ConsumerState<_MapViewer> createState() => _MapViewerState();
}

class _MapViewerState extends ConsumerState<_MapViewer> {
  final _txCtrl = TransformationController();
  Offset? _tapDownImagePixel;

  @override
  void dispose() {
    _txCtrl.dispose();
    super.dispose();
  }

  // The PNG is (Nx, Ny) with axis-0 = world-X, axis-1 = world-Y.
  // np.flipud flips axis-0, so: col = Y-index, row = (Nx-1 - X-index).
  Offset _worldToPixel(double wx, double wy) {
    final px = (wy - widget.info.originY) / widget.info.resolution;
    final py = (widget.info.height - 1) - (wx - widget.info.originX) / widget.info.resolution;
    return Offset(px, py);
  }

  Offset _pixelToWorld(Offset pixel) {
    final wx = widget.info.originX + (widget.info.height - 1 - pixel.dy) * widget.info.resolution;
    final wy = widget.info.originY + pixel.dx * widget.info.resolution;
    return Offset(wx, wy);
  }

  // Convert a position in the outer GestureDetector's local coordinates
  // (= InteractiveViewer viewport coords) to image-pixel coordinates.
  // InteractiveViewer applies _txCtrl.value to its child (Center(SizedBox)).
  // Inverse transform gives us child coords; subtract the Center offset.
  Offset _viewportToImagePixel(Offset viewportPos, Size viewportSize) {
    final inv = Matrix4.inverted(_txCtrl.value);
    final childPoint = MatrixUtils.transformPoint(inv, viewportPos);
    final cx = (viewportSize.width - widget.info.width) / 2;
    final cy = (viewportSize.height - widget.info.height) / 2;
    return childPoint - Offset(cx, cy);
  }

  Future<void> _addPoi(Offset imagePixel) async {
    final world = _pixelToWorld(imagePixel);
    final ctrl = TextEditingController();
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add POI'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              controller: ctrl,
              decoration: const InputDecoration(labelText: 'Name', hintText: 'e.g. Room A'),
              autofocus: true,
              textCapitalization: TextCapitalization.sentences,
            ),
            const SizedBox(height: 8),
            Text(
              '(${world.dx.toStringAsFixed(2)}, ${world.dy.toStringAsFixed(2)})',
              style: const TextStyle(fontSize: 12, color: Colors.grey),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Add')),
        ],
      ),
    );
    if (ok != true || ctrl.text.trim().isEmpty) return;
    try {
      await ref.read(dioProvider).post('/map/preview/${widget.mapName}/pois', data: {
        'name': ctrl.text.trim(),
        'position': [world.dx, world.dy, 0.0],
      });
      ref.invalidate(mapFileInfoProvider(widget.mapName));
    } on DioException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
          backgroundColor: Colors.red,
        ));
      }
    }
  }

  Future<void> _deletePoisNear(Offset imagePixel) async {
    // Find nearest POI within 25 px (image-pixel space).
    Poi? nearest;
    var nearestDist = 25.0;
    for (final poi in widget.info.pois) {
      final px = _worldToPixel(poi.x, poi.y);
      final d = (px - imagePixel).distance;
      if (d < nearestDist) { nearest = poi; nearestDist = d; }
    }
    if (nearest == null) return;

    final poi = nearest;
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
      await ref.read(dioProvider).delete('/map/preview/${widget.mapName}/pois/${poi.id}');
      ref.invalidate(mapFileInfoProvider(widget.mapName));
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
    final imageUrl = '${widget.baseUrl}${widget.info.imageUrl}';
    final imageW = widget.info.width.toDouble();
    final imageH = widget.info.height.toDouble();

    return Stack(
      children: [
        LayoutBuilder(
          builder: (_, constraints) {
            final vp = Size(constraints.maxWidth, constraints.maxHeight);
            return GestureDetector(
              behavior: HitTestBehavior.opaque,
              // Long-press → add POI
              onLongPressStart: (d) {
                final px = _viewportToImagePixel(d.localPosition, vp);
                if (px.dx >= 0 && px.dx < imageW && px.dy >= 0 && px.dy < imageH) {
                  _addPoi(px);
                }
              },
              // Tap → delete nearest POI (onTapDown records pos; onTap fires only for real taps)
              onTapDown: (d) {
                _tapDownImagePixel = _viewportToImagePixel(d.localPosition, vp);
              },
              onTap: () {
                final px = _tapDownImagePixel;
                _tapDownImagePixel = null;
                if (px != null) _deletePoisNear(px);
              },
              child: InteractiveViewer(
                transformationController: _txCtrl,
                minScale: 0.3,
                maxScale: 8.0,
                boundaryMargin: const EdgeInsets.all(80),
                child: Center(
                  child: SizedBox(
                    width: imageW,
                    height: imageH,
                    child: Stack(
                      clipBehavior: Clip.none,
                      children: [
                        // ── Occupancy map ─────────────────────────────────────
                        Image.network(
                          imageUrl,
                          width: imageW,
                          height: imageH,
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
                        // ── POI markers ───────────────────────────────────────
                        ...widget.info.pois.map((poi) {
                          final px = _worldToPixel(poi.x, poi.y);
                          return Positioned(
                            left: px.dx - 10,
                            top: px.dy - 10,
                            child: _PoiMarker(label: poi.name),
                          );
                        }),
                      ],
                    ),
                  ),
                ),
              ),
            );
          },
        ),
        // ── Info bar + hint ──────────────────────────────────────────────────
        Positioned(
          bottom: 16,
          left: 16,
          right: 16,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _InfoBar(info: widget.info),
              const SizedBox(height: 6),
              Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Text(
                    'Long-press to add POI  ·  Tap POI to delete',
                    style: TextStyle(color: Colors.white54, fontSize: 11),
                  ),
                ),
              ),
            ],
          ),
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
          width: 20,
          height: 20,
          decoration: BoxDecoration(
            color: const Color(0xFF4A90D9),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white, width: 1.5),
            boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 3)],
          ),
          child: const Icon(Icons.place, color: Colors.white, size: 12),
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
