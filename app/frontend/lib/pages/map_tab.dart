import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/models.dart';
import '../core/providers.dart';
import 'map_painter.dart';

class MapTab extends ConsumerWidget {
  const MapTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final mapAsync = ref.watch(mapInfoProvider);
    final poisAsync = ref.watch(poisProvider);
    final poseAsync = ref.watch(poseStreamProvider);
    final baseUrl = ref.watch(baseUrlProvider);

    return RefreshIndicator(
      onRefresh: () async {
        ref.invalidate(mapInfoProvider);
        ref.invalidate(poisProvider);
      },
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Map image
          mapAsync.when(
            data: (mapInfo) => mapInfo == null
                ? const _NoMapCard()
                : _MapView(
                    mapInfo: mapInfo,
                    imageUrl: '${baseUrl!}${mapInfo.imageUrl}',
                    pose: poseAsync.valueOrNull,
                    pois: poisAsync.valueOrNull ?? [],
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
          // POI management
          _PoiCard(poisAsync: poisAsync, pose: poseAsync.valueOrNull),
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

  const _MapView({
    required this.mapInfo,
    required this.imageUrl,
    required this.pois,
    this.pose,
  });

  @override
  Widget build(BuildContext context) {
    final aspect = mapInfo.width / mapInfo.height;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: AspectRatio(
        aspectRatio: aspect > 0 ? aspect : 1.0,
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.network(
              imageUrl,
              fit: BoxFit.fill,
              loadingBuilder: (ctx, child, progress) => progress == null
                  ? child
                  : Center(
                      child: CircularProgressIndicator(
                        value: progress.expectedTotalBytes != null
                            ? progress.cumulativeBytesLoaded / progress.expectedTotalBytes!
                            : null,
                      ),
                    ),
              errorBuilder: (_, e, __) => Center(
                child: Text('Image error: $e', style: const TextStyle(color: Colors.red)),
              ),
            ),
            CustomPaint(
              painter: MapOverlayPainter(
                mapInfo: mapInfo,
                pose: pose,
                pois: pois,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NoMapCard extends StatelessWidget {
  const _NoMapCard();

  @override
  Widget build(BuildContext context) {
    return const Card(
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: 40, horizontal: 16),
        child: Column(children: [
          Icon(Icons.map_outlined, size: 52, color: Colors.grey),
          SizedBox(height: 8),
          Text('No map available', style: TextStyle(color: Colors.grey, fontSize: 16)),
          SizedBox(height: 4),
          Text('Build a map from the Device tab',
              style: TextStyle(color: Colors.grey, fontSize: 12)),
        ]),
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
