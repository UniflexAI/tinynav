import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/models.dart';
import '../core/providers.dart';

class NavTab extends ConsumerWidget {
  const NavTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statusAsync = ref.watch(deviceStatusProvider);
    final poisAsync = ref.watch(poisProvider);

    return RefreshIndicator(
      onRefresh: () async => ref.invalidate(poisProvider),
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          statusAsync.when(
            data: (s) => _NavStatusCard(status: s),
            loading: () => const Card(
              child: Padding(padding: EdgeInsets.all(32), child: Center(child: CircularProgressIndicator())),
            ),
            error: (e, _) => Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text('$e', style: const TextStyle(color: Colors.red)),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Row(children: [
                  Icon(Icons.place_outlined, size: 20),
                  SizedBox(width: 8),
                  Text('Select Destination',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                ]),
                const Divider(height: 20),
                poisAsync.when(
                  data: (pois) => pois.isEmpty
                      ? const Center(
                          child: Padding(
                            padding: EdgeInsets.all(20),
                            child: Text(
                              'No POIs available.\nAdd them in the Map tab first.',
                              style: TextStyle(color: Colors.grey),
                              textAlign: TextAlign.center,
                            ),
                          ),
                        )
                      : Column(
                          children: pois
                              .map((poi) => _PoiTile(poi: poi, statusAsync: statusAsync))
                              .toList(),
                        ),
                  loading: () => const Center(child: CircularProgressIndicator()),
                  error: (e, _) => Text('$e', style: const TextStyle(color: Colors.red)),
                ),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Nav status card ──────────────────────────────────────────────────────────

class _NavStatusCard extends ConsumerStatefulWidget {
  final DeviceStatus status;
  const _NavStatusCard({required this.status});

  @override
  ConsumerState<_NavStatusCard> createState() => _NavStatusCardState();
}

class _NavStatusCardState extends ConsumerState<_NavStatusCard> {
  bool _canceling = false;

  Future<void> _cancel() async {
    setState(() => _canceling = true);
    try {
      await ref.read(dioProvider).post('/nav/cancel');
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
      if (mounted) setState(() => _canceling = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.status;
    final isNavigating = s.rawState == 'navigation';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(children: [
          Icon(
            isNavigating ? Icons.navigation : Icons.navigation_outlined,
            color: isNavigating ? Colors.blue : Colors.grey,
            size: 28,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Text('Navigation',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              Text(
                isNavigating ? 'Navigating...' : s.navStatus,
                style: TextStyle(color: isNavigating ? Colors.blue : Colors.grey),
              ),
            ]),
          ),
          if (isNavigating)
            OutlinedButton(
              onPressed: _canceling ? null : _cancel,
              style: OutlinedButton.styleFrom(foregroundColor: Colors.red),
              child: _canceling
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Cancel'),
            ),
        ]),
      ),
    );
  }
}

// ── POI tile with Go button ──────────────────────────────────────────────────

class _PoiTile extends ConsumerStatefulWidget {
  final Poi poi;
  final AsyncValue<DeviceStatus> statusAsync;

  const _PoiTile({required this.poi, required this.statusAsync});

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
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.response?.data?['detail'] ?? e.message ?? 'Error'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final status = widget.statusAsync.valueOrNull;
    final canGo = status != null && status.online && status.rawState == 'idle';

    return ListTile(
      leading: const Icon(Icons.place, color: Colors.amber),
      title: Text(widget.poi.name),
      subtitle: Text(
        '(${widget.poi.x.toStringAsFixed(2)}, ${widget.poi.y.toStringAsFixed(2)})',
        style: const TextStyle(fontSize: 12),
      ),
      trailing: _loading
          ? const SizedBox(
              width: 32,
              height: 32,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          : FilledButton(
              onPressed: canGo ? _go : null,
              child: const Text('Go'),
            ),
      dense: true,
    );
  }
}
