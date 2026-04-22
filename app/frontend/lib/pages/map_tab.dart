import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/models.dart';
import '../core/providers.dart';
import 'map_preview_page.dart';

class MapTab extends ConsumerWidget {
  const MapTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statusAsync = ref.watch(deviceStatusProvider);

    return RefreshIndicator(
      onRefresh: () async {
        ref.invalidate(deviceStatusProvider);
        ref.invalidate(bagFilesProvider);
        ref.invalidate(mapFilesProvider);
      },
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Bag recording ───────────────────────────────────────────────
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
          // ── Map building ────────────────────────────────────────────────
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
