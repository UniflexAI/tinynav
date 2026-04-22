import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'models.dart';

final sharedPreferencesProvider = Provider<SharedPreferences>(
  (ref) => throw UnimplementedError('Override in main.dart'),
);

final deviceIpProvider = StateProvider<String?>((ref) => null);

final baseUrlProvider = Provider<String?>((ref) {
  final ip = ref.watch(deviceIpProvider);
  return ip != null ? 'http://$ip:8000' : null;
});

final dioProvider = Provider<Dio>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  return Dio(BaseOptions(
    baseUrl: baseUrl ?? '',
    connectTimeout: const Duration(seconds: 5),
    receiveTimeout: const Duration(seconds: 10),
  ));
});

/// Streams DeviceStatus from WS /ws/status (~1 s interval pushed by backend).
final deviceStatusProvider = StreamProvider<DeviceStatus>((ref) {
  final ip = ref.watch(deviceIpProvider);
  if (ip == null) return const Stream.empty();

  final channel = WebSocketChannel.connect(Uri.parse('ws://$ip:8000/ws/status'));
  ref.onDispose(() => channel.sink.close());

  return channel.stream.map(
    (data) => DeviceStatus.fromJson(jsonDecode(data as String) as Map<String, dynamic>),
  );
});

/// Streams robot Pose from WS /ws/pose (pushed on every odometry message).
final poseStreamProvider = StreamProvider<Pose>((ref) {
  final ip = ref.watch(deviceIpProvider);
  if (ip == null) return const Stream.empty();

  final channel = WebSocketChannel.connect(Uri.parse('ws://$ip:8000/ws/pose'));
  ref.onDispose(() => channel.sink.close());

  return channel.stream.map(
    (data) => Pose.fromJson(jsonDecode(data as String) as Map<String, dynamic>),
  );
});

/// One-shot fetch of map metadata from GET /map/current.
/// Returns null if no map has been built yet (404).
final mapInfoProvider = FutureProvider.autoDispose<MapInfo?>((ref) async {
  final dio = ref.watch(dioProvider);
  final baseUrl = ref.watch(baseUrlProvider);
  if (baseUrl == null) return null;
  try {
    final resp = await dio.get('/map/current');
    return MapInfo.fromJson(resp.data as Map<String, dynamic>);
  } on DioException catch (e) {
    if (e.response?.statusCode == 404 || e.response?.statusCode == 503) return null;
    rethrow;
  }
});

/// Sensor mode: 'looper' | 'realsense' | 'unknown'
final sensorModeProvider = FutureProvider.autoDispose<String>((ref) async {
  final dio = ref.watch(dioProvider);
  final baseUrl = ref.watch(baseUrlProvider);
  if (baseUrl == null) return 'unknown';
  try {
    final resp = await dio.get('/sensor/mode');
    return (resp.data['mode'] as String?) ?? 'unknown';
  } catch (_) {
    return 'unknown';
  }
});

/// Available image topics from the backend.
final imageTopicsProvider = FutureProvider.autoDispose<List<String>>((ref) async {
  final dio = ref.watch(dioProvider);
  final baseUrl = ref.watch(baseUrlProvider);
  if (baseUrl == null) return [];
  try {
    final resp = await dio.get('/sensor/image-topics');
    return (resp.data['topics'] as List).cast<String>();
  } catch (_) {
    return [];
  }
});

/// Currently selected preview topic (null = preview closed).
final selectedPreviewTopicProvider = StateProvider<String?>((ref) => null);

/// Streams raw JPEG bytes from WS /ws/preview?topic=<topic>.
final previewStreamProvider =
    StreamProvider.family.autoDispose<Uint8List, String>((ref, topic) {
  final ip = ref.watch(deviceIpProvider);
  if (ip == null) return const Stream.empty();

  final encoded = Uri.encodeQueryComponent(topic);
  final channel =
      WebSocketChannel.connect(Uri.parse('ws://$ip:8000/ws/preview?topic=$encoded'));
  ref.onDispose(() => channel.sink.close());

  return channel.stream.map((data) {
    if (data is Uint8List) return data;
    if (data is List<int>) return Uint8List.fromList(data);
    return Uint8List(0);
  }).where((b) => b.isNotEmpty);
});

/// Streams PlanningState from WS /ws/planning at ~5 fps.
final planningStreamProvider = StreamProvider<PlanningState>((ref) {
  final ip = ref.watch(deviceIpProvider);
  if (ip == null) return const Stream.empty();

  final channel = WebSocketChannel.connect(Uri.parse('ws://$ip:8000/ws/planning'));
  ref.onDispose(() => channel.sink.close());

  return channel.stream.map(
    (data) => PlanningState.fromJson(jsonDecode(data as String) as Map<String, dynamic>),
  );
});

/// One-shot system info from GET /device/sysinfo. autoDispose → re-fetches on each page enter.
final sysInfoProvider = FutureProvider.autoDispose<SysInfo>((ref) async {
  final dio = ref.watch(dioProvider);
  if (ref.watch(baseUrlProvider) == null) throw Exception('No device connected');
  final resp = await dio.get('/device/sysinfo');
  return SysInfo.fromJson(resp.data as Map<String, dynamic>);
});

/// File lists from /files/bags and /files/maps.
final bagFilesProvider = FutureProvider.autoDispose<List<FileEntry>>((ref) async {
  final dio = ref.watch(dioProvider);
  if (ref.watch(baseUrlProvider) == null) return [];
  try {
    final resp = await dio.get('/files/bags');
    return (resp.data['files'] as List)
        .map((j) => FileEntry.fromJson(j as Map<String, dynamic>))
        .toList();
  } catch (_) {
    return [];
  }
});

final mapFilesProvider = FutureProvider.autoDispose<List<FileEntry>>((ref) async {
  final dio = ref.watch(dioProvider);
  if (ref.watch(baseUrlProvider) == null) return [];
  try {
    final resp = await dio.get('/files/maps');
    return (resp.data['files'] as List)
        .map((j) => FileEntry.fromJson(j as Map<String, dynamic>))
        .toList();
  } catch (_) {
    return [];
  }
});

/// One-shot fetch of POI list from GET /map/pois.
final poisProvider = FutureProvider.autoDispose<List<Poi>>((ref) async {
  final dio = ref.watch(dioProvider);
  final baseUrl = ref.watch(baseUrlProvider);
  if (baseUrl == null) return [];
  try {
    final resp = await dio.get('/map/pois');
    final list = (resp.data['pois'] as List).cast<Map<String, dynamic>>();
    return list.map(Poi.fromJson).toList();
  } on DioException catch (e) {
    if (e.response?.statusCode == 503) return [];
    rethrow;
  }
});
