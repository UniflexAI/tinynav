import 'dart:convert';

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
