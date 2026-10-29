import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tinynav_app/widgets/bag_check_dialog.dart';

class CheckAdapter implements HttpClientAdapter {
  bool cancelled = false;
  bool hold = false;
  int status = 200;
  RequestOptions? request;
  Map<String, dynamic> body = {
    'report': {
      'warnings': ['/camera/camera/infra1/image_rect_raw: data gap'],
      'duration_s': 60,
      'bytes': 1000000000,
      'mode': 'looper',
      'mode_reason': 'has /camera/camera/vio_image (Looper VIO)',
      'missing': ['/camera/camera/color/image_rect_raw/compressed'],
      'streams': [
        '/camera/camera/infra1/image_rect_raw',
        '/camera/camera/depth/image_rect_raw',
        '/camera/camera/vio_image',
        '/camera/camera/color/image_rect_raw/compressed',
      ],
      'pairing': {'paired': 646, 'base_count': 650, 'pct': 99.4},
      'color': {'slop_ms': 20, 'matched': 0, 'base_count': 1000, 'pct': 0},
      'tf_count': {'/tf_static': 2, '/tf': 0},
      'topics': {
        '/camera/camera/infra1/image_rect_raw': {
          'count': 1000,
          'hz': 20.0,
          'gaps': [
            {'duration_s': 9.5}
          ]
        },
        '/camera/camera/depth/image_rect_raw': {
          'count': 650,
          'hz': 12.0,
          'gaps': []
        },
        '/camera/camera/vio_image': {'count': 1000, 'hz': 20.0, 'gaps': []},
        '/camera/camera/color/image_rect_raw/compressed': {
          'count': 0,
          'gaps': [
            {'duration_s': 60, 'boundary': 'whole_bag'}
          ]
        },
      },
    },
    'text': 'Bag integrity passed; infrared data gap',
  };

  @override
  Future<ResponseBody> fetch(
    RequestOptions options,
    Stream<Uint8List>? requestStream,
    Future<void>? cancelFuture,
  ) async {
    request = options;
    if (hold) {
      final completion = Completer<ResponseBody>();
      cancelFuture?.then((_) {
        cancelled = true;
        completion.completeError(DioException(
            requestOptions: options, type: DioExceptionType.cancel));
      });
      return completion.future;
    }
    return ResponseBody.fromString(jsonEncode(body), status, headers: {
      Headers.contentTypeHeader: [Headers.jsonContentType]
    });
  }

  @override
  void close({bool force = false}) {}
}

Future<void> openDialog(WidgetTester tester, CheckAdapter adapter) async {
  final dio = Dio(BaseOptions(baseUrl: 'http://robot'))
    ..httpClientAdapter = adapter;
  await tester.pumpWidget(MaterialApp(
      home: Scaffold(
          body: Builder(
    builder: (context) => TextButton(
        onPressed: () => showBagCheckDialog(context, dio, 'bag_example'),
        child: const Text('Open check')),
  ))));
  await tester.tap(find.text('Open check'));
  await tester.pump();
}

void main() {
  testWidgets('shows sensor, pairing, colour and missing intervals',
      (tester) async {
    final adapter = CheckAdapter();
    await openDialog(tester, adapter);
    await tester.pumpAndSettle();
    expect(adapter.request?.path, '/files/bags/bag_example/check');
    expect(find.text('Needs attention'), findsOneWidget);
    expect(find.text('Sensor: Looper'), findsOneWidget);
    expect(find.text('Map building needs: Colour'), findsOneWidget);
    expect(find.text('Colour: missing'), findsOneWidget);
    expect(find.textContaining('646/650'), findsOneWidget);
    expect(find.textContaining('0/1000'), findsOneWidget);
    expect(find.text('Longest missing interval: 9.50 s'), findsOneWidget);
    expect(find.textContaining('USB incidents'), findsNothing);
  });

  testWidgets('a clean RealSense bag reads as healthy', (tester) async {
    final adapter = CheckAdapter()
      ..body = {
        'report': {
          'warnings': [],
          'duration_s': 44,
          'bytes': 2000000000,
          'mode': 'realsense',
          'mode_reason': 'has stereo infrared and IMU, no Looper VIO',
          'missing': [],
          'streams': ['/camera/camera/color/image_raw'],
          'pairing': {'paired': 1317, 'base_count': 1317, 'pct': 100},
          'color': {
            'slop_ms': 20,
            'matched': 1317,
            'base_count': 1317,
            'pct': 100
          },
          'tf_count': {'/tf_static': 0, '/tf': 44},
          'topics': {
            '/camera/camera/color/image_raw': {
              'count': 1317,
              'hz': 30.0,
              'gaps': []
            },
          },
        },
        'text': '',
      };
    await openDialog(tester, adapter);
    await tester.pumpAndSettle();
    expect(find.text('No bag anomalies found'), findsOneWidget);
    expect(find.text('Sensor: RealSense'), findsOneWidget);
    expect(find.text('Colour: 1317 messages, 30.0 Hz'), findsOneWidget);
    expect(find.text('No receive gap over 250 ms'), findsOneWidget);
    expect(find.textContaining('Map building needs'), findsNothing);
  });

  testWidgets('closing a running check cancels the request', (tester) async {
    final adapter = CheckAdapter()..hold = true;
    await openDialog(tester, adapter);
    await tester.pump(const Duration(milliseconds: 300));
    await tester.tap(find.text('Cancel'));
    await tester.pumpAndSettle();
    expect(adapter.cancelled, isTrue);
    expect(find.text('Check bag'), findsNothing);
  });

  testWidgets('reports a backend busy error without false success',
      (tester) async {
    final adapter = CheckAdapter()
      ..status = 409
      ..body = {'detail': 'Another bag check is running.'};
    await openDialog(tester, adapter);
    await tester.pumpAndSettle();
    expect(find.text('Another bag check is running.'), findsOneWidget);
    expect(find.text('Retry'), findsOneWidget);
    expect(find.text('No bag anomalies found'), findsNothing);
  });
}
