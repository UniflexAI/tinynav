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
      'warnings': ['front: data gap'],
      'duration_s': 60,
      'bytes': 1000000000,
      'tf_static_count': 2,
      'cameras': {
        'front': {
          'received_depth_pair_pct': 99.4,
          'paired': 646,
          'received_unique_depth': 650,
          'topics': ['/ir', '/depth', '/pose'],
        },
      },
      'topics': {
        '/ir': {
          'count': 1000,
          'gaps': [
            {'duration_s': 9.5}
          ]
        },
        '/depth': {'count': 650, 'gaps': []},
        '/pose': {'count': 1000, 'gaps': []},
      },
      'usb': {
        'status': 'available',
        'gap_correlations': [{}],
        'events': [{}]
      },
    },
    'text': 'Bag integrity passed; front data gap',
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
  testWidgets('shows received-depth coverage and missing intervals',
      (tester) async {
    final adapter = CheckAdapter();
    await openDialog(tester, adapter);
    await tester.pumpAndSettle();
    expect(adapter.request?.path, '/files/bags/bag_example/check');
    expect(find.text('Needs attention'), findsOneWidget);
    expect(find.text('Front camera'), findsOneWidget);
    expect(find.textContaining('646/650'), findsOneWidget);
    expect(find.text('Longest missing interval: 9.50 s'), findsOneWidget);
    expect(find.textContaining('USB incidents'), findsNothing);
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
