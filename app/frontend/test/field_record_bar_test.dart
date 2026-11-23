import 'dart:convert';
import 'dart:typed_data';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tinynav_app/core/providers.dart';
import 'package:tinynav_app/widgets/field_record_bar.dart';

class RecorderAdapter implements HttpClientAdapter {
  bool recording = false, saved = false, failStart = false, empty = false;
  final calls = <String>[];
  @override
  Future<ResponseBody> fetch(RequestOptions o, Stream<Uint8List>? requestStream, Future<void>? cancelFuture) async {
    calls.add(o.path);
    if (o.path.endsWith('/start')) {
      if (failStart) return ResponseBody.fromString('{"detail":"Disk full"}',503,headers:{Headers.contentTypeHeader:['application/json']});
      recording=true;
    }
    if (o.path.endsWith('/stop')) { recording=false; saved=true; }
    return ResponseBody.fromString(jsonEncode({
      'recording':recording,'path':'/tinynav_data/debug/field_test','seconds':recording?3:null,
      'verified':saved&&!empty,'message_count':saved?(empty?0:42):null,
      'error':saved&&empty?'No messages recorded':null,
    }),200,headers:{Headers.contentTypeHeader:['application/json']});
  }
  @override
  void close({bool force=false}) {}
}

Future<void> mount(WidgetTester t,RecorderAdapter a,{bool simulated=false}) async {
  final dio=Dio(BaseOptions(baseUrl:'http://recorder.test'))..httpClientAdapter=a;
  await t.pumpWidget(ProviderScope(overrides:[dioProvider.overrideWithValue(dio)],
    child:MaterialApp(home:Scaffold(body:FieldRecordBar(simulated:simulated)))));
  await t.pumpAndSettle();
}

void main() {
  testWidgets('starts, shows elapsed recording, stops and exposes verified bag path',(t) async {
    final a=RecorderAdapter();await mount(t,a);
    await t.tap(find.text('Record bag'));await t.pumpAndSettle();
    expect(find.text('REC 00:03'),findsOneWidget);
    await t.tap(find.text('Stop & save'));await t.pumpAndSettle();
    expect(find.text('Saved · 42 messages'),findsOneWidget);
    await t.tap(find.byTooltip('Recording files'));await t.pumpAndSettle();
    expect(find.text('/tinynav_data/debug/field_test'),findsOneWidget);
    expect(find.text('Copy path'),findsOneWidget);
    expect(a.calls.where((s)=>s.endsWith('/start')).length,1);
    expect(a.calls.where((s)=>s.endsWith('/stop')).length,1);
    await t.pumpWidget(const SizedBox());
  });
  testWidgets('keeps recording failure visible and never claims success',(t) async {
    final a=RecorderAdapter()..failStart=true;await mount(t,a);
    await t.tap(find.text('Record bag'));await t.pumpAndSettle();
    expect(find.text('Disk full'),findsOneWidget);
    expect(find.text('Stop & save'),findsNothing);
    await t.pumpWidget(const SizedBox());
  });
  testWidgets('an empty bag does not show saved success',(t) async {
    final a=RecorderAdapter()..empty=true;await mount(t,a);
    await t.tap(find.text('Record bag'));await t.pumpAndSettle();
    await t.tap(find.text('Stop & save'));await t.pumpAndSettle();
    expect(find.text('No messages recorded'),findsOneWidget);
    expect(find.textContaining('Saved ·'),findsNothing);
    await t.pumpWidget(const SizedBox());
  });
  testWidgets('simulation does not start a fake field bag',(t) async {
    final a=RecorderAdapter();await mount(t,a,simulated:true);
    expect(find.text('Simulation traces are saved automatically'),findsOneWidget);
    expect(t.widget<FilledButton>(find.byKey(const Key('field-record-toggle'))).onPressed,isNull);
    expect(a.calls,isEmpty);
    await t.pumpWidget(const SizedBox());
  });
}
