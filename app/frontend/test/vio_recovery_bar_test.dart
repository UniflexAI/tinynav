import 'dart:convert';
import 'dart:typed_data';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tinynav_app/core/providers.dart';
import 'package:tinynav_app/widgets/vio_recovery_bar.dart';

class RecoveryAdapter implements HttpClientAdapter {
  final requests=<RequestOptions>[];
  bool stale=false;
  @override
  Future<ResponseBody> fetch(RequestOptions o,Stream<Uint8List>? s,Future<void>? c) async {
    requests.add(o);
    return ResponseBody.fromString(jsonEncode(stale?{'detail':'Fault changed'}:{'ok':true}),
      stale?409:200,headers:{Headers.contentTypeHeader:['application/json']});
  }
  @override
  void close({bool force=false}) {}
}

Future<void> mount(WidgetTester t,RecoveryAdapter adapter,{required bool ready}) async {
  final dio=Dio(BaseOptions(baseUrl:'http://vio.test'))..httpClientAdapter=adapter;
  await t.pumpWidget(ProviderScope(overrides:[dioProvider.overrideWithValue(dio)],
    child:MaterialApp(home:Scaffold(body:VioRecoveryBar(safety:{
      'blocked':true,'locked':true,'ready':ready,'fault_id':7,'reason':'meter_jump',
    })))));
  await t.pumpAndSettle();
}

void main() {
  testWidgets('recovery cannot be confirmed while inputs are unhealthy',(t) async {
    final a=RecoveryAdapter();await mount(t,a,ready:false);
    expect(t.widget<FilledButton>(find.byKey(const Key('vio-confirm'))).onPressed,isNull);
    expect(find.text('Waiting for stable inputs'),findsOneWidget);expect(a.requests,isEmpty);
  });
  testWidgets('requires explicit dialog confirmation and sends captured fault ID',(t) async {
    final a=RecoveryAdapter();await mount(t,a,ready:true);
    await t.tap(find.byKey(const Key('vio-confirm')));await t.pumpAndSettle();
    expect(a.requests,isEmpty);
    await t.tap(find.text('Keep stopped'));await t.pumpAndSettle();expect(a.requests,isEmpty);
    await t.tap(find.byKey(const Key('vio-confirm')));await t.pumpAndSettle();
    await t.tap(find.text('Confirm recovery'));await t.pumpAndSettle();
    expect(a.requests.single.path,'/nav/vio/confirm');expect(a.requests.single.data,{'fault_id':7});
  });
  testWidgets('shows a fault-change rejection instead of claiming resumed',(t) async {
    final a=RecoveryAdapter()..stale=true;await mount(t,a,ready:true);
    await t.tap(find.byKey(const Key('vio-confirm')));await t.pumpAndSettle();
    await t.tap(find.text('Confirm recovery'));await t.pumpAndSettle();
    expect(find.text('Fault changed'),findsOneWidget);
    expect(find.textContaining('VIO safety stop'),findsOneWidget);
  });
}
