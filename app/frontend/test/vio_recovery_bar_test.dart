import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tinynav_app/widgets/vio_recovery_bar.dart';

void main() {
  testWidgets('fault waits for automatic recovery without a confirmation button',(t) async {
    await t.pumpWidget(const MaterialApp(home:Scaffold(body:VioRecoveryBar(safety:{
      'blocked':true,'locked':true,'ready':false,'reason':'meter_jump',
    }))));
    expect(find.text('Automatic recovery · waiting for stable inputs'),findsOneWidget);
    expect(find.byType(FilledButton),findsNothing);
    expect(find.textContaining('Position estimate jumped'),findsOneWidget);
  });
  testWidgets('legacy ready state does not ask the operator to confirm',(t) async {
    await t.pumpWidget(const MaterialApp(home:Scaffold(body:VioRecoveryBar(safety:{
      'blocked':true,'locked':true,'ready':true,'reason':'vio_restarting',
    }))));
    expect(find.byType(FilledButton),findsNothing);
    expect(find.textContaining('VIO restarted'),findsOneWidget);
  });
  testWidgets('healthy state removes the recovery banner',(t) async {
    await t.pumpWidget(const MaterialApp(home:Scaffold(body:VioRecoveryBar(safety:{'blocked':false}))));
    expect(find.textContaining('Automatic recovery'),findsNothing);
  });
}
