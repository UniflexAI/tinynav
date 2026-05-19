import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:tinynav_app/main.dart';

void main() {
  testWidgets('TinyNav app smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const ProviderScope(child: TinyNavApp()));
    await tester.pump();

    expect(find.text('TinyNav'), findsOneWidget);
  });
}
