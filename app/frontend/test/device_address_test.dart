import 'package:flutter_test/flutter_test.dart';
import 'package:tinynav_app/core/providers.dart';

void main() {
  test('bare device addresses retain port 8000', () {
    expect(deviceBaseUrl('192.168.1.100'), 'http://192.168.1.100:8000');
    expect(deviceWsUrl('192.168.1.100', '/ws/status'), 'ws://192.168.1.100:8000/ws/status');
  });
  test('simulation API port is shared by HTTP and WebSocket', () {
    expect(deviceBaseUrl('192.168.11.116:18081'), 'http://192.168.11.116:18081');
    expect(deviceWsUrl('192.168.11.116:18081', '/ws/status'), 'ws://192.168.11.116:18081/ws/status');
  });
}
