import 'package:flutter/material.dart';
class SimulationView extends StatelessWidget {
  final String url;
  const SimulationView({super.key, required this.url});
  @override
  Widget build(BuildContext context) => SelectableText('Open the simulator in a browser: $url');
}
