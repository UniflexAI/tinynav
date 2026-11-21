// ignore_for_file: avoid_web_libraries_in_flutter
// Browser-only iframe; native builds use simulation_view_stub.dart.
// ignore: deprecated_member_use
import 'dart:html' as html;
import 'dart:ui_web' as ui_web;
import 'package:flutter/material.dart';

class SimulationView extends StatefulWidget {
  final String url;
  const SimulationView({super.key, required this.url});
  @override
  State<SimulationView> createState() => _SimulationViewState();
}
class _SimulationViewState extends State<SimulationView> {
  late final String _id;
  static int _nextId = 0;
  @override
  void initState() {
    super.initState();
    _id = 'tinynav-mujoco-${_nextId++}';
    ui_web.platformViewRegistry.registerViewFactory(_id, (int viewId) => html.IFrameElement()
      ..src = widget.url
      ..title = 'MuJoCo simulation · live BEV · front measured depth'
      ..style.border = '0'
      ..style.width = '100%'
      ..style.height = '100%');
  }
  @override
  Widget build(BuildContext context) => HtmlElementView(viewType: _id);
}
