import 'dart:math' as math;

import 'package:flutter/material.dart';

/// The fitted image and its markers share this coordinate space before pan/zoom.
class MapViewport {
  final Size imageSize;
  final Size viewportSize;

  const MapViewport({required this.imageSize, required this.viewportSize});

  double get scale => math.min(
        viewportSize.width / imageSize.width,
        viewportSize.height / imageSize.height,
      );

  Offset get padding => Offset(
        (viewportSize.width - imageSize.width * scale) / 2,
        (viewportSize.height - imageSize.height * scale) / 2,
      );

  Offset imageToScene(Offset pixel) => pixel * scale + padding;

  Offset sceneToImage(Offset point) => (point - padding) / scale;
}
