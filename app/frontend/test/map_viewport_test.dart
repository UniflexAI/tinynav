import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tinynav_app/pages/map_viewport.dart';

void main() {
  test('large map fits phone without moving its center marker', () {
    const layout = MapViewport(
      imageSize: Size(1600, 900),
      viewportSize: Size(400, 700),
    );
    expect(layout.scale, 0.25);
    expect(layout.imageToScene(const Offset(800, 450)), const Offset(200, 350));
    expect(layout.sceneToImage(const Offset(200, 350)), const Offset(800, 450));
  });

  test('desktop and phone preserve image pixels through fit, zoom and pan', () {
    for (final viewport in [
      const Size(390, 700),
      const Size(1920, 1080),
      const Size(700, 390),
    ]) {
      for (final image in [
        const Size(1600, 900),
        const Size(900, 1600),
        const Size(100, 100),
      ]) {
        final layout = MapViewport(imageSize: image, viewportSize: viewport);
        for (final pixel in [
          Offset.zero,
          Offset(image.width / 2, image.height / 2),
          Offset(image.width - 1, image.height - 1),
        ]) {
          for (final zoom in [0.5, 1.0, 3.0]) {
            final transform = Matrix4.identity()
              ..translate(-120.0, 75.0)
              ..scale(zoom);
            final screen = MatrixUtils.transformPoint(transform, layout.imageToScene(pixel));
            final scene = MatrixUtils.transformPoint(Matrix4.inverted(transform), screen);
            final restored = layout.sceneToImage(scene);
            expect(restored.dx, closeTo(pixel.dx, 1e-8));
            expect(restored.dy, closeTo(pixel.dy, 1e-8));
          }
        }
      }
    }
  });
}
