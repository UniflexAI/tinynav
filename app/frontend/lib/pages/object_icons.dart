import 'package:flutter/material.dart';

/// Maps a detected object's COCO class_id (see
/// tinynav/core/models_trt.py:COCO_CLASS_NAMES) to a Material icon glyph and
/// color for rendering as a symbol on the planning views. Only a few classes
/// are distinguished explicitly; everything else falls back to a generic dot
/// so widening the backend's detection allowlist doesn't require a frontend
/// change to show *something* for the new class.
const int kCocoClassPerson = 0;
const Set<int> _kCocoClassVehicles = {1, 2, 3, 5, 7}; // bicycle/car/motorcycle/bus/truck
const Set<int> _kCocoClassAnimals = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}; // bird..giraffe
const int kCocoClassChair = 56;
const int kCocoClassPottedPlant = 58;
const int kCocoClassDiningTable = 60;

IconData objectIconFor(int classId) {
  if (classId == kCocoClassPerson) return Icons.person;
  if (_kCocoClassVehicles.contains(classId)) return Icons.directions_car;
  if (_kCocoClassAnimals.contains(classId)) return Icons.pets;
  if (classId == kCocoClassChair) return Icons.chair;
  if (classId == kCocoClassPottedPlant) return Icons.local_florist;
  if (classId == kCocoClassDiningTable) return Icons.table_restaurant;
  return Icons.circle;
}

Color objectColorFor(int classId) {
  if (classId == kCocoClassPerson) return Colors.redAccent;
  if (_kCocoClassVehicles.contains(classId)) return Colors.blueAccent;
  if (_kCocoClassAnimals.contains(classId)) return Colors.orangeAccent;
  if (classId == kCocoClassPottedPlant) return Colors.greenAccent;
  if (classId == kCocoClassChair || classId == kCocoClassDiningTable) return Colors.brown;
  return Colors.grey;
}
