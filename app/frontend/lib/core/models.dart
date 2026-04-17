class DeviceStatus {
  final bool online;
  final double? battery;
  final String bagStatus;
  final bool bagFileReady;
  final String mapStatus;
  final double mappingPercent;
  final String navStatus;
  final String rawState;

  const DeviceStatus({
    required this.online,
    this.battery,
    required this.bagStatus,
    required this.bagFileReady,
    required this.mapStatus,
    required this.mappingPercent,
    required this.navStatus,
    required this.rawState,
  });

  factory DeviceStatus.fromJson(Map<String, dynamic> json) => DeviceStatus(
        online: json['online'] as bool? ?? false,
        battery: (json['battery'] as num?)?.toDouble(),
        bagStatus: json['bagStatus'] as String? ?? 'idle',
        bagFileReady: json['bagFileReady'] as bool? ?? false,
        mapStatus: json['mapStatus'] as String? ?? 'idle',
        mappingPercent: (json['mappingPercent'] as num?)?.toDouble() ?? 0.0,
        navStatus: json['navStatus'] as String? ?? 'idle',
        rawState: json['rawState'] as String? ?? 'unknown',
      );
}

class Pose {
  final double x;
  final double y;
  final double yaw;
  final double? timestamp;

  const Pose({required this.x, required this.y, required this.yaw, this.timestamp});

  factory Pose.fromJson(Map<String, dynamic> json) => Pose(
        x: (json['x'] as num).toDouble(),
        y: (json['y'] as num).toDouble(),
        yaw: (json['yaw'] as num).toDouble(),
        timestamp: (json['timestamp'] as num?)?.toDouble(),
      );
}

class MapInfo {
  final String imageUrl;
  final double originX;
  final double originY;
  final double resolution;
  final int width;
  final int height;

  const MapInfo({
    required this.imageUrl,
    required this.originX,
    required this.originY,
    required this.resolution,
    required this.width,
    required this.height,
  });

  factory MapInfo.fromJson(Map<String, dynamic> json) => MapInfo(
        imageUrl: json['imageUrl'] as String,
        originX: (json['origin_x'] as num).toDouble(),
        originY: (json['origin_y'] as num).toDouble(),
        resolution: (json['resolution'] as num).toDouble(),
        width: json['width'] as int,
        height: json['height'] as int,
      );
}

class Poi {
  final int id;
  final String name;
  final double x;
  final double y;
  final double z;

  const Poi({
    required this.id,
    required this.name,
    required this.x,
    required this.y,
    required this.z,
  });

  factory Poi.fromJson(Map<String, dynamic> json) {
    final pos = json['position'] as List;
    return Poi(
      id: json['id'] as int,
      name: json['name'] as String,
      x: (pos[0] as num).toDouble(),
      y: (pos[1] as num).toDouble(),
      z: (pos[2] as num).toDouble(),
    );
  }
}
