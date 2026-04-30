import 'dart:convert';
import 'dart:typed_data';

class NavProgress {
  final int poiIndex;
  final double percent;
  final double pathRemainingM;
  final double pathTotalM;
  final double estimatedRemainingS;

  const NavProgress({
    required this.poiIndex,
    required this.percent,
    required this.pathRemainingM,
    required this.pathTotalM,
    required this.estimatedRemainingS,
  });

  factory NavProgress.fromJson(Map<String, dynamic> json) => NavProgress(
        poiIndex: json['poi_index'] as int? ?? 0,
        percent: (json['percent'] as num?)?.toDouble() ?? 0.0,
        pathRemainingM: (json['path_remaining_m'] as num?)?.toDouble() ?? 0.0,
        pathTotalM: (json['path_total_m'] as num?)?.toDouble() ?? 0.0,
        estimatedRemainingS: (json['estimated_remaining_s'] as num?)?.toDouble() ?? -1.0,
      );
}

class DeviceStatus {
  final bool online;
  final double? battery;
  final String bagStatus;
  final bool bagFileReady;
  final String mapStatus;
  final double mappingPercent;
  final String navStatus;
  final String rawState;
  final bool navNodesRunning;
  final bool navPaused;
  final NavProgress? navProgress;

  const DeviceStatus({
    required this.online,
    this.battery,
    required this.bagStatus,
    required this.bagFileReady,
    required this.mapStatus,
    required this.mappingPercent,
    required this.navStatus,
    required this.rawState,
    required this.navNodesRunning,
    required this.navPaused,
    this.navProgress,
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
        navNodesRunning: json['navNodesRunning'] as bool? ?? false,
        navPaused: json['navPaused'] as bool? ?? false,
        navProgress: json['navProgress'] != null
            ? NavProgress.fromJson(json['navProgress'] as Map<String, dynamic>)
            : null,
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

class MapFileInfo {
  final String imageUrl;
  final double originX;
  final double originY;
  final double resolution;
  final int width;
  final int height;
  final List<Poi> pois;

  const MapFileInfo({
    required this.imageUrl,
    required this.originX,
    required this.originY,
    required this.resolution,
    required this.width,
    required this.height,
    required this.pois,
  });

  factory MapFileInfo.fromJson(Map<String, dynamic> json) => MapFileInfo(
        imageUrl: json['imageUrl'] as String,
        originX: (json['origin_x'] as num).toDouble(),
        originY: (json['origin_y'] as num).toDouble(),
        resolution: (json['resolution'] as num).toDouble(),
        width: json['width'] as int,
        height: json['height'] as int,
        pois: (json['pois'] as List)
            .map((p) => Poi.fromJson(p as Map<String, dynamic>))
            .toList(),
      );
}

class TrajPoint {
  final double x;
  final double y;
  const TrajPoint(this.x, this.y);
}

class GridInfo {
  final double originX;
  final double originY;
  final double resolution;
  final int width;
  final int height;

  const GridInfo({
    required this.originX,
    required this.originY,
    required this.resolution,
    required this.width,
    required this.height,
  });

  factory GridInfo.fromJson(Map<String, dynamic> j) => GridInfo(
        originX: (j['origin_x'] as num).toDouble(),
        originY: (j['origin_y'] as num).toDouble(),
        resolution: (j['resolution'] as num).toDouble(),
        width: j['width'] as int,
        height: j['height'] as int,
      );
}

class PlanningState {
  final bool localized;
  final Pose? odomPose;
  final Pose? odomPoseAtKf;
  final Pose? mapPose;
  final Uint8List? esdfImage;
  final Uint8List? obstacleImage;
  final List<TrajPoint> trajectory;
  final List<TrajPoint> globalPath;
  final List<TrajPoint> mapGlobalPath;
  final GridInfo? gridInfo;
  final TrajPoint? navTargetPose;

  const PlanningState({
    required this.localized,
    this.odomPose,
    this.odomPoseAtKf,
    this.mapPose,
    this.esdfImage,
    this.obstacleImage,
    required this.trajectory,
    required this.globalPath,
    this.mapGlobalPath = const [],
    this.gridInfo,
    this.navTargetPose,
  });

  factory PlanningState.fromJson(Map<String, dynamic> j) {
    Uint8List? decodeImg(String? b64) {
      if (b64 == null || b64.isEmpty) return null;
      return base64Decode(b64);
    }

    Pose? parsePose(Object? raw) {
      if (raw == null) return null;
      return Pose.fromJson(raw as Map<String, dynamic>);
    }

    List<TrajPoint> parsePath(String key) =>
        (j[key] as List? ?? []).map((p) {
          final m = p as Map<String, dynamic>;
          return TrajPoint((m['x'] as num).toDouble(), (m['y'] as num).toDouble());
        }).toList();

    return PlanningState(
      localized: j['localized'] as bool? ?? false,
      odomPose: parsePose(j['odom_pose']),
      odomPoseAtKf: parsePose(j['odom_pose_at_kf']),
      mapPose: parsePose(j['map_pose']),
      esdfImage: decodeImg(j['esdf_image'] as String?),
      obstacleImage: decodeImg(j['obstacle_image'] as String?),
      trajectory: parsePath('trajectory'),
      globalPath: parsePath('global_path'),
      mapGlobalPath: parsePath('map_global_path'),
      gridInfo: j['grid_info'] != null
          ? GridInfo.fromJson(j['grid_info'] as Map<String, dynamic>)
          : null,
      navTargetPose: j['nav_target_pose'] != null
          ? TrajPoint(
              (j['nav_target_pose']['x'] as num).toDouble(),
              (j['nav_target_pose']['y'] as num).toDouble(),
            )
          : null,
    );
  }
}

class SysInfo {
  final double cpuPercent;
  final double memPercent;
  final double memUsedGb;
  final double memTotalGb;
  final double diskPercent;
  final double diskUsedGb;
  final double diskTotalGb;
  final double? gpuPercent;

  const SysInfo({
    required this.cpuPercent,
    required this.memPercent,
    required this.memUsedGb,
    required this.memTotalGb,
    required this.diskPercent,
    required this.diskUsedGb,
    required this.diskTotalGb,
    this.gpuPercent,
  });

  factory SysInfo.fromJson(Map<String, dynamic> j) => SysInfo(
        cpuPercent: (j['cpu_percent'] as num).toDouble(),
        memPercent: (j['mem_percent'] as num).toDouble(),
        memUsedGb: (j['mem_used_gb'] as num).toDouble(),
        memTotalGb: (j['mem_total_gb'] as num).toDouble(),
        diskPercent: (j['disk_percent'] as num).toDouble(),
        diskUsedGb: (j['disk_used_gb'] as num).toDouble(),
        diskTotalGb: (j['disk_total_gb'] as num).toDouble(),
        gpuPercent: (j['gpu_percent'] as num?)?.toDouble(),
      );
}

class FileEntry {
  final String name;
  final int size;
  final double mtime;
  final bool isDir;

  const FileEntry({
    required this.name,
    required this.size,
    required this.mtime,
    required this.isDir,
  });

  factory FileEntry.fromJson(Map<String, dynamic> j) => FileEntry(
        name: j['name'] as String,
        size: (j['size'] as num).toInt(),
        mtime: (j['mtime'] as num).toDouble(),
        isDir: j['is_dir'] as bool? ?? false,
      );

  String get sizeLabel {
    if (size < 1024) return '${size}B';
    if (size < 1024 * 1024) return '${(size / 1024).toStringAsFixed(1)}KB';
    if (size < 1024 * 1024 * 1024) {
      return '${(size / (1024 * 1024)).toStringAsFixed(1)}MB';
    }
    return '${(size / (1024 * 1024 * 1024)).toStringAsFixed(2)}GB';
  }
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
