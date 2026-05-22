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
        estimatedRemainingS:
            (json['estimated_remaining_s'] as num?)?.toDouble() ?? -1.0,
      );
}

class SisoTracePoint {
  final double t;
  final double cmdVx;
  final double odomVx;
  final double xRel;
  final double yRel;
  final double zRel;
  final double yawRel;
  final double? x;
  final double? y;

  const SisoTracePoint({
    required this.t,
    required this.cmdVx,
    required this.odomVx,
    required this.xRel,
    this.yRel = 0.0,
    this.zRel = 0.0,
    this.yawRel = 0.0,
    this.x,
    this.y,
  });

  factory SisoTracePoint.fromJson(Map<String, dynamic> json) => SisoTracePoint(
        t: (json['t'] as num?)?.toDouble() ?? 0.0,
        cmdVx: (json['cmd_vx'] as num?)?.toDouble() ?? 0.0,
        odomVx: (json['odom_vx'] as num?)?.toDouble() ?? 0.0,
        xRel: (json['x_rel'] as num?)?.toDouble() ?? 0.0,
        yRel: (json['y_rel'] as num?)?.toDouble() ?? 0.0,
        zRel: (json['z_rel'] as num?)?.toDouble() ?? 0.0,
        yawRel: (json['yaw_rel'] as num?)?.toDouble() ?? 0.0,
        x: (json['x'] as num?)?.toDouble(),
        y: (json['y'] as num?)?.toDouble(),
      );
}

class BenchmarkStatus {
  final String state;
  final String mode;
  final bool running;
  final double progressM;
  final double totalM;
  final double percent;
  final int trajectorySamples;
  final List<SisoTracePoint> sisoTrace;
  final BenchmarkResult? result;

  const BenchmarkStatus({
    required this.state,
    required this.mode,
    required this.running,
    required this.progressM,
    required this.totalM,
    required this.percent,
    required this.trajectorySamples,
    this.sisoTrace = const [],
    this.result,
  });

  factory BenchmarkStatus.fromJson(Map<String, dynamic> json) =>
      BenchmarkStatus(
        state: json['state'] as String? ?? 'idle',
        mode: json['mode'] as String? ?? 'figure8',
        running: json['running'] as bool? ?? (json['state'] == 'running'),
        progressM: (json['progress_m'] as num?)?.toDouble() ?? 0.0,
        totalM: (json['total_m'] as num?)?.toDouble() ?? 0.0,
        percent: (json['percent'] as num?)?.toDouble() ?? 0.0,
        trajectorySamples: (json['trajectory_samples'] as num?)?.toInt() ?? 0,
        sisoTrace: (json['siso_trace'] as List? ?? [])
            .whereType<Map<String, dynamic>>()
            .map(SisoTracePoint.fromJson)
            .toList(),
        result: json['result'] is Map<String, dynamic>
            ? BenchmarkResult.fromJson(json['result'] as Map<String, dynamic>)
            : null,
      );
}

class BenchmarkResult {
  final String state;
  final double score;
  final double? rmseM;
  final double? meanErrorM;
  final double? maxErrorM;
  final double completionPercent;
  final int samples;
  final double? durationS;

  const BenchmarkResult({
    required this.state,
    required this.score,
    this.rmseM,
    this.meanErrorM,
    this.maxErrorM,
    required this.completionPercent,
    required this.samples,
    this.durationS,
  });

  factory BenchmarkResult.fromJson(Map<String, dynamic> json) =>
      BenchmarkResult(
        state: json['state'] as String? ?? 'completed',
        score: (json['score'] as num?)?.toDouble() ?? 0.0,
        rmseM: (json['rmse_m'] as num?)?.toDouble(),
        meanErrorM: (json['mean_error_m'] as num?)?.toDouble(),
        maxErrorM: (json['max_error_m'] as num?)?.toDouble(),
        completionPercent:
            (json['completion_percent'] as num?)?.toDouble() ?? 0.0,
        samples: (json['samples'] as num?)?.toInt() ?? 0,
        durationS: (json['duration_s'] as num?)?.toDouble(),
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
      );
}

class Pose {
  final double x;
  final double y;
  final double yaw;
  final double? z;
  final double? timestamp;

  const Pose(
      {required this.x,
      required this.y,
      required this.yaw,
      this.z,
      this.timestamp});

  factory Pose.fromJson(Map<String, dynamic> json) => Pose(
        x: (json['x'] as num).toDouble(),
        y: (json['y'] as num).toDouble(),
        yaw: (json['yaw'] as num).toDouble(),
        z: (json['z'] as num?)?.toDouble(),
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

class VoxelPoint {
  final double x;
  final double y;
  final double z;
  const VoxelPoint(this.x, this.y, this.z);
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
  final List<TrajPoint> centerline;
  final List<TrajPoint> globalPath;
  final List<TrajPoint> mapGlobalPath;
  final GridInfo? gridInfo;
  final TrajPoint? navTargetPose;
  final List<TrajPoint> footprint;
  final List<VoxelPoint> voxelPoints;

  const PlanningState({
    required this.localized,
    this.odomPose,
    this.odomPoseAtKf,
    this.mapPose,
    this.esdfImage,
    this.obstacleImage,
    required this.trajectory,
    this.centerline = const [],
    required this.globalPath,
    this.mapGlobalPath = const [],
    this.gridInfo,
    this.navTargetPose,
    this.footprint = const [],
    this.voxelPoints = const [],
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

    List<TrajPoint> parsePath(String key) => (j[key] as List? ?? []).map((p) {
          final m = p as Map<String, dynamic>;
          return TrajPoint(
              (m['x'] as num).toDouble(), (m['y'] as num).toDouble());
        }).toList();

    return PlanningState(
      localized: j['localized'] as bool? ?? false,
      odomPose: parsePose(j['odom_pose']),
      odomPoseAtKf: parsePose(j['odom_pose_at_kf']),
      mapPose: parsePose(j['map_pose']),
      esdfImage: decodeImg(j['esdf_image'] as String?),
      obstacleImage: decodeImg(j['obstacle_image'] as String?),
      trajectory: parsePath('trajectory'),
      centerline: parsePath('centerline'),
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
      footprint: (j['footprint'] as List? ?? []).map((p) {
        final m = p as Map<String, dynamic>;
        return TrajPoint(
            (m['x'] as num).toDouble(), (m['y'] as num).toDouble());
      }).toList(),
      voxelPoints: (j['voxel_points'] as List? ?? []).map((p) {
        final m = p as Map<String, dynamic>;
        return VoxelPoint(
          (m['x'] as num).toDouble(),
          (m['y'] as num).toDouble(),
          (m['z'] as num).toDouble(),
        );
      }).toList(),
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
