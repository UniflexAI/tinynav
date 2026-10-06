const REALTIME_TICK_DELAY_MS = 33;
const GRID_STEP_M = 1.0;
const MARKER_HIT_RADIUS = window.matchMedia?.("(pointer: coarse)").matches ? 22 : 14;
const DEFAULT_SCENE = { xMin: -0.8, xMax: 9.8, yMin: -5.4, yMax: 5.4 };
const LAB_VIEW_HALF_M = 5.0;
let sceneBounds = { ...DEFAULT_SCENE };
const DEPTH_STOPS = [
  [0.0, [255, 230, 80]],
  [0.18, [255, 120, 40]],
  [0.36, [236, 52, 98]],
  [0.55, [156, 64, 255]],
  [0.75, [56, 120, 255]],
  [1.0, [12, 28, 72]],
];

let config = null;
let selectedIndex = 0;
let dragState = null;
let realtimeRunning = false;
let realtimeBusy = false;
let realtimeTimer = null;
let realtimePending = false;
let realtimePath = [];
let currentFrame = null;
let mapCatalog = [];
let robotPresets = {};
let mapBackgroundImage = null;
let mapBackgroundSrc = null;

function simBox(name, center, size) {
  return { name, kind: "box", center, size };
}

const SCENARIOS = {
  l_turn: {
    label: "L turn",
    start: { xy: [0.0, 0.0], yaw_deg: 0.0 },
    target: [3.9, 4.4, 0.0],
    cameraMaxRange: 8.0,
    bounds: { xMin: -1.4, xMax: 5.6, yMin: -1.6, yMax: 5.7 },
    objects: [
      simBox("lower_horizontal_wall", [1.8, -0.85, 0.65], [5.6, 0.3, 1.3]),
      simBox("upper_horizontal_wall_before_turn", [1.15, 0.85, 0.65], [4.3, 0.3, 1.3]),
      simBox("inside_corner_block", [3.45, 0.85, 0.65], [0.3, 0.3, 1.3]),
      simBox("left_vertical_wall_after_turn", [3.15, 2.8, 0.65], [0.3, 3.6, 1.3]),
      simBox("right_vertical_wall", [4.85, 2.55, 0.65], [0.3, 5.1, 1.3]),
      simBox("entry_left_stub", [-1.05, 0.85, 0.65], [0.8, 0.3, 1.3]),
      simBox("entry_right_stub", [-1.05, -0.85, 0.65], [0.8, 0.3, 1.3]),
      simBox("far_end_cap", [4.0, 5.25, 0.65], [2.0, 0.3, 1.3]),
    ],
  },
  straight: {
    label: "Straight",
    start: { xy: [0.0, 0.0], yaw_deg: 0.0 },
    target: [5.2, 0.0, 0.0],
    cameraMaxRange: 8.0,
    bounds: { xMin: -0.8, xMax: 6.0, yMin: -1.8, yMax: 1.8 },
    objects: [
      simBox("left_wall", [2.55, 0.9, 0.65], [6.1, 0.25, 1.3]),
      simBox("right_wall", [2.55, -0.9, 0.65], [6.1, 0.25, 1.3]),
      simBox("far_cap", [5.8, 0.0, 0.65], [0.25, 2.0, 1.3]),
    ],
  },
  s_bend: {
    label: "S bend",
    start: { xy: [0.0, -0.7], yaw_deg: 0.0 },
    target: [5.2, 0.7, 0.0],
    cameraMaxRange: 8.0,
    bounds: { xMin: -0.8, xMax: 6.0, yMin: -2.2, yMax: 2.2 },
    objects: [
      simBox("lower_wall_entry", [1.25, -1.45, 0.65], [3.1, 0.22, 1.3]),
      simBox("upper_wall_entry", [1.15, 0.55, 0.65], [2.9, 0.22, 1.3]),
      simBox("lower_wall_exit", [4.05, -0.35, 0.65], [3.1, 0.22, 1.3]),
      simBox("upper_wall_exit", [4.05, 1.45, 0.65], [3.1, 0.22, 1.3]),
      simBox("left_deflector", [2.55, -0.95, 0.65], [0.22, 0.9, 1.3]),
      simBox("right_deflector", [3.25, 0.95, 0.65], [0.22, 0.9, 1.3]),
    ],
  },
  narrow_gate: {
    label: "Narrow gate",
    start: { xy: [0.0, 0.0], yaw_deg: 0.0 },
    target: [4.8, 0.0, 0.0],
    cameraMaxRange: 8.0,
    bounds: { xMin: -0.8, xMax: 5.6, yMin: -2.1, yMax: 2.1 },
    objects: [
      simBox("left_wall", [2.2, 1.05, 0.65], [5.6, 0.25, 1.3]),
      simBox("right_wall", [2.2, -1.05, 0.65], [5.6, 0.25, 1.3]),
      simBox("gate_left_block", [2.65, 0.75, 0.65], [0.45, 0.45, 1.3]),
      simBox("gate_right_block", [2.65, -0.75, 0.65], [0.45, 0.45, 1.3]),
      simBox("far_cap", [5.15, 0.0, 0.65], [0.25, 2.3, 1.3]),
    ],
  },
  open_target: {
    label: "Open target",
    start: { xy: [0.0, 0.0], yaw_deg: -45.0 },
    target: [4.2, 2.4, 0.0],
    cameraMaxRange: 8.0,
    bounds: { xMin: -1.0, xMax: 5.4, yMin: -2.4, yMax: 3.4 },
    objects: [
      simBox("near_column", [1.45, 0.65, 0.65], [0.45, 0.45, 1.3]),
      simBox("middle_column", [2.65, -0.65, 0.65], [0.5, 0.5, 1.3]),
      simBox("far_column", [3.35, 1.45, 0.65], [0.45, 0.45, 1.3]),
      simBox("side_shelf", [3.9, -1.55, 0.65], [1.2, 0.35, 1.3]),
    ],
  },
  back_target: {
    label: "Back target",
    start: { xy: [0.0, 0.0], yaw_deg: 0.0 },
    target: [-1.35, 0.0, 0.0],
    cameraMaxRange: 5.0,
    bounds: { xMin: -2.2, xMax: 1.6, yMin: -1.6, yMax: 1.6 },
    objects: [
      simBox("front_block", [0.95, 0.0, 0.65], [0.25, 2.0, 1.3]),
      simBox("left_boundary", [-0.35, 1.05, 0.65], [2.6, 0.22, 1.3]),
      simBox("right_boundary", [-0.35, -1.05, 0.65], [2.6, 0.22, 1.3]),
    ],
  },
};

const canvas = document.getElementById("sceneCanvas");
const ctx = canvas.getContext("2d");
const mapCanvas = document.getElementById("mapCanvas");
const mapCtx = mapCanvas.getContext("2d");
const depthCanvas = document.getElementById("depthCanvas");
const depthCtx = depthCanvas.getContext("2d");
const statusEl = document.getElementById("status");
const controlsPanel = document.getElementById("controlsPanel");
const controlsToggle = document.getElementById("controlsToggle");
const realtimeButton = document.getElementById("realtimeButton");
const $ = (id) => document.getElementById(id);

const fields = {
  mapSelect: $("mapSelect"),
  mapSummary: $("mapSummary"),
  startX: $("startX"),
  startY: $("startY"),
  startYaw: $("startYaw"),
  targetX: $("targetX"),
  targetY: $("targetY"),
  objectName: $("objectName"),
  objectX: $("objectX"),
  objectY: $("objectY"),
  objectSX: $("objectSX"),
  objectSY: $("objectSY"),
  objectSZ: $("objectSZ"),
  configText: $("configText"),
  robotPreset: $("robotPreset"),
  scenarioSelect: $("scenarioSelect"),
  robotSummary: $("robotSummary"),
  cameraSummary: $("cameraSummary"),
};

const boundInputs = () => [...document.querySelectorAll("[data-bind]")];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function pathGet(obj, path) {
  return path.split(".").reduce((cur, key) => (cur == null ? cur : cur[key]), obj);
}

function pathSet(obj, path, value) {
  const keys = path.split(".");
  const last = keys.pop();
  let cur = obj;
  keys.forEach((key) => {
    if (cur[key] == null || typeof cur[key] !== "object") cur[key] = {};
    cur = cur[key];
  });
  cur[last] = value;
}

function readBoundFields() {
  boundInputs().forEach((el) => {
    const path = el.dataset.bind;
    if (el.type === "number") {
      const v = Number(el.value);
      if (Number.isFinite(v)) pathSet(config, path, v);
    } else {
      pathSet(config, path, el.value);
    }
  });
}

function writeBoundFields() {
  boundInputs().forEach((el) => {
    const v = pathGet(config, el.dataset.bind);
    if (v != null) el.value = v;
  });
}

function lerpRGB(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function logicalSize(target = canvas) {
  return {
    width: Math.max(1, target.clientWidth || target.width),
    height: Math.max(1, target.clientHeight || target.height),
  };
}

function sceneView(target = canvas, bounds = sceneBounds) {
  const b = bounds;
  const { width, height } = logicalSize(target);
  const pad = 12;
  const availW = Math.max(1, width - pad * 2);
  const availH = Math.max(1, height - pad * 2);
  const scale = Math.min(availW / (b.yMax - b.yMin), availH / (b.xMax - b.xMin));
  return {
    b,
    scale,
    offsetX: pad + (availW - (b.yMax - b.yMin) * scale) / 2,
    offsetY: pad + (availH - (b.xMax - b.xMin) * scale) / 2,
  };
}

function worldToCanvas(x, y, target = canvas, bounds = sceneBounds) {
  const v = sceneView(target, bounds);
  return [v.offsetX + (y - v.b.yMin) * v.scale, v.offsetY + (v.b.xMax - x) * v.scale];
}

function canvasToWorld(px, py, target = canvas, bounds = sceneBounds) {
  const v = sceneView(target, bounds);
  return [v.b.xMax - (py - v.offsetY) / v.scale, v.b.yMin + (px - v.offsetX) / v.scale];
}

function pointerPosOn(event, targetCanvas = canvas) {
  const rect = targetCanvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

function syncSceneCanvasSize() {
  const { width: cssW, height: cssH } = logicalSize();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const nextW = Math.max(1, Math.round(cssW * dpr));
  const nextH = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== nextW || canvas.height !== nextH) {
    canvas.width = nextW;
    canvas.height = nextH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: cssW, height: cssH };
}

function selectedObject() {
  return config?.objects?.[selectedIndex] || null;
}

function objectCanvasRect(obj) {
  const [x, y] = obj.center;
  const [sx, sy] = obj.size;
  const [px0, py0] = worldToCanvas(x - sx / 2, y - sy / 2);
  const [px1, py1] = worldToCanvas(x + sx / 2, y + sy / 2);
  return [Math.min(px0, px1), Math.min(py0, py1), Math.abs(px1 - px0), Math.abs(py1 - py0)];
}

function cameraPose() {
  const yaw = (Number(config.start.yaw_deg || 0) * Math.PI) / 180;
  const df = Number(config.robot.camera_x || 0) - Number(config.robot.control_x || 0);
  const dl = Number(config.robot.camera_y || 0) - Number(config.robot.control_y || 0);
  const forward = [Math.cos(yaw), Math.sin(yaw)];
  const left = [-Math.sin(yaw), Math.cos(yaw)];
  return {
    xy: [
      config.start.xy[0] + forward[0] * df + left[0] * dl,
      config.start.xy[1] + forward[1] * df + left[1] * dl,
    ],
  };
}

function fovDeg(size, focal) {
  return (2 * Math.atan((Math.max(1, size) / 2) / Math.max(1e-6, focal)) * 180) / Math.PI;
}

function summaryLines(lines) {
  return lines.map(([k, v]) => `${k}: <code>${v}</code>`).join("<br />");
}

function robotSummaryHtml() {
  const r = config.robot || {};
  const [fl, rl, hw] = footprintFromControl(r);
  return summaryLines([["footprint F/R/W", `${fl.toFixed(2)} / ${rl.toFixed(2)} / ${hw.toFixed(2)} m`]]);
}

function footprintFromControl(r) {
  const hl = r.shape === "circle" ? Number(r.radius || 0) : Number(r.length || 0) / 2;
  const hw = r.shape === "circle" ? Number(r.radius || 0) : Number(r.width || 0) / 2;
  const cx = Number(r.control_x || 0);
  return [hl - cx, hl + cx, hw];
}

function cameraSummaryHtml() {
  const cam = config.camera || {};
  const w = Number(cam.width || 0);
  const h = Number(cam.image_height || cam.height || 0);
  const fx = Number(cam.fx || 0);
  const fy = Number(cam.fy || 0);
  return summaryLines([
    ["HFOV", `${fovDeg(w, fx).toFixed(1)}°`],
    ["VFOV", `${fovDeg(h, fy).toFixed(1)}°`],
    ["resolution", `${w} x ${h}`],
  ]);
}

function updateLabViewBounds(focusXY = null) {
  const xy = focusXY || currentFrame?.robot_xy || config?.start?.xy;
  if (!xy) return;
  sceneBounds = {
    xMin: xy[0] - LAB_VIEW_HALF_M,
    xMax: xy[0] + LAB_VIEW_HALF_M,
    yMin: xy[1] - LAB_VIEW_HALF_M,
    yMax: xy[1] + LAB_VIEW_HALF_M,
  };
}

function perceptionBounds() {
  const info = currentFrame?.map_info || config?.map_info;
  if (info?.bounds) {
    const b = info.bounds;
    const pad = 0.3;
    return {
      xMin: b.x_min - pad,
      xMax: b.x_max + pad,
      yMin: b.y_min - pad,
      yMax: b.y_max + pad,
    };
  }
  return { ...DEFAULT_SCENE };
}

function mapSummaryHtml() {
  const info = config?.map_info || currentFrame?.map_info;
  if (!info) return "No map loaded. Using synthetic demo scene.";
  const b = info.bounds;
  return summaryLines([
    ["name", config?.map_name || info.map_path?.split("/").pop() || "—"],
    ["size", `${info.width} x ${info.height} x ${info.depth} @ ${Number(info.resolution).toFixed(2)} m`],
    ["origin", `[${info.origin.map((v) => Number(v).toFixed(2)).join(", ")}]`],
    ["world XY", `${Number(b.x_min).toFixed(1)}..${Number(b.x_max).toFixed(1)}, ${Number(b.y_min).toFixed(1)}..${Number(b.y_max).toFixed(1)}`],
  ]);
}

function setMapBackground(bg) {
  const src = bg?.data_url || null;
  if (src === mapBackgroundSrc) return;
  mapBackgroundSrc = src;
  mapBackgroundImage = null;
  if (!src) return;
  const img = new Image();
  img.onload = () => {
    if (mapBackgroundSrc !== src) return;
    mapBackgroundImage = img;
    drawPerceptionMap();
  };
  img.src = src;
}

function drawMapBackgroundOn(targetCtx, targetCanvas, bounds) {
  const bg = currentFrame?.map_background || config?.map_background;
  const info = currentFrame?.map_info || config?.map_info;
  if (!bg?.data_url || !info?.bounds) return;
  setMapBackground(bg);
  if (!mapBackgroundImage) return;
  const b = info.bounds;
  // Image width = world-Y span, height = world-X span (see map_volume.background_rgb).
  const [topLeftX, topLeftY] = worldToCanvas(b.x_max, b.y_min, targetCanvas, bounds);
  const [botRightX, botRightY] = worldToCanvas(b.x_min, b.y_max, targetCanvas, bounds);
  const x = Math.min(topLeftX, botRightX);
  const y = Math.min(topLeftY, botRightY);
  const w = Math.abs(botRightX - topLeftX);
  const h = Math.abs(botRightY - topLeftY);
  targetCtx.drawImage(mapBackgroundImage, x, y, w, h);
}

function drawGrid() {
  const { width, height } = logicalSize();
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e0e7ee";
  ctx.lineWidth = 1;
  const b = sceneBounds;
  for (let x = Math.ceil(b.xMin / GRID_STEP_M) * GRID_STEP_M; x <= b.xMax + 1e-9; x += GRID_STEP_M) {
    const [px0, py] = worldToCanvas(x, b.yMin);
    const [px1] = worldToCanvas(x, b.yMax);
    ctx.beginPath();
    ctx.moveTo(px0, py);
    ctx.lineTo(px1, py);
    ctx.stroke();
  }
  for (let y = Math.ceil(b.yMin / GRID_STEP_M) * GRID_STEP_M; y <= b.yMax + 1e-9; y += GRID_STEP_M) {
    const [px, py0] = worldToCanvas(b.xMin, y);
    const [, py1] = worldToCanvas(b.xMax, y);
    ctx.beginPath();
    ctx.moveTo(px, py0);
    ctx.lineTo(px, py1);
    ctx.stroke();
  }
}

function drawGround() {
  if (config?.map_path || currentFrame?.map_info) return;
  const b = sceneBounds;
  const [px0, py0] = worldToCanvas(b.xMin + 0.3, b.yMin + 0.3);
  const [px1, py1] = worldToCanvas(b.xMax - 0.3, b.yMax - 0.3);
  ctx.fillStyle = "rgba(226, 232, 220, 0.85)";
  ctx.fillRect(Math.min(px0, px1), Math.min(py0, py1), Math.abs(px1 - px0), Math.abs(py1 - py0));
}

function drawObjects() {
  config.objects.forEach((obj, idx) => {
    const [x, y, w, h] = objectCanvasRect(obj);
    const selected = idx === selectedIndex;
    const fence = String(obj.name || "").startsWith("fence_");
    ctx.fillStyle = selected ? "rgba(16, 107, 103, 0.55)" : fence ? "rgba(120, 113, 108, 0.55)" : "rgba(95, 108, 120, 0.45)";
    ctx.strokeStyle = selected ? "#106b67" : fence ? "#57534e" : "#56616c";
    ctx.lineWidth = selected ? 3 : 1.5;
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#17202a";
    ctx.font = "13px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(obj.name, x + w / 2, y + h / 2 + 4);
  });
}

function drawMarker(xy, label, color, radius = 7, targetCtx = ctx, targetCanvas = canvas, bounds = sceneBounds) {
  const [px, py] = worldToCanvas(xy[0], xy[1], targetCanvas, bounds);
  targetCtx.fillStyle = color;
  targetCtx.beginPath();
  targetCtx.arc(px, py, radius, 0, Math.PI * 2);
  targetCtx.fill();
  if (label) targetCtx.fillText(label, px, py - 12);
}

function drawHeadingArrow(xy, yawDeg, color, length = 0.45, targetCtx = ctx, targetCanvas = canvas, bounds = sceneBounds) {
  if (!xy || yawDeg == null) return;
  const yaw = (Number(yawDeg) * Math.PI) / 180;
  const tip = [xy[0] + Math.cos(yaw) * length, xy[1] + Math.sin(yaw) * length];
  const [x0, y0] = worldToCanvas(xy[0], xy[1], targetCanvas, bounds);
  const [x1, y1] = worldToCanvas(tip[0], tip[1], targetCanvas, bounds);
  targetCtx.strokeStyle = targetCtx.fillStyle = color;
  targetCtx.lineWidth = 3;
  targetCtx.beginPath();
  targetCtx.moveTo(x0, y0);
  targetCtx.lineTo(x1, y1);
  targetCtx.stroke();
  const angle = Math.atan2(y1 - y0, x1 - x0);
  targetCtx.beginPath();
  targetCtx.moveTo(x1, y1);
  targetCtx.lineTo(x1 - Math.cos(angle - 0.55) * 12, y1 - Math.sin(angle - 0.55) * 12);
  targetCtx.lineTo(x1 - Math.cos(angle + 0.55) * 12, y1 - Math.sin(angle + 0.55) * 12);
  targetCtx.closePath();
  targetCtx.fill();
}

function drawStartTarget() {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  drawMarker(config.start.xy, "control", "#24a148");
  const camera = cameraPose();
  drawMarker(camera.xy, "camera", "#6d28d9", 6);
  drawHeadingArrow(camera.xy, config.start.yaw_deg, "#6d28d9", 0.32);
  drawMarker(config.target, "target", "#da1e28", realtimeRunning ? 5 : 7);
  ctx.restore();
}

function gridCellToCanvasRect(payload, ix, iy) {
  const r = Number(payload.resolution);
  const o = payload.origin;
  const [px0, py0] = worldToCanvas(o[0] + ix * r, o[1] + iy * r);
  const [px1, py1] = worldToCanvas(o[0] + (ix + 1) * r, o[1] + (iy + 1) * r);
  return [Math.min(px0, px1), Math.min(py0, py1), Math.abs(px1 - px0), Math.abs(py1 - py0)];
}

function esdfColor(value, obstacleValue) {
  if (obstacleValue > 0) return "rgba(239, 68, 68, 0.62)";
  const c = clamp(value / 255, 0, 1);
  if (c < 0.18) {
    const t = c / 0.18;
    return `rgba(${Math.round(249 - 80 * t)}, ${Math.round(115 + 80 * t)}, ${Math.round(22 + 20 * t)}, 0.46)`;
  }
  if (c < 0.45) {
    const t = (c - 0.18) / 0.27;
    return `rgba(${Math.round(169 - 80 * t)}, ${Math.round(195 + 35 * t)}, ${Math.round(42 + 85 * t)}, 0.28)`;
  }
  return "rgba(15, 22, 33, 0.10)";
}

function drawEsdfLegend() {
  const { width } = logicalSize();
  const boxW = 154;
  const x = Math.max(12, width - boxW - 16);
  const y = 16;
  ctx.save();
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.fillRect(x, y, boxW, 58);
  ctx.fillStyle = "#17202a";
  ctx.font = "12px system-ui";
  ctx.fillText("ESDF clearance", x + 12, y + 18);
  ["rgba(239, 68, 68, 0.72)", "rgba(249, 115, 22, 0.62)", "rgba(89, 230, 127, 0.36)", "rgba(15, 22, 33, 0.14)"]
    .forEach((color, i) => {
      ctx.fillStyle = color;
      ctx.fillRect(x + 12 + i * 30, y + 28, 30, 12);
    });
  ctx.fillStyle = "#5c6975";
  ctx.fillText("near", x + 12, y + 53);
  ctx.fillText("clear", x + 106, y + 53);
  ctx.restore();
}

function drawEsdfSceneOverlay() {
  const esdf = currentFrame?.esdf_u8;
  if (!esdf?.data) return;
  const obstacle = currentFrame?.obstacle_u8?.data || [];
  ctx.save();
  for (let ix = 0; ix < esdf.height; ix += 1) {
    for (let iy = 0; iy < esdf.width; iy += 1) {
      const idx = ix * esdf.width + iy;
      const value = esdf.data[idx];
      const occ = obstacle[idx] || 0;
      if (value > 150 && occ === 0) continue;
      const [x, y, w, h] = gridCellToCanvasRect(esdf, ix, iy);
      ctx.fillStyle = esdfColor(value, occ);
      ctx.fillRect(x, y, Math.max(w, 1), Math.max(h, 1));
    }
  }
  drawEsdfLegend();
  ctx.restore();
}

function drawPath(points, color, width, alpha = 1, targetCtx = ctx, targetCanvas = canvas, bounds = sceneBounds) {
  if (!points || points.length < 2) return;
  targetCtx.save();
  targetCtx.globalAlpha = alpha;
  targetCtx.strokeStyle = color;
  targetCtx.lineWidth = width;
  targetCtx.beginPath();
  points.forEach(([x, y], i) => {
    const [px, py] = worldToCanvas(x, y, targetCanvas, bounds);
    if (i === 0) targetCtx.moveTo(px, py);
    else targetCtx.lineTo(px, py);
  });
  targetCtx.stroke();
  targetCtx.restore();
}

function drawFootprint(footprint, colorFill = "rgba(0, 169, 201, 0.24)", colorStroke = "#00a9c9") {
  if (!footprint?.length) return;
  ctx.fillStyle = colorFill;
  ctx.strokeStyle = colorStroke;
  ctx.lineWidth = 3;
  ctx.beginPath();
  footprint.forEach(([x, y], i) => {
    const [px, py] = worldToCanvas(x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawHudPanel(lines) {
  const size = logicalSize();
  const view = sceneView();
  const pad = 10;
  const lineH = 18;
  const boxPadX = 12;
  const boxPadY = 10;
  ctx.save();
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.font = "13px system-ui";
  const textW = Math.max(0, ...lines.map((line) => ctx.measureText(line).width));
  const boxW = Math.min(size.width - pad * 2, Math.ceil(textW + boxPadX * 2));
  const boxH = boxPadY * 2 + lines.length * lineH;
  const x = clamp(view.offsetX + 8, pad, Math.max(pad, size.width - boxW - pad));
  const y = clamp(view.offsetY + 8, pad, Math.max(pad, size.height - boxH - pad));
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.fillRect(x, y, boxW, boxH);
  ctx.beginPath();
  ctx.rect(x, y, boxW, boxH);
  ctx.clip();
  ctx.fillStyle = "#17202a";
  lines.forEach((line, i) => ctx.fillText(line, x + boxPadX, y + boxPadY + (i + 1) * lineH - 4));
  ctx.restore();
}

function drawRealtimeOverlay() {
  if (!currentFrame) {
    drawHudPanel(["Click Realtime to start live planning."]);
    return;
  }
  drawPath(realtimePath, "#0f766e", 4);
  drawPath(currentFrame.selected_trajectory_xy, "#00a9c9", 3, 0.95);
  const hit = Boolean(currentFrame.collision);
  drawFootprint(
    currentFrame.robot_footprint_xy || [],
    hit ? "rgba(218, 30, 40, 0.35)" : "rgba(0, 169, 201, 0.24)",
    hit ? "#da1e28" : "#00a9c9",
  );
  drawHeadingArrow(currentFrame.robot_xy, currentFrame.robot_yaw_deg, "#003f4a");
  if (currentFrame.robot_xy) drawMarker(currentFrame.robot_xy, "", "#003f4a", 5);

  const hud = [
    `realtime tick ${Math.max(0, realtimePath.length - 1)}`,
    `cmd [${(currentFrame.selected_param || []).map((v) => Number(v).toFixed(2)).join(", ")}]`,
  ];
  if (currentFrame.collision) hud.unshift("COLLISION (geometry) — reset to continue");
  drawHudPanel(hud);
}

function drawScene() {
  if (!config) return;
  const size = syncSceneCanvasSize();
  ctx.clearRect(0, 0, size.width, size.height);
  drawGrid();
  drawGround();
  drawEsdfSceneOverlay();
  drawObjects();
  drawStartTarget();
  drawRealtimeOverlay();
}

function syncMapCanvasSize() {
  const { width: cssW, height: cssH } = logicalSize(mapCanvas);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const nextW = Math.max(1, Math.round(cssW * dpr));
  const nextH = Math.max(1, Math.round(cssH * dpr));
  if (mapCanvas.width !== nextW || mapCanvas.height !== nextH) {
    mapCanvas.width = nextW;
    mapCanvas.height = nextH;
  }
  mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: cssW, height: cssH };
}

function drawPerceptionMap() {
  if (!config) return;
  const bounds = perceptionBounds();
  const size = syncMapCanvasSize();
  mapCtx.clearRect(0, 0, size.width, size.height);
  mapCtx.fillStyle = "#fbfcfd";
  mapCtx.fillRect(0, 0, size.width, size.height);

  const hasMap = config?.map_info || currentFrame?.map_info;
  if (!hasMap) {
    mapCtx.fillStyle = "#5c6975";
    mapCtx.font = "13px system-ui";
    mapCtx.textAlign = "center";
    mapCtx.textBaseline = "middle";
    mapCtx.fillText("Load a map to see overview", size.width / 2, size.height / 2);
    return;
  }

  drawMapBackgroundOn(mapCtx, mapCanvas, bounds);

  config.objects.forEach((obj) => {
    const [x, y] = obj.center;
    const [sx, sy] = obj.size;
    const [px0, py0] = worldToCanvas(x - sx / 2, y - sy / 2, mapCanvas, bounds);
    const [px1, py1] = worldToCanvas(x + sx / 2, y + sy / 2, mapCanvas, bounds);
    const rx = Math.min(px0, px1);
    const ry = Math.min(py0, py1);
    mapCtx.fillStyle = "rgba(95, 108, 120, 0.55)";
    mapCtx.strokeStyle = "#56616c";
    mapCtx.lineWidth = 1.5;
    mapCtx.beginPath();
    mapCtx.rect(rx, ry, Math.abs(px1 - px0), Math.abs(py1 - py0));
    mapCtx.fill();
    mapCtx.stroke();
  });

  mapCtx.save();
  mapCtx.textAlign = "center";
  mapCtx.textBaseline = "alphabetic";
  drawPath(realtimePath, "#0f766e", 3, 1, mapCtx, mapCanvas, bounds);
  const robotXY = robotDragXY();
  const showStart = !realtimeRunning
    || Math.hypot(config.start.xy[0] - robotXY[0], config.start.xy[1] - robotXY[1]) > 0.15;
  if (showStart) drawMarker(config.start.xy, "", "#24a148", 5, mapCtx, mapCanvas, bounds);
  drawMarker(config.target, "", "#da1e28", 5, mapCtx, mapCanvas, bounds);
  drawMarker(robotXY, "", "#003f4a", 6, mapCtx, mapCanvas, bounds);
  const robotYaw = realtimeRunning && currentFrame?.robot_yaw_deg != null
    ? currentFrame.robot_yaw_deg
    : config.start.yaw_deg;
  drawHeadingArrow(robotXY, robotYaw, "#003f4a", 0.35, mapCtx, mapCanvas, bounds);
  mapCtx.restore();
}

function depthColor(value) {
  if (value <= 0) return [10, 12, 20];
  const t = clamp(value / 255, 0, 1);
  for (let i = 0; i < DEPTH_STOPS.length - 1; i += 1) {
    const [t0, c0] = DEPTH_STOPS[i];
    const [t1, c1] = DEPTH_STOPS[i + 1];
    if (t <= t1 || i === DEPTH_STOPS.length - 2) {
      return lerpRGB(c0, c1, (t - t0) / Math.max(1e-6, t1 - t0));
    }
  }
  return DEPTH_STOPS.at(-1)[1];
}

function drawU8Canvas(targetCanvas, targetCtx, payload) {
  if (!payload?.data) {
    targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
    return;
  }
  const image = targetCtx.createImageData(payload.width, payload.height);
  for (let i = 0; i < payload.data.length; i += 1) {
    const [r, g, b] = depthColor(payload.data[i]);
    const j = i * 4;
    image.data[j] = r;
    image.data[j + 1] = g;
    image.data[j + 2] = b;
    image.data[j + 3] = 255;
  }
  const offscreen = document.createElement("canvas");
  offscreen.width = payload.width;
  offscreen.height = payload.height;
  offscreen.getContext("2d").putImageData(image, 0, 0);
  targetCtx.imageSmoothingEnabled = false;
  targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetCtx.save();
  targetCtx.scale(-1, 1);
  targetCtx.drawImage(offscreen, -targetCanvas.width, 0, targetCanvas.width, targetCanvas.height);
  targetCtx.restore();
}

function setControlsVisible(visible) {
  controlsPanel.classList.toggle("is-hidden", !visible);
  controlsPanel.classList.toggle("is-visible", visible);
  controlsToggle.setAttribute("aria-expanded", String(visible));
  controlsToggle.textContent = visible ? "Hide Controls" : "Show Controls";
}

let lastControlsWide = null;
function syncControlsLayout() {
  const wide = window.innerWidth > 1180;
  if (wide === lastControlsWide) return;
  lastControlsWide = wide;
  setControlsVisible(wide);
}

function drawRealtimeFrame(frame) {
  currentFrame = frame;
  drawU8Canvas(depthCanvas, depthCtx, frame.depth_u8);
  if (config?.map_info) updateLabViewBounds(frame.robot_xy);
  drawScene();
  drawPerceptionMap();
}

function refreshFields() {
  const obj = selectedObject();
  config.start = config.start || { xy: [0, 0], yaw_deg: 0 };
  if (config.robot?.name) fields.robotPreset.value = config.robot.name;
  if (config.map_name) fields.mapSelect.value = config.map_name;
  else if (config.map_path) {
    const match = mapCatalog.find((entry) => entry.path === config.map_path);
    fields.mapSelect.value = match?.name || "";
  } else {
    fields.mapSelect.value = "";
  }
  writeBoundFields();
  if (obj) {
    fields.objectName.value = obj.name;
    fields.objectX.value = obj.center[0];
    fields.objectY.value = obj.center[1];
    fields.objectSX.value = obj.size[0];
    fields.objectSY.value = obj.size[1];
    fields.objectSZ.value = obj.size[2];
  }
  fields.configText.value = JSON.stringify(config, null, 2);
  fields.robotSummary.innerHTML = robotSummaryHtml();
  fields.cameraSummary.innerHTML = cameraSummaryHtml();
  fields.mapSummary.innerHTML = mapSummaryHtml();
  if (config.map_info && !realtimeRunning) updateLabViewBounds(config.start?.xy);
  drawScene();
  drawPerceptionMap();
}

function syncConfigFromFields() {
  const obj = selectedObject();
  config.start = config.start || { xy: [0, 0], yaw_deg: 0 };
  config.map_name = fields.mapSelect.value.trim() || null;
  config.map_path = mapCatalog.find((entry) => entry.name === config.map_name)?.path || config.map_path || null;
  readBoundFields();
  if (config.robot) {
    config.robot.name = fields.robotPreset.value || config.robot.name || "go2";
    config.obstacle = structuredClone(config.robot.obstacle || {});
  }
  if (obj) {
    obj.name = fields.objectName.value || obj.name;
    obj.center[0] = Number(fields.objectX.value);
    obj.center[1] = Number(fields.objectY.value);
    obj.size[0] = Number(fields.objectSX.value);
    obj.size[1] = Number(fields.objectSY.value);
    obj.size[2] = Number(fields.objectSZ.value);
  }
}

function applyFieldChanges() {
  syncConfigFromFields();
  refreshFields();
}

function syncStartFieldsFromFrame(frame) {
  config.start.xy = frame.next_start.xy;
  config.start.yaw_deg = frame.next_start.yaw_deg;
  fields.startX.value = config.start.xy[0];
  fields.startY.value = config.start.xy[1];
  fields.startYaw.value = config.start.yaw_deg;
}

async function pushConfigToServer(reset = false) {
  syncConfigFromFields();
  const response = await fetch("/api/update-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config, reset }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Config update failed");
  return data;
}

function robotDragXY() {
  if (realtimeRunning && currentFrame?.robot_xy) return currentFrame.robot_xy;
  return config.start.xy;
}

function labBottomRightAddXY() {
  const pad = 0.5;
  const b = sceneBounds;
  // Lab view: screen right = +Y, screen bottom = low X.
  return [Number((b.xMin + pad).toFixed(2)), Number((b.yMax - pad).toFixed(2))];
}

function stopRealtimeForEdit(message = "Realtime stopped (editing)") {
  if (!realtimeRunning) return;
  stopRealtime();
  statusEl.textContent = message;
}

function editScene() {
  stopRealtimeForEdit();
  applyFieldChanges();
}

function hitTestTarget(px, py, targetCanvas = canvas, bounds = sceneBounds) {
  const [tpx, tpy] = worldToCanvas(config.target[0], config.target[1], targetCanvas, bounds);
  return Math.hypot(px - tpx, py - tpy) <= MARKER_HIT_RADIUS;
}

function hitTestStart(px, py, targetCanvas = canvas, bounds = sceneBounds) {
  const [sx, sy] = worldToCanvas(config.start.xy[0], config.start.xy[1], targetCanvas, bounds);
  return Math.hypot(px - sx, py - sy) <= MARKER_HIT_RADIUS;
}

function hitTestRobot(px, py, targetCanvas = canvas, bounds = sceneBounds) {
  const xy = robotDragXY();
  const [rx, ry] = worldToCanvas(xy[0], xy[1], targetCanvas, bounds);
  return Math.hypot(px - rx, py - ry) <= MARKER_HIT_RADIUS;
}

function hitTestObject(px, py) {
  for (let i = config.objects.length - 1; i >= 0; i -= 1) {
    const [x, y, w, h] = objectCanvasRect(config.objects[i]);
    if (px >= x && px <= x + w && py >= y && py <= y + h) return i;
  }
  return -1;
}

function stopRealtime() {
  realtimeRunning = false;
  realtimePending = false;
  if (realtimeTimer !== null) {
    clearTimeout(realtimeTimer);
    realtimeTimer = null;
  }
  realtimeButton.textContent = "Realtime";
  realtimeButton.classList.add("primary");
}

function scheduleRealtimeTick(delay = REALTIME_TICK_DELAY_MS) {
  if (!realtimeRunning) return;
  if (realtimeBusy) {
    realtimePending = true;
    return;
  }
  if (realtimeTimer !== null) clearTimeout(realtimeTimer);
  realtimeTimer = setTimeout(() => {
    realtimeTimer = null;
    realtimeTick();
  }, delay);
}

async function realtimeTick() {
  if (!realtimeRunning) return;
  if (realtimeBusy) {
    realtimePending = true;
    return;
  }
  realtimeBusy = true;
  realtimePending = false;
  try {
    const response = await fetch("/api/sim-state");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Sim state failed");
    const frame = data.frame;
    syncStartFieldsFromFrame(frame);
    realtimePath.push(frame.robot_xy);
    if (realtimePath.length > 300) realtimePath = realtimePath.slice(-300);
    drawRealtimeFrame(frame);
    statusEl.textContent = "Realtime running";
  } catch (error) {
    statusEl.textContent = "Realtime error";
    stopRealtime();
  } finally {
    realtimeBusy = false;
    if (realtimeRunning) scheduleRealtimeTick(realtimePending ? 0 : REALTIME_TICK_DELAY_MS);
  }
}

async function toggleRealtime() {
  if (realtimeRunning) {
    stopRealtime();
    statusEl.textContent = "Realtime stopped";
    return;
  }
  applyFieldChanges();
  statusEl.textContent = "Starting ROS loop...";
  try {
    const response = await fetch("/api/start-ros-loop", { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).detail || "Failed to start ROS loop");
    await pushConfigToServer(true);
  } catch (error) {
    statusEl.textContent = "ROS start error";
    return;
  }
  realtimeRunning = true;
  realtimeBusy = false;
  realtimePending = false;
  realtimePath = [config.start.xy.slice()];
  currentFrame = null;
  realtimeButton.textContent = "Stop";
  realtimeButton.classList.remove("primary");
  statusEl.textContent = "Realtime starting...";
  realtimeTick();
}

async function fetchRobotPresets() {
  try {
    const data = await (await fetch("/api/robot-presets")).json();
    robotPresets = Object.fromEntries((data.presets || []).map((entry) => [entry.name, entry.robot]));
    const select = fields.robotPreset;
    select.innerHTML = "";
    (data.presets || []).forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.name;
      option.textContent = entry.name.toUpperCase();
      select.appendChild(option);
    });
    if (config?.robot?.name && robotPresets[config.robot.name]) {
      select.value = config.robot.name;
    }
  } catch (error) {
    statusEl.textContent = "Robot preset load error";
  }
}

async function fetchMapCatalog() {
  if (!fields.mapSelect) {
    fields.mapSummary.innerHTML = "Map selector missing — hard refresh the page (Ctrl+Shift+R).";
    return;
  }
  try {
    const response = await fetch("/api/map-catalog");
    let data = null;
    try {
      data = await response.json();
    } catch {
      throw new Error(`Invalid response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data?.detail || `HTTP ${response.status}`);
    }
    mapCatalog = data.maps || [];
    const select = fields.mapSelect;
    const current = select.value;
    select.innerHTML = '<option value="">— select a map —</option>';
    mapCatalog.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.name;
      option.textContent = entry.name;
      select.appendChild(option);
    });
    if (current && mapCatalog.some((entry) => entry.name === current)) {
      select.value = current;
    } else if (config?.map_name && mapCatalog.some((entry) => entry.name === config.map_name)) {
      select.value = config.map_name;
    }
    if (!data.maps_root_exists) {
      fields.mapSummary.innerHTML = `Maps folder missing: <code>${data.maps_root}</code>`;
    } else if (!mapCatalog.length) {
      fields.mapSummary.innerHTML = `No loadable maps under <code>${data.maps_root}</code> (need <code>occupancy_grid.npy</code> in each subfolder).`;
    } else if (!config?.map_info) {
      fields.mapSummary.innerHTML = `${mapCatalog.length} map(s) available under <code>${data.maps_root}</code>.`;
    }
  } catch (error) {
    fields.mapSummary.innerHTML = `Map catalog failed: ${error.message}. Restart <code>./scripts/run_ros_planning_web.sh</code> and hard-refresh.`;
    statusEl.textContent = "Map catalog error";
  }
}

async function loadDefault() {
  stopRealtime();
  config = await (await fetch("/api/default-config")).json();
  sceneBounds = { ...DEFAULT_SCENE };
  config.map_info = null;
  config.map_background = null;
  setMapBackground(null);
  selectedIndex = 0;
  currentFrame = null;
  realtimePath = [];
  fields.mapSelect.value = "";
  refreshFields();
  statusEl.textContent = "Ready";
}

async function loadScenarioFromSelection() {
  const key = fields.scenarioSelect.value;
  const scenario = SCENARIOS[key];
  if (!scenario) {
    statusEl.textContent = "Select a scenario first";
    return;
  }
  stopRealtime();
  if (!config) config = await (await fetch("/api/default-config")).json();
  const robot = structuredClone(config.robot || robotPresets.go2 || {});
  const camera = structuredClone(config.camera || {});
  camera.max_range = scenario.cameraMaxRange;
  config = {
    ...structuredClone(config),
    name: scenario.label,
    robot,
    obstacle: structuredClone(robot.obstacle || config.obstacle || {}),
    camera,
    start: structuredClone(scenario.start),
    target: structuredClone(scenario.target),
    map_path: null,
    map_name: null,
    map_info: null,
    map_background: null,
    objects: structuredClone(scenario.objects),
  };
  setMapBackground(null);
  sceneBounds = { ...scenario.bounds };
  selectedIndex = config.objects.length ? 0 : -1;
  currentFrame = null;
  realtimePath = [];
  fields.mapSelect.value = "";
  refreshFields();
  statusEl.textContent = `Loading ${scenario.label}...`;
  try {
    await pushConfigToServer(true);
    statusEl.textContent = `Loaded ${scenario.label}`;
  } catch (error) {
    statusEl.textContent = `Scenario error: ${error.message}`;
  }
}

async function loadMapFromSelection() {
  const mapName = fields.mapSelect.value.trim();
  if (!mapName) {
    statusEl.textContent = "Select a map first";
    return;
  }
  stopRealtime();
  applyFieldChanges();
  statusEl.textContent = "Loading map...";
  try {
    const response = await fetch("/api/load-map", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        map_name: mapName,
        start_xy: null,
        yaw_deg: Number(fields.startYaw.value) || 0,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Map load failed");
    config = data.config;
    config.map_info = data.map_info;
    config.map_background = data.map_background;
    setMapBackground(data.map_background);
    updateLabViewBounds(config.start.xy);
    selectedIndex = config.objects?.length ? Math.min(selectedIndex, config.objects.length - 1) : 0;
    currentFrame = null;
    realtimePath = [config.start.xy.slice()];
    refreshFields();
    statusEl.textContent = `Map loaded: ${mapName}`;
  } catch (error) {
    statusEl.textContent = `Map error: ${error.message}`;
  }
}

function selectObject(index) {
  selectedIndex = index;
  refreshFields();
}

boundInputs().forEach((el) => {
  el.addEventListener("change", editScene);
});
[fields.objectName, fields.objectX, fields.objectY, fields.objectSX, fields.objectSY, fields.objectSZ].forEach((el) => {
  el.addEventListener("change", editScene);
});

fields.robotPreset.addEventListener("change", () => {
  const preset = robotPresets[fields.robotPreset.value];
  if (preset) {
    config.robot = structuredClone(preset);
    config.obstacle = structuredClone(preset.obstacle || {});
  }
  stopRealtimeForEdit();
  refreshFields();
});

realtimeButton.addEventListener("click", toggleRealtime);
controlsToggle.addEventListener("click", () => setControlsVisible(controlsPanel.classList.contains("is-hidden")));
$("resetScene").addEventListener("click", loadDefault);
$("loadMap").addEventListener("click", loadMapFromSelection);
$("loadScenario").addEventListener("click", loadScenarioFromSelection);
$("applyJson").addEventListener("click", () => {
  const next = JSON.parse(fields.configText.value);
  if (!next.obstacle) next.obstacle = config.obstacle;
  config = next;
  if (!config.robot) config.robot = structuredClone(robotPresets.go2 || {});
  if (config.map_info) updateLabViewBounds(config.start?.xy);
  else sceneBounds = { ...DEFAULT_SCENE };
  selectedIndex = config.objects?.length ? Math.min(selectedIndex, config.objects.length - 1) : 0;
  currentFrame = null;
  refreshFields();
});
$("addBox").addEventListener("click", () => {
  const [x, y] = labBottomRightAddXY();
  const z = Number(config.camera?.mount_height ?? 0.45) * 0.75;
  stopRealtimeForEdit();
  config.objects.push({
    name: `box_${config.objects.length + 1}`,
    kind: "box",
    center: [x, y, z],
    size: [0.5, 0.5, 0.8],
  });
  selectObject(config.objects.length - 1);
});
$("duplicateObject").addEventListener("click", () => {
  const obj = selectedObject();
  if (!obj) return;
  stopRealtimeForEdit();
  const copy = structuredClone(obj);
  copy.name = `${copy.name}_copy`;
  copy.center[1] += 0.4;
  config.objects.push(copy);
  selectObject(config.objects.length - 1);
});
$("deleteObject").addEventListener("click", () => {
  if (!config.objects.length) return;
  stopRealtimeForEdit();
  config.objects.splice(selectedIndex, 1);
  selectObject(Math.max(0, selectedIndex - 1));
});

function beginDrag(type, wx, wy, event, sourceCanvas) {
  let ox;
  let oy;
  if (type === "target") {
    ox = config.target[0];
    oy = config.target[1];
  } else if (realtimeRunning && currentFrame?.robot_xy) {
    [ox, oy] = currentFrame.robot_xy;
  } else {
    [ox, oy] = config.start.xy;
  }
  dragState = { type, dx: ox - wx, dy: oy - wy, sourceCanvas };
  sourceCanvas.setPointerCapture(event.pointerId);
}

function applyDragMove(wx, wy) {
  if (!dragState) return;
  if (dragState.type === "target") {
    config.target[0] = Number((wx + dragState.dx).toFixed(2));
    config.target[1] = Number((wy + dragState.dy).toFixed(2));
  } else {
    config.start.xy[0] = Number((wx + dragState.dx).toFixed(2));
    config.start.xy[1] = Number((wy + dragState.dy).toFixed(2));
    if (currentFrame?.robot_xy) currentFrame.robot_xy = config.start.xy.slice();
    if (config.map_info) updateLabViewBounds(config.start.xy);
  }
  refreshFields();
}

function handleCanvasPointerDown(event, targetCanvas, bounds) {
  if (!config) return;
  const [px, py] = pointerPosOn(event, targetCanvas);
  if (hitTestTarget(px, py, targetCanvas, bounds)) {
    stopRealtimeForEdit();
    const [wx, wy] = canvasToWorld(px, py, targetCanvas, bounds);
    beginDrag("target", wx, wy, event, targetCanvas);
    return;
  }
  if (hitTestRobot(px, py, targetCanvas, bounds)) {
    stopRealtimeForEdit();
    const [wx, wy] = canvasToWorld(px, py, targetCanvas, bounds);
    beginDrag("start", wx, wy, event, targetCanvas);
    return;
  }
  if (targetCanvas === canvas && hitTestStart(px, py, targetCanvas, bounds)) {
    stopRealtimeForEdit();
    const [wx, wy] = canvasToWorld(px, py, targetCanvas, bounds);
    beginDrag("start", wx, wy, event, targetCanvas);
    return;
  }
  if (targetCanvas === canvas) {
    const hit = hitTestObject(px, py);
    if (hit < 0) return;
    stopRealtimeForEdit();
    selectedIndex = hit;
    const [wx, wy] = canvasToWorld(px, py, targetCanvas, bounds);
    const obj = selectedObject();
    dragState = { type: "object", dx: obj.center[0] - wx, dy: obj.center[1] - wy, sourceCanvas: targetCanvas };
    targetCanvas.setPointerCapture(event.pointerId);
    refreshFields();
  }
}

function handleCanvasPointerMove(event, targetCanvas, bounds) {
  if (!dragState || dragState.sourceCanvas !== targetCanvas) return;
  const [px, py] = pointerPosOn(event, targetCanvas);
  const [wx, wy] = canvasToWorld(px, py, targetCanvas, bounds);
  if (dragState.type === "object") {
    const obj = selectedObject();
    obj.center[0] = Number((wx + dragState.dx).toFixed(2));
    obj.center[1] = Number((wy + dragState.dy).toFixed(2));
  } else {
    applyDragMove(wx, wy);
  }
  refreshFields();
}

function handleCanvasPointerUp(event, targetCanvas) {
  if (dragState?.sourceCanvas === targetCanvas) dragState = null;
}

canvas.addEventListener("pointerdown", (event) => handleCanvasPointerDown(event, canvas, sceneBounds));
canvas.addEventListener("pointermove", (event) => handleCanvasPointerMove(event, canvas, sceneBounds));
canvas.addEventListener("pointerup", (event) => handleCanvasPointerUp(event, canvas));
canvas.addEventListener("pointercancel", (event) => handleCanvasPointerUp(event, canvas));

mapCanvas.addEventListener("pointerdown", (event) => {
  if (!config?.map_info && !currentFrame?.map_info) return;
  handleCanvasPointerDown(event, mapCanvas, perceptionBounds());
});
mapCanvas.addEventListener("pointermove", (event) => handleCanvasPointerMove(event, mapCanvas, perceptionBounds()));
mapCanvas.addEventListener("pointerup", (event) => handleCanvasPointerUp(event, mapCanvas));
mapCanvas.addEventListener("pointercancel", (event) => handleCanvasPointerUp(event, mapCanvas));

loadDefault().then(async () => {
  await Promise.all([fetchMapCatalog(), fetchRobotPresets()]);
});
syncControlsLayout();
window.addEventListener("resize", () => {
  syncControlsLayout();
  drawScene();
  drawPerceptionMap();
});
if (typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => {
    drawScene();
    drawPerceptionMap();
  }).observe(canvas);
  new ResizeObserver(() => drawPerceptionMap()).observe(mapCanvas);
}
