const REALTIME_TICK_DELAY_MS = 33;
const GRID_STEP_M = 1.0;
const SCENE = { xMin: -0.8, xMax: 9.8, yMin: -5.4, yMax: 5.4 };
const CAM_DEFAULTS = { width: 160, image_height: 100, fx: 80, fy: 25, max_range: 15, mount_height: 0.45 };
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
let realtimeNeedsReset = true;

const canvas = document.getElementById("sceneCanvas");
const ctx = canvas.getContext("2d");
const depthCanvas = document.getElementById("depthCanvas");
const depthCtx = depthCanvas.getContext("2d");
const statusEl = document.getElementById("status");
const controlsPanel = document.getElementById("controlsPanel");
const controlsToggle = document.getElementById("controlsToggle");
const realtimeButton = document.getElementById("realtimeButton");
const $ = (id) => document.getElementById(id);

const fields = {
  targetX: $("targetX"),
  targetY: $("targetY"),
  camWidth: $("camWidth"),
  camHeight: $("camHeight"),
  camFx: $("camFx"),
  camFy: $("camFy"),
  camMaxRange: $("camMaxRange"),
  camMountHeight: $("camMountHeight"),
  objectName: $("objectName"),
  objectX: $("objectX"),
  objectY: $("objectY"),
  objectSX: $("objectSX"),
  objectSY: $("objectSY"),
  objectSZ: $("objectSZ"),
  configText: $("configText"),
  robotSummary: $("robotSummary"),
  cameraSummary: $("cameraSummary"),
};

const editableFields = [
  fields.targetX, fields.targetY,
  fields.camWidth, fields.camHeight, fields.camFx, fields.camFy, fields.camMaxRange, fields.camMountHeight,
  fields.objectName, fields.objectX, fields.objectY, fields.objectSX, fields.objectSY, fields.objectSZ,
];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function num(el, fallback, min = null, round = false) {
  let v = Number(el.value);
  if (!Number.isFinite(v)) v = fallback;
  if (round) v = Math.round(v);
  return min == null ? v : Math.max(min, v);
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

function sceneView(target = canvas) {
  const b = SCENE;
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

function worldToCanvas(x, y, target = canvas) {
  const v = sceneView(target);
  return [v.offsetX + (y - v.b.yMin) * v.scale, v.offsetY + (v.b.xMax - x) * v.scale];
}

function canvasToWorld(px, py) {
  const v = sceneView();
  return [v.b.xMax - (py - v.offsetY) / v.scale, v.b.yMin + (px - v.offsetX) / v.scale];
}

function pointerPos(event) {
  const rect = canvas.getBoundingClientRect();
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
  const o = config.obstacle || {};
  return summaryLines([
    ["preset", r.preset || "go2"],
    ["size", `${Number(r.length || 0).toFixed(2)} x ${Number(r.width || 0).toFixed(2)} m`],
    ["camera/control", `${Number(r.camera_x || 0).toFixed(2)} / ${Number(r.control_x || 0).toFixed(2)} m`],
    ["safety radius", `${Number(r.safety_radius || 0).toFixed(2)} m`],
    ["obstacle dilation", `${Number(o.dilation_cells || 0)}`],
  ]);
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

function drawGrid() {
  const { width, height } = logicalSize();
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e0e7ee";
  ctx.lineWidth = 1;
  const b = SCENE;
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
  const b = SCENE;
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

function drawMarker(xy, label, color, radius = 7) {
  const [px, py] = worldToCanvas(xy[0], xy[1]);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillText(label, px, py - 12);
}

function drawHeadingArrow(xy, yawDeg, color, length = 0.45) {
  if (!xy || yawDeg == null) return;
  const yaw = (Number(yawDeg) * Math.PI) / 180;
  const tip = [xy[0] + Math.cos(yaw) * length, xy[1] + Math.sin(yaw) * length];
  const [x0, y0] = worldToCanvas(xy[0], xy[1]);
  const [x1, y1] = worldToCanvas(tip[0], tip[1]);
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  const angle = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - Math.cos(angle - 0.55) * 12, y1 - Math.sin(angle - 0.55) * 12);
  ctx.lineTo(x1 - Math.cos(angle + 0.55) * 12, y1 - Math.sin(angle + 0.55) * 12);
  ctx.closePath();
  ctx.fill();
}

function drawStartTarget() {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  drawMarker(config.start.xy, "control", "#24a148");
  const camera = cameraPose();
  drawMarker(camera.xy, "camera", "#6d28d9", 6);
  drawHeadingArrow(camera.xy, config.start.yaw_deg, "#6d28d9", 0.32);
  drawMarker(config.target, "target", "#da1e28");
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

function drawPath(points, color, width, alpha = 1) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  points.forEach(([x, y], i) => {
    const [px, py] = worldToCanvas(x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.restore();
}

function drawFootprint(footprint) {
  if (!footprint?.length) return;
  ctx.fillStyle = "rgba(0, 169, 201, 0.24)";
  ctx.strokeStyle = "#007d95";
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
  (currentFrame.candidate_trajectories_xy || []).forEach((path) => drawPath(path, "#8d99a6", 1.1, 0.32));
  drawPath(currentFrame.selected_trajectory_xy, "#00a9c9", 3, 0.95);
  drawFootprint(currentFrame.robot_footprint_xy || []);
  drawHeadingArrow(currentFrame.robot_xy, currentFrame.robot_yaw_deg, "#003f4a");
  if (currentFrame.robot_xy) drawMarker(currentFrame.robot_xy, "", "#003f4a", 5);

  const lines = [
    `realtime tick ${Math.max(0, realtimePath.length - 1)}`,
    `cmd [${(currentFrame.selected_param || []).map((v) => Number(v).toFixed(2)).join(", ")}]`,
    `clearance ${Number(currentFrame.front_clearance || 0).toFixed(2)}m`,
  ];
  if (currentFrame.should_reverse || currentFrame.selected_is_reverse) {
    lines.push(`reverse ${currentFrame.selected_is_reverse ? "selected" : "gated"}`);
  }
  drawHudPanel(lines);
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

function syncControlsLayout() {
  setControlsVisible(window.innerWidth > 1180);
}

function drawRealtimeFrame(frame) {
  currentFrame = frame;
  drawU8Canvas(depthCanvas, depthCtx, frame.depth_u8);
  drawScene();
}

function refreshFields() {
  const obj = selectedObject();
  const cam = config.camera || (config.camera = {});
  fields.targetX.value = config.target[0];
  fields.targetY.value = config.target[1];
  fields.camWidth.value = cam.width ?? CAM_DEFAULTS.width;
  fields.camHeight.value = cam.image_height ?? cam.height ?? CAM_DEFAULTS.image_height;
  fields.camFx.value = cam.fx ?? CAM_DEFAULTS.fx;
  fields.camFy.value = cam.fy ?? CAM_DEFAULTS.fy;
  fields.camMaxRange.value = cam.max_range ?? CAM_DEFAULTS.max_range;
  fields.camMountHeight.value = cam.mount_height ?? CAM_DEFAULTS.mount_height;
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
  drawScene();
}

function applyFieldChanges() {
  const obj = selectedObject();
  const cam = config.camera || (config.camera = {});
  config.target[0] = Number(fields.targetX.value);
  config.target[1] = Number(fields.targetY.value);
  cam.width = num(fields.camWidth, CAM_DEFAULTS.width, 16, true);
  cam.image_height = num(fields.camHeight, CAM_DEFAULTS.image_height, 16, true);
  cam.fx = num(fields.camFx, CAM_DEFAULTS.fx, 1);
  cam.fy = num(fields.camFy, CAM_DEFAULTS.fy, 1);
  cam.max_range = num(fields.camMaxRange, CAM_DEFAULTS.max_range, 0.5);
  cam.mount_height = num(fields.camMountHeight, CAM_DEFAULTS.mount_height, 0.05);
  if (obj) {
    obj.name = fields.objectName.value || obj.name;
    obj.center[0] = Number(fields.objectX.value);
    obj.center[1] = Number(fields.objectY.value);
    obj.size[0] = Number(fields.objectSX.value);
    obj.size[1] = Number(fields.objectSY.value);
    obj.size[2] = Number(fields.objectSZ.value);
  }
  refreshFields();
}

function hitTestTarget(px, py) {
  const [tpx, tpy] = worldToCanvas(config.target[0], config.target[1]);
  return Math.hypot(px - tpx, py - tpy) <= 14;
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

function noteSceneEdited(resetOccupancy = false, immediate = true) {
  if (resetOccupancy) realtimeNeedsReset = true;
  if (realtimeRunning && immediate) scheduleRealtimeTick(0);
}

async function realtimeTick() {
  if (!realtimeRunning) return;
  if (realtimeBusy) {
    realtimePending = true;
    return;
  }
  realtimeBusy = true;
  realtimePending = false;
  applyFieldChanges();
  try {
    const response = await fetch("/api/realtime-step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, advance_step: 3, reset: realtimeNeedsReset }),
    });
    realtimeNeedsReset = false;
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Realtime step failed");
    const frame = data.frame;
    config.start.xy = frame.next_start.xy;
    config.start.yaw_deg = frame.next_start.yaw_deg;
    realtimePath.push(frame.robot_xy);
    if (realtimePath.length > 300) realtimePath = realtimePath.slice(-300);
    drawRealtimeFrame(frame);
    refreshFields();
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
  } catch (error) {
    statusEl.textContent = "ROS start error";
    return;
  }
  realtimeRunning = true;
  realtimeBusy = false;
  realtimePending = false;
  realtimeNeedsReset = true;
  realtimePath = [config.start.xy.slice()];
  currentFrame = null;
  realtimeButton.textContent = "Stop";
  realtimeButton.classList.remove("primary");
  statusEl.textContent = "Realtime starting...";
  realtimeTick();
}

async function loadDefault() {
  stopRealtime();
  config = await (await fetch("/api/default-config")).json();
  selectedIndex = 0;
  currentFrame = null;
  realtimePath = [];
  refreshFields();
  statusEl.textContent = "Ready";
}

function selectObject(index) {
  selectedIndex = index;
  refreshFields();
  noteSceneEdited(true, true);
}

editableFields.forEach((field) => {
  field.addEventListener("change", () => {
    applyFieldChanges();
    noteSceneEdited(true, true);
  });
});

realtimeButton.addEventListener("click", toggleRealtime);
controlsToggle.addEventListener("click", () => setControlsVisible(controlsPanel.classList.contains("is-hidden")));
$("resetScene").addEventListener("click", loadDefault);
$("applyJson").addEventListener("click", () => {
  const next = JSON.parse(fields.configText.value);
  next.robot = config.robot;
  next.obstacle = config.obstacle;
  config = next;
  selectedIndex = Math.min(selectedIndex, config.objects.length - 1);
  currentFrame = null;
  refreshFields();
  noteSceneEdited(true, true);
});
$("addBox").addEventListener("click", () => {
  config.objects.push({ name: `box_${config.objects.length + 1}`, kind: "box", center: [2.0, 0.0, 0.35], size: [0.5, 0.5, 0.8] });
  selectObject(config.objects.length - 1);
});
$("duplicateObject").addEventListener("click", () => {
  const obj = selectedObject();
  if (!obj) return;
  const copy = structuredClone(obj);
  copy.name = `${copy.name}_copy`;
  copy.center[1] += 0.4;
  config.objects.push(copy);
  selectObject(config.objects.length - 1);
});
$("deleteObject").addEventListener("click", () => {
  if (!config.objects.length) return;
  config.objects.splice(selectedIndex, 1);
  selectObject(Math.max(0, selectedIndex - 1));
});

canvas.addEventListener("pointerdown", (event) => {
  const [px, py] = pointerPos(event);
  if (hitTestTarget(px, py)) {
    const [wx, wy] = canvasToWorld(px, py);
    dragState = { type: "target", dx: config.target[0] - wx, dy: config.target[1] - wy };
    canvas.setPointerCapture(event.pointerId);
    return;
  }
  const hit = hitTestObject(px, py);
  if (hit < 0) return;
  selectedIndex = hit;
  const [wx, wy] = canvasToWorld(px, py);
  const obj = selectedObject();
  dragState = { type: "object", dx: obj.center[0] - wx, dy: obj.center[1] - wy };
  canvas.setPointerCapture(event.pointerId);
  refreshFields();
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragState) return;
  const [px, py] = pointerPos(event);
  const [wx, wy] = canvasToWorld(px, py);
  if (dragState.type === "target") {
    config.target[0] = Number((wx + dragState.dx).toFixed(2));
    config.target[1] = Number((wy + dragState.dy).toFixed(2));
    noteSceneEdited(false, true);
  } else {
    const obj = selectedObject();
    obj.center[0] = Number((wx + dragState.dx).toFixed(2));
    obj.center[1] = Number((wy + dragState.dy).toFixed(2));
    noteSceneEdited(true, true);
  }
  refreshFields();
});

canvas.addEventListener("pointerup", () => {
  dragState = null;
});

loadDefault();
syncControlsLayout();
window.addEventListener("resize", () => {
  syncControlsLayout();
  drawScene();
});
if (typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => drawScene()).observe(canvas);
}
