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
const REALTIME_TICK_DELAY_MS = 33;

const canvas = document.getElementById("sceneCanvas");
const ctx = canvas.getContext("2d");
const depthCanvas = document.getElementById("depthCanvas");
const depthCtx = depthCanvas.getContext("2d");
const statusEl = document.getElementById("status");
const realtimeButton = document.getElementById("realtimeButton");
const diagnosticNote = document.getElementById("diagnosticNote");
const GRID_DIVISIONS = 20;

const fields = {
  targetX: document.getElementById("targetX"),
  targetY: document.getElementById("targetY"),
  objectName: document.getElementById("objectName"),
  objectX: document.getElementById("objectX"),
  objectY: document.getElementById("objectY"),
  objectSX: document.getElementById("objectSX"),
  objectSY: document.getElementById("objectSY"),
  objectSZ: document.getElementById("objectSZ"),
  configText: document.getElementById("configText"),
  robotSummary: document.getElementById("robotSummary"),
};

function sceneBounds() {
  return { xMin: -0.5, xMax: 5.0, yMin: -2.8, yMax: 2.8 };
}

function worldToCanvasOn(targetCanvas, x, y) {
  const b = sceneBounds();
  const px = ((y - b.yMin) / (b.yMax - b.yMin)) * targetCanvas.width;
  const py = targetCanvas.height - ((x - b.xMin) / (b.xMax - b.xMin)) * targetCanvas.height;
  return [px, py];
}

function worldToCanvas(x, y) {
  return worldToCanvasOn(canvas, x, y);
}

function canvasToWorld(px, py) {
  const b = sceneBounds();
  const y = b.yMin + (px / canvas.width) * (b.yMax - b.yMin);
  const x = b.xMin + ((canvas.height - py) / canvas.height) * (b.xMax - b.xMin);
  return [x, y];
}

function selectedObject() {
  return config?.objects?.[selectedIndex] || null;
}

function hitTestTarget(px, py) {
  const [tx, ty] = config.target;
  const [tpx, tpy] = worldToCanvas(tx, ty);
  return Math.hypot(px - tpx, py - tpy) <= 14;
}

function drawGrid(drawCtx, targetCanvas, mapper) {
  drawCtx.fillStyle = "#fbfcfd";
  drawCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
  drawCtx.strokeStyle = "#e0e7ee";
  drawCtx.lineWidth = 1;
  const b = sceneBounds();
  for (let i = 0; i <= GRID_DIVISIONS; i += 1) {
    const x = b.xMin + ((b.xMax - b.xMin) * i) / GRID_DIVISIONS;
    const [px0, py] = mapper(x, b.yMin);
    const [px1] = mapper(x, b.yMax);
    drawCtx.beginPath();
    drawCtx.moveTo(px0, py);
    drawCtx.lineTo(px1, py);
    drawCtx.stroke();
  }
  for (let i = 0; i <= GRID_DIVISIONS; i += 1) {
    const y = b.yMin + ((b.yMax - b.yMin) * i) / GRID_DIVISIONS;
    const [px, py0] = mapper(b.xMin, y);
    const [, py1] = mapper(b.xMax, y);
    drawCtx.beginPath();
    drawCtx.moveTo(px, py0);
    drawCtx.lineTo(px, py1);
    drawCtx.stroke();
  }
}

function drawObjects(drawCtx, mapper, highlightSelected) {
  config.objects.forEach((obj, idx) => {
    const [x, y] = obj.center;
    const [sx, sy] = obj.size;
    const [px0, py0] = mapper(x - sx / 2, y - sy / 2);
    const [px1, py1] = mapper(x + sx / 2, y + sy / 2);
    const selected = highlightSelected && idx === selectedIndex;
    drawCtx.fillStyle = selected ? "rgba(16, 107, 103, 0.55)" : "rgba(95, 108, 120, 0.45)";
    drawCtx.strokeStyle = selected ? "#106b67" : "#56616c";
    drawCtx.lineWidth = selected ? 3 : 1.5;
    drawCtx.beginPath();
    drawCtx.rect(Math.min(px0, px1), Math.min(py0, py1), Math.abs(px1 - px0), Math.abs(py1 - py0));
    drawCtx.fill();
    drawCtx.stroke();
    drawCtx.fillStyle = "#17202a";
    drawCtx.font = "13px system-ui";
    drawCtx.textAlign = "center";
    drawCtx.fillText(obj.name, (px0 + px1) / 2, (py0 + py1) / 2 + 4);
  });
}

function drawStartTarget(drawCtx, mapper) {
  const [sx, sy] = config.start.xy;
  const [spx, spy] = mapper(sx, sy);
  drawCtx.fillStyle = "#24a148";
  drawCtx.beginPath();
  drawCtx.arc(spx, spy, 7, 0, Math.PI * 2);
  drawCtx.fill();
  drawCtx.fillText("control", spx, spy - 12);

  const camera = cameraPose();
  const [cpx, cpy] = mapper(camera.xy[0], camera.xy[1]);
  drawCtx.fillStyle = "#6d28d9";
  drawCtx.beginPath();
  drawCtx.arc(cpx, cpy, 6, 0, Math.PI * 2);
  drawCtx.fill();
  drawCtx.fillText("camera", cpx, cpy - 12);
  drawHeadingArrow(drawCtx, mapper, camera.xy, config.start.yaw_deg, "#6d28d9", 0.32);

  const [tx, ty] = config.target;
  const [tpx, tpy] = mapper(tx, ty);
  drawCtx.fillStyle = "#da1e28";
  drawCtx.beginPath();
  drawCtx.arc(tpx, tpy, 7, 0, Math.PI * 2);
  drawCtx.fill();
  drawCtx.fillText("target", tpx, tpy - 12);
}

function cameraPose() {
  const yaw = (Number(config.start.yaw_deg || 0) * Math.PI) / 180;
  const forwardOffset = Number(config.robot.camera_x || 0) - Number(config.robot.control_x || 0);
  const leftOffset = Number(config.robot.camera_y || 0) - Number(config.robot.control_y || 0);
  const forward = [Math.cos(yaw), Math.sin(yaw)];
  const left = [-Math.sin(yaw), Math.cos(yaw)];
  return {
    xy: [
      config.start.xy[0] + forward[0] * forwardOffset + left[0] * leftOffset,
      config.start.xy[1] + forward[1] * forwardOffset + left[1] * leftOffset,
    ],
  };
}

function robotSummaryHtml() {
  const robot = config.robot || {};
  const obstacle = config.obstacle || {};
  return [
    `preset: <code>${robot.preset || "go2"}</code>`,
    `size: <code>${Number(robot.length || 0).toFixed(2)} x ${Number(robot.width || 0).toFixed(2)} m</code>`,
    `camera/control: <code>${Number(robot.camera_x || 0).toFixed(2)} / ${Number(robot.control_x || 0).toFixed(2)} m</code>`,
    `safety radius: <code>${Number(robot.safety_radius || 0).toFixed(2)} m</code>`,
    `obstacle dilation: <code>${Number(obstacle.dilation_cells || 0)}</code>`,
  ].join("<br />");
}

function drawScene() {
  if (!config) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid(ctx, canvas, worldToCanvas);
  drawEsdfSceneOverlay();
  drawObjects(ctx, worldToCanvas, true);
  drawStartTarget(ctx, worldToCanvas);
  drawRealtimeOverlay();
}

function gridCellToCanvasRect(payload, ix, iy) {
  const resolution = Number(payload.resolution);
  const origin = payload.origin;
  const x0 = origin[0] + ix * resolution;
  const y0 = origin[1] + iy * resolution;
  const x1 = origin[0] + (ix + 1) * resolution;
  const y1 = origin[1] + (iy + 1) * resolution;
  const [px0, py0] = worldToCanvas(x0, y0);
  const [px1, py1] = worldToCanvas(x1, y1);
  return [Math.min(px0, px1), Math.min(py0, py1), Math.abs(px1 - px0), Math.abs(py1 - py0)];
}

function esdfColor(value, obstacleValue) {
  const clearance = Math.max(0, Math.min(1, value / 255));
  if (obstacleValue > 0) return "rgba(239, 68, 68, 0.62)";
  if (clearance < 0.18) {
    const t = clearance / 0.18;
    return `rgba(${Math.round(249 - 80 * t)}, ${Math.round(115 + 80 * t)}, ${Math.round(22 + 20 * t)}, 0.46)`;
  }
  if (clearance < 0.45) {
    const t = (clearance - 0.18) / 0.27;
    return `rgba(${Math.round(169 - 80 * t)}, ${Math.round(195 + 35 * t)}, ${Math.round(42 + 85 * t)}, 0.28)`;
  }
  return "rgba(15, 22, 33, 0.10)";
}

function drawEsdfSceneOverlay() {
  const esdf = currentFrame?.esdf_u8;
  if (!esdf || !esdf.data) return;
  const obstacle = currentFrame?.obstacle_u8?.data || [];
  ctx.save();
  for (let ix = 0; ix < esdf.height; ix += 1) {
    for (let iy = 0; iy < esdf.width; iy += 1) {
      const idx = ix * esdf.width + iy;
      const value = esdf.data[idx];
      const obstacleValue = obstacle[idx] || 0;
      if (value > 150 && obstacleValue === 0) continue;
      const [x, y, w, h] = gridCellToCanvasRect(esdf, ix, iy);
      ctx.fillStyle = esdfColor(value, obstacleValue);
      ctx.fillRect(x, y, Math.max(w, 1), Math.max(h, 1));
    }
  }
  drawEsdfLegend();
  ctx.restore();
}

function drawEsdfLegend() {
  const x = canvas.width - 178;
  const y = 16;
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.fillRect(x, y, 154, 58);
  ctx.fillStyle = "#17202a";
  ctx.font = "12px system-ui";
  ctx.fillText("ESDF clearance", x + 12, y + 18);
  const colors = [
    "rgba(239, 68, 68, 0.72)",
    "rgba(249, 115, 22, 0.62)",
    "rgba(89, 230, 127, 0.36)",
    "rgba(15, 22, 33, 0.14)",
  ];
  colors.forEach((color, index) => {
    ctx.fillStyle = color;
    ctx.fillRect(x + 12 + index * 30, y + 28, 30, 12);
  });
  ctx.fillStyle = "#5c6975";
  ctx.fillText("near", x + 12, y + 53);
  ctx.fillText("clear", x + 106, y + 53);
}

function drawPath(points, color, width, alpha = 1) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    const [px, py] = worldToCanvas(x, y);
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.restore();
}

function drawRealtimeOverlay() {
  if (!currentFrame) {
    ctx.fillStyle = "#5c6975";
    ctx.font = "15px system-ui";
    ctx.fillText("Click Realtime to start live planning.", 22, 30);
    return;
  }

  drawPath(realtimePath, "#0f766e", 4, 1);
  (currentFrame.candidate_trajectories_xy || []).forEach((path) => drawPath(path, "#8d99a6", 1.1, 0.32));
  drawPath(currentFrame.selected_trajectory_xy, "#00a9c9", 3, 0.95);
  drawFootprint(currentFrame.robot_footprint_xy || []);
  drawHeadingArrow(ctx, worldToCanvas, currentFrame.robot_xy, currentFrame.robot_yaw_deg, "#003f4a");
  if (currentFrame.robot_xy) {
    const [rx, ry] = worldToCanvas(currentFrame.robot_xy[0], currentFrame.robot_xy[1]);
    ctx.fillStyle = "#003f4a";
    ctx.beginPath();
    ctx.arc(rx, ry, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillRect(12, 12, 210, currentFrame.should_reverse || currentFrame.selected_is_reverse ? 92 : 72);
  ctx.fillStyle = "#17202a";
  ctx.font = "13px system-ui";
  ctx.fillText(`realtime tick ${Math.max(0, realtimePath.length - 1)}`, 24, 34);
  ctx.fillText(`cmd [${currentFrame.selected_param.map((v) => Number(v).toFixed(2)).join(", ")}]`, 24, 54);
  ctx.fillText(`clearance ${Number(currentFrame.front_clearance).toFixed(2)}m`, 24, 74);
  if (currentFrame.should_reverse || currentFrame.selected_is_reverse) {
    ctx.fillText(`reverse ${currentFrame.selected_is_reverse ? "selected" : "gated"}`, 24, 94);
  }
}

function drawDiagnostic(step = 0) {
  if (!currentFrame) {
    diagnosticNote.textContent = "Run the closed-loop simulation to generate per-step perception snapshots.";
    return;
  }
  diagnosticNote.textContent = "Depth updates here; ESDF safety layer is fused into the scene canvas.";
}

function colorize(value, mode) {
  const t = Math.max(0, Math.min(1, value / 255));
  if (mode === "depth") {
    return [
      Math.round(30 + 220 * t),
      Math.round(20 + 120 * (1 - Math.abs(t - 0.5) * 2)),
      Math.round(70 + 160 * (1 - t)),
    ];
  }
  return [
    Math.round(15 + 25 * t),
    Math.round(22 + 55 * t),
    Math.round(33 + 70 * t),
  ];
}

function drawU8Canvas(targetCanvas, targetCtx, payload, mode) {
  if (!payload || !payload.data) {
    targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
    return;
  }
  const image = targetCtx.createImageData(payload.width, payload.height);
  for (let i = 0; i < payload.data.length; i += 1) {
    const [r, g, b] = colorize(payload.data[i], mode);
    const j = i * 4;
    image.data[j] = r;
    image.data[j + 1] = g;
    image.data[j + 2] = b;
    image.data[j + 3] = 255;
  }
  const offscreen = document.createElement("canvas");
  offscreen.width = payload.width;
  offscreen.height = payload.height;
  const offscreenCtx = offscreen.getContext("2d");
  offscreenCtx.putImageData(image, 0, 0);
  targetCtx.imageSmoothingEnabled = false;
  targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetCtx.drawImage(offscreen, 0, 0, targetCanvas.width, targetCanvas.height);
}

function drawRealtimeFrame(frame) {
  currentFrame = frame;
  drawU8Canvas(depthCanvas, depthCtx, frame.depth_u8, "depth");
  drawScene();
  const esdfState = frame.esdf_u8 ? "ESDF live" : "waiting for ESDF";
  diagnosticNote.textContent = `${esdfState}; depth updates here. Valid trajectories: ${frame.valid_trajectories}, clearance: ${Number(frame.front_clearance).toFixed(2)}m, reverse: ${frame.selected_is_reverse ? "selected" : frame.should_reverse ? "gated" : "no"}, recovery: ${frame.recovery_reason || "normal"}.`;
}

function drawHeadingArrow(drawCtx, mapper, xy, yawDeg, color, length = 0.45) {
  if (!xy || yawDeg === undefined || yawDeg === null) return;
  const yaw = (Number(yawDeg) * Math.PI) / 180;
  const tip = [xy[0] + Math.cos(yaw) * length, xy[1] + Math.sin(yaw) * length];
  const [x0, y0] = mapper(xy[0], xy[1]);
  const [x1, y1] = mapper(tip[0], tip[1]);
  drawCtx.strokeStyle = color;
  drawCtx.fillStyle = color;
  drawCtx.lineWidth = 3;
  drawCtx.beginPath();
  drawCtx.moveTo(x0, y0);
  drawCtx.lineTo(x1, y1);
  drawCtx.stroke();
  const angle = Math.atan2(y1 - y0, x1 - x0);
  drawCtx.beginPath();
  drawCtx.moveTo(x1, y1);
  drawCtx.lineTo(x1 - Math.cos(angle - 0.55) * 12, y1 - Math.sin(angle - 0.55) * 12);
  drawCtx.lineTo(x1 - Math.cos(angle + 0.55) * 12, y1 - Math.sin(angle + 0.55) * 12);
  drawCtx.closePath();
  drawCtx.fill();
}

function drawFootprint(footprint) {
  if (footprint.length) {
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
}

function refreshFields() {
  const obj = selectedObject();
  fields.targetX.value = config.target[0];
  fields.targetY.value = config.target[1];
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
  drawScene();
}

function applyFieldChanges() {
  const obj = selectedObject();
  config.target[0] = Number(fields.targetX.value);
  config.target[1] = Number(fields.targetY.value);
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

function hitTestObject(px, py) {
  for (let i = config.objects.length - 1; i >= 0; i -= 1) {
    const obj = config.objects[i];
    const [x, y] = obj.center;
    const [sx, sy] = obj.size;
    const [px0, py0] = worldToCanvas(x - sx / 2, y - sy / 2);
    const [px1, py1] = worldToCanvas(x + sx / 2, y + sy / 2);
    if (px >= Math.min(px0, px1) && px <= Math.max(px0, px1) &&
        py >= Math.min(py0, py1) && py <= Math.max(py0, py1)) {
      return i;
    }
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
    if (realtimeRunning) {
      scheduleRealtimeTick(realtimePending ? 0 : REALTIME_TICK_DELAY_MS);
    }
  }
}

async function startRosLoop() {
  const response = await fetch("/api/start-ros-loop", { method: "POST" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to start ROS loop");
  return data;
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
    await startRosLoop();
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
  const response = await fetch("/api/default-config");
  config = await response.json();
  selectedIndex = 0;
  currentFrame = null;
  realtimePath = [];
  refreshFields();
  statusEl.textContent = "Ready";
}

Object.values(fields).forEach((field) => {
  if (field !== fields.configText && typeof field.addEventListener === "function") {
    field.addEventListener("change", () => {
      if (field === fields.targetX || field === fields.targetY || field === fields.objectName || field === fields.objectX || field === fields.objectY || field === fields.objectSX || field === fields.objectSY || field === fields.objectSZ) {
        applyFieldChanges();
      }
      noteSceneEdited(true, true);
    });
  }
});

realtimeButton.addEventListener("click", toggleRealtime);
document.getElementById("resetScene").addEventListener("click", loadDefault);
document.getElementById("applyJson").addEventListener("click", () => {
  const nextConfig = JSON.parse(fields.configText.value);
  nextConfig.robot = config.robot;
  nextConfig.obstacle = config.obstacle;
  config = nextConfig;
  selectedIndex = Math.min(selectedIndex, config.objects.length - 1);
  currentFrame = null;
  refreshFields();
  noteSceneEdited(true, true);
});
document.getElementById("addBox").addEventListener("click", () => {
  config.objects.push({
    name: `box_${config.objects.length + 1}`,
    kind: "box",
    center: [2.0, 0.0, 0.35],
    size: [0.5, 0.5, 0.8],
  });
  selectedIndex = config.objects.length - 1;
  refreshFields();
  noteSceneEdited(true, true);
});
document.getElementById("duplicateObject").addEventListener("click", () => {
  const obj = selectedObject();
  if (!obj) return;
  const copy = JSON.parse(JSON.stringify(obj));
  copy.name = `${copy.name}_copy`;
  copy.center[1] += 0.4;
  config.objects.push(copy);
  selectedIndex = config.objects.length - 1;
  refreshFields();
  noteSceneEdited(true, true);
});
document.getElementById("deleteObject").addEventListener("click", () => {
  if (!config.objects.length) return;
  config.objects.splice(selectedIndex, 1);
  selectedIndex = Math.max(0, selectedIndex - 1);
  refreshFields();
  noteSceneEdited(true, true);
});

canvas.addEventListener("pointerdown", (event) => {
  const rect = canvas.getBoundingClientRect();
  const px = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const py = ((event.clientY - rect.top) / rect.height) * canvas.height;
  if (hitTestTarget(px, py)) {
    const [wx, wy] = canvasToWorld(px, py);
    dragState = { type: "target", dx: config.target[0] - wx, dy: config.target[1] - wy };
    canvas.setPointerCapture(event.pointerId);
    return;
  }
  const hit = hitTestObject(px, py);
  if (hit >= 0) {
    selectedIndex = hit;
    const [wx, wy] = canvasToWorld(px, py);
    const obj = selectedObject();
    dragState = { type: "object", dx: obj.center[0] - wx, dy: obj.center[1] - wy };
    canvas.setPointerCapture(event.pointerId);
    refreshFields();
  }
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragState) return;
  const rect = canvas.getBoundingClientRect();
  const px = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const py = ((event.clientY - rect.top) / rect.height) * canvas.height;
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
