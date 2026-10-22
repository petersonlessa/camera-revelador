"use strict";
// Câmera Revelador — foto e vídeo com as LUTs .cube do Revelador, aplicadas na GPU.
// A LUT é interpolada no shader com a mesma conta trilinear do Revelador
// (8 leituras exatas da tabela, sem o filtro da placa), para a câmera dar o
// mesmo resultado que o .exe.

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const TEST = params.has("teste");

// ---------------------------------------------------------------- .cube
function parseCube(text, name) {
  let size = 0;
  const vals = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line[0] === "#") continue;
    const parts = line.split(/\s+/);
    const head = parts[0].toUpperCase();
    if (head === "LUT_3D_SIZE") { size = parseInt(parts[1], 10); continue; }
    if (head === "TITLE" || head === "DOMAIN_MIN" || head === "DOMAIN_MAX" ||
        head === "LUT_3D_INPUT_RANGE") continue;
    if (head === "LUT_1D_SIZE") throw new Error("LUT 1D não é suportada (use uma LUT 3D).");
    if (parts.length >= 3) {
      const r = parseFloat(parts[0]);
      if (!Number.isNaN(r)) vals.push(r, parseFloat(parts[1]), parseFloat(parts[2]));
    }
  }
  if (size < 2 || size > 129) throw new Error("LUT_3D_SIZE ausente ou inválido.");
  if (vals.length !== size * size * size * 3)
    throw new Error(`O arquivo tem ${vals.length / 3} cores; esperava ${size ** 3}.`);
  return { name, size, data: Float32Array.from(vals) };
}

// ---------------------------------------------------------------- WebGL
const VERT = `#version 300 es
in vec2 aPos;
uniform vec4 uCrop;      // x0, y0, largura, altura na textura (0..1)
uniform bool uMirror;
out vec2 vUv;
void main() {
  vec2 uv = aPos * 0.5 + 0.5;
  uv.y = 1.0 - uv.y;
  if (uMirror) uv.x = 1.0 - uv.x;
  vUv = uCrop.xy + uv * uCrop.zw;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

// Trilinear igual ao interp_table do Revelador: interpola em R, depois G, depois B.
const FRAG = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler3D;
uniform sampler2D uFrame;
uniform sampler3D uLut;
uniform int uSize;
uniform float uMix;
in vec2 vUv;
out vec4 outColor;
vec3 L(ivec3 p) { return texelFetch(uLut, p, 0).rgb; }
void main() {
  vec3 c = texture(uFrame, vUv).rgb;
  if (uSize < 2 || uMix <= 0.0) { outColor = vec4(c, 1.0); return; }
  vec3 x = clamp(c, 0.0, 1.0) * float(uSize - 1);
  ivec3 i = min(ivec3(x), ivec3(uSize - 2));
  vec3 f = x - vec3(i);
  vec3 c000 = L(i), c100 = L(i + ivec3(1, 0, 0));
  vec3 c010 = L(i + ivec3(0, 1, 0)), c110 = L(i + ivec3(1, 1, 0));
  vec3 c001 = L(i + ivec3(0, 0, 1)), c101 = L(i + ivec3(1, 0, 1));
  vec3 c011 = L(i + ivec3(0, 1, 1)), c111 = L(i + ivec3(1, 1, 1));
  vec3 c00 = c000 + (c100 - c000) * f.r;
  vec3 c10 = c010 + (c110 - c010) * f.r;
  vec3 c01 = c001 + (c101 - c001) * f.r;
  vec3 c11 = c011 + (c111 - c011) * f.r;
  vec3 c0 = c00 + (c10 - c00) * f.g;
  vec3 c1 = c01 + (c11 - c01) * f.g;
  vec3 g = c0 + (c1 - c0) * f.b;
  // Arredonda para 8 bits do mesmo jeito que o Revelador (piso de v*255 + 0.5).
  vec3 v = clamp(c + (g - c) * uMix, 0.0, 1.0);
  outColor = vec4(floor(v * 255.0 + 0.5) / 255.0, 1.0);
}`;

class Renderer {
  constructor(canvas, smooth) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false,
      premultipliedAlpha: false, alpha: false });
    if (!gl) throw new Error("Este navegador não tem WebGL2.");
    this.gl = gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    this.u = {};
    for (const n of ["uCrop", "uMirror", "uFrame", "uLut", "uSize", "uMix"]) this.u[n] = gl.getUniformLocation(prog, n);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    // Valores crus da foto/vídeo, sem gestão de cor do navegador (como o Revelador lê).
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

    this.frameTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    const filt = smooth ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.lutTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_3D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R])
      gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    gl.uniform1i(this.u.uFrame, 0);
    gl.uniform1i(this.u.uLut, 1);
    this.lutSize = 0;
  }

  setLut(lut) {
    const gl = this.gl;
    if (!lut) { this.lutSize = 0; return; }
    const n = lut.size, rgba = new Float32Array(n * n * n * 4);
    for (let i = 0, j = 0; i < lut.data.length; i += 3, j += 4) {
      rgba[j] = lut.data[i]; rgba[j + 1] = lut.data[i + 1]; rgba[j + 2] = lut.data[i + 2]; rgba[j + 3] = 1;
    }
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    // Ordem .cube = R mais rápido, depois G, depois B = x, y, z da textura 3D.
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, n, n, n, 0, gl.RGBA, gl.FLOAT, rgba);
    this.lutSize = n;
  }

  resize(w, h) {
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  smooth(on) {
    // NEAREST quando a saída tem o tamanho da fonte (pixels exatos); LINEAR ao reduzir.
    const gl = this.gl, f = on ? gl.LINEAR : gl.NEAREST;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
  }

  draw(source, { crop = [0, 0, 1, 1], mirror = false, mix = 1 } = {}) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.uniform4f(this.u.uCrop, crop[0], crop[1], crop[2], crop[3]);
    gl.uniform1i(this.u.uMirror, mirror ? 1 : 0);
    gl.uniform1i(this.u.uSize, this.lutSize);
    gl.uniform1f(this.u.uMix, mix);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

// ---------------------------------------------------------------- LUTs guardadas
const DB_NAME = "camera-revelador";
function db() {
  return new Promise((ok, fail) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore("luts", { keyPath: "id" });
    req.onsuccess = () => ok(req.result);
    req.onerror = () => fail(req.error);
  });
}
async function dbAll() {
  try {
    const d = await db();
    return await new Promise((ok, fail) => {
      const r = d.transaction("luts").objectStore("luts").getAll();
      r.onsuccess = () => ok(r.result);
      r.onerror = () => fail(r.error);
    });
  } catch { return []; }
}
async function dbPut(item) {
  const d = await db();
  return new Promise((ok, fail) => {
    const t = d.transaction("luts", "readwrite");
    t.objectStore("luts").put(item);
    t.oncomplete = ok;
    t.onerror = () => fail(t.error);
  });
}
async function dbDel(id) {
  const d = await db();
  return new Promise((ok) => {
    const t = d.transaction("luts", "readwrite");
    t.objectStore("luts").delete(id);
    t.oncomplete = ok;
  });
}

const BUILTIN = [
  { id: "none", name: "Sem LUT" },
  { id: "raizen-33", name: "Raizen", url: "luts/raizen-33.cube" },
  { id: "raizen-pb-33", name: "Raizen P&B", url: "luts/raizen-pb-33.cube" },
];

// ---------------------------------------------------------------- estado
const state = {
  facing: "user", mode: "foto", light: false, kelvin: 5000, ring: 18, mirror: true,
  mix: 1, lutId: "raizen-33", recording: false, compare: false, last: null,
};
const luts = new Map();      // id -> {id, name, size, data, builtin}
let view, out, stream = null, recorder = null, chunks = [], recStart = 0, audioStream = null;
const cam = $("#cam");
let testImage = null;

function msg(text, ms = 2600) {
  const el = $("#msg");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(msg.t);
  if (ms) msg.t = setTimeout(() => (el.hidden = true), ms);
}

function kelvinRGB(k) {
  // Cor aproximada de um corpo negro (Tanner Helland), para a luz da tela.
  const t = k / 100;
  let r, g, b;
  if (t <= 66) { r = 255; g = 99.47 * Math.log(t) - 161.12; }
  else { r = 329.7 * Math.pow(t - 60, -0.1332); g = 288.12 * Math.pow(t - 60, -0.0755); }
  if (t >= 66) b = 255; else if (t <= 19) b = 0; else b = 138.52 * Math.log(t - 10) - 305.04;
  const c = (v) => Math.round(Math.min(255, Math.max(0, v)));
  return `rgb(${c(r)}, ${c(g)}, ${c(b)})`;
}

function applyLight() {
  const color = kelvinRGB(state.kelvin);
  document.documentElement.style.setProperty("--light", color);
  document.documentElement.style.setProperty("--ring", state.ring + "%");
  const front = state.facing === "user";
  $("#stage").classList.toggle("ring", state.light && front && state.mode === "video");
  $("#b-flash").classList.toggle("on", state.light);
  $("#kval").textContent = state.kelvin + " K";
  $("#rval").textContent = state.ring + "%";
  resizeView();
}

// ---------------------------------------------------------------- câmera
async function startCamera() {
  if (TEST) return;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: state.facing, width: { ideal: 3840 }, height: { ideal: 2160 },
               frameRate: { ideal: 30 } },
    });
  } catch (e) {
    msg("Sem acesso à câmera. Permita a câmera para este site nos Ajustes do Safari. (" + e.name + ")", 0);
    return;
  }
  cam.srcObject = stream;
  await cam.play().catch(() => {});
  resizeView();
}

function sourceSize() {
  if (TEST && testImage) return [testImage.naturalWidth, testImage.naturalHeight];
  return [cam.videoWidth || 0, cam.videoHeight || 0];
}
function source() { return TEST ? testImage : cam; }
function ready() {
  return TEST ? !!(testImage && testImage.complete) : cam.readyState >= 2 && cam.videoWidth > 0;
}

function resizeView() {
  const c = view.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = c.getBoundingClientRect();
  view.resize(Math.max(1, Math.round(r.width * dpr)), Math.max(1, Math.round(r.height * dpr)));
}

function coverCrop(sw, sh, dw, dh) {
  // Preenche o visor sem distorcer (corta o que sobra).
  const s = sw / sh, d = dw / dh;
  if (s > d) { const w = d / s; return [(1 - w) / 2, 0, w, 1]; }
  const h = s / d;
  return [0, (1 - h) / 2, 1, h];
}

const mirrored = () => state.mirror && state.facing === "user";
const mixNow = () => (state.compare ? 0 : state.mix);
const VIDEO_MAX = 1920;   // vídeo pelo visor: até 1920 px no lado maior (4K engasga no Safari)

function videoSize() {
  const [sw, sh] = sourceSize();
  const k = Math.min(1, VIDEO_MAX / Math.max(sw, sh));
  return [Math.round((sw * k) / 2) * 2, Math.round((sh * k) / 2) * 2];
}

function frame() {
  if (ready()) {
    const [sw, sh] = sourceSize();
    view.draw(source(), { crop: coverCrop(sw, sh, view.canvas.width, view.canvas.height),
                          mirror: mirrored(), mix: mixNow() });
    if (state.recording) {
      out.draw(source(), { mirror: mirrored(), mix: state.mix });
    }
  }
  if (!TEST && cam.requestVideoFrameCallback) cam.requestVideoFrameCallback(frame);
  else requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- LUT ativa
async function loadLut(id) {
  if (id === "none") return null;
  if (luts.has(id) && luts.get(id).data) return luts.get(id);
  const b = BUILTIN.find((x) => x.id === id);
  if (!b) return null;
  const text = await (await fetch(b.url)).text();
  const lut = { ...parseCube(text, b.name), id, builtin: true };
  luts.set(id, lut);
  return lut;
}

async function selectLut(id) {
  try {
    const lut = await loadLut(id);
    state.lutId = id;
    view.setLut(lut);
    out.setLut(lut);
    try { localStorage.setItem("lut", id); } catch {}
    renderChips();
  } catch (e) {
    msg("Não foi possível abrir a LUT: " + e.message);
  }
}

function renderChips() {
  const box = $("#luts");
  box.textContent = "";
  const items = [...BUILTIN, ...[...luts.values()].filter((l) => !l.builtin)];
  for (const item of items) {
    const b = document.createElement("button");
    b.className = "chip" + (item.id === state.lutId ? " on" : "");
    b.textContent = item.name;
    if (!BUILTIN.find((x) => x.id === item.id) && item.id === state.lutId) {
      const x = document.createElement("span");
      x.className = "x";
      x.textContent = "✕";
      x.onclick = async (e) => {
        e.stopPropagation();
        await dbDel(item.id);
        luts.delete(item.id);
        await selectLut("raizen-33");
        msg(`"${item.name}" removida.`);
      };
      b.appendChild(x);
    }
    b.onclick = () => selectLut(item.id);
    box.appendChild(b);
  }
  const add = document.createElement("button");
  add.className = "chip";
  add.textContent = "+ .cube";
  add.onclick = () => $("#file").click();
  box.appendChild(add);
}

async function importFiles(files) {
  let last = null;
  for (const f of files) {
    try {
      const name = f.name.replace(/\.cube$/i, "");
      const lut = parseCube(await f.text(), name);
      const item = { id: "u-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6),
                     name, size: lut.size, data: lut.data };
      await dbPut(item);
      luts.set(item.id, item);
      last = item.id;
    } catch (e) {
      msg(`${f.name}: ${e.message}`, 4000);
    }
  }
  if (last) { await selectLut(last); msg("LUT importada e guardada neste iPhone."); }
}

// ---------------------------------------------------------------- foto e vídeo
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const stamp = () => new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);

async function takePhoto() {
  if (!ready()) return;
  const flash = state.light && state.facing === "user";
  if (flash) { $("#flash").classList.add("on"); await sleep(450); }   // a câmera se ajusta à luz da tela
  const [sw, sh] = sourceSize();
  out.resize(sw, sh);
  out.smooth(false);
  out.draw(source(), { mirror: mirrored(), mix: state.mix });
  if (flash) $("#flash").classList.remove("on");
  const blob = await new Promise((r) => out.canvas.toBlob(r, "image/jpeg", 0.95));
  showResult(blob, `Revelador_${stamp()}.jpg`, "image/jpeg");
}

async function startVideo() {
  if (!ready() || !window.MediaRecorder || !out.canvas.captureStream) {
    msg("Este navegador não grava vídeo do visor.");
    return;
  }
  const [vw, vh] = videoSize();
  out.resize(vw, vh);
  out.smooth(vw !== sourceSize()[0]);
  out.draw(source(), { mirror: mirrored(), mix: state.mix });
  const tracks = [...out.canvas.captureStream(30).getVideoTracks()];
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    tracks.push(...audioStream.getAudioTracks());
  } catch { msg("Gravando sem som (microfone não permitido)."); }
  const types = ["video/mp4;codecs=avc1.640028,mp4a.40.2", "video/mp4;codecs=avc1", "video/mp4",
                 "video/webm;codecs=vp9,opus", "video/webm"];
  const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
  recorder = new MediaRecorder(new MediaStream(tracks), { mimeType, videoBitsPerSecond: 16_000_000 });
  chunks = [];
  recorder.ondataavailable = (e) => e.data && e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const type = recorder.mimeType || "video/mp4";
    const ext = type.includes("webm") ? "webm" : "mp4";
    showResult(new Blob(chunks, { type }), `Revelador_${stamp()}.${ext}`, type);
    if (audioStream) audioStream.getTracks().forEach((t) => t.stop());
    audioStream = null;
  };
  recorder.start(1000);
  state.recording = true;
  recStart = performance.now();
  $("#shutter").classList.add("recording");
  $("#timer").hidden = false;
  tick();
}

function stopVideo() {
  state.recording = false;
  $("#shutter").classList.remove("recording");
  $("#timer").hidden = true;
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

function tick() {
  if (!state.recording) return;
  const s = Math.floor((performance.now() - recStart) / 1000);
  $("#timer").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  setTimeout(tick, 250);
}

function showResult(blob, name, type) {
  state.last = { blob, name, type };
  const url = URL.createObjectURL(blob);
  const isVideo = type.startsWith("video");
  $("#thumb").style.backgroundImage = isVideo ? "" : `url(${url})`;
  $("#preview-img").hidden = isVideo;
  $("#preview-vid").hidden = !isVideo;
  if (isVideo) $("#preview-vid").src = url; else $("#preview-img").src = url;
  $("#result").hidden = false;
}

async function saveLast() {
  const it = state.last;
  if (!it) return;
  const file = new File([it.blob], it.name, { type: it.type });
  try {
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file] });   // "Salvar Imagem"/"Salvar Vídeo" vai para a Fotos
      return;
    }
  } catch (e) {
    if (e.name === "AbortError") return;
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(it.blob);
  a.download = it.name;
  a.click();
}

// ---------------------------------------------------------------- interface
function bindUi() {
  $("#shutter").onclick = () => {
    if (state.mode === "foto") takePhoto();
    else if (state.recording) stopVideo();
    else startVideo();
  };
  document.querySelectorAll(".mode").forEach((b) => (b.onclick = () => {
    if (state.recording) return;
    state.mode = b.dataset.mode;
    document.querySelectorAll(".mode").forEach((x) => x.classList.toggle("on", x === b));
    $("#shutter").classList.toggle("video", state.mode === "video");
    applyLight();
  }));
  $("#b-flip").onclick = async () => {
    if (state.recording) return;
    state.facing = state.facing === "user" ? "environment" : "user";
    applyLight();
    await startCamera();
  };
  $("#b-flash").onclick = () => {
    state.light = !state.light;
    applyLight();
    if (state.light && state.facing !== "user") msg("A luz da tela serve para a câmera frontal.");
    else if (state.light) msg("Luz ligada. Brilho do iPhone no máximo!");
  };
  $("#b-cfg").onclick = () => ($("#cfg").hidden = false);
  $("#b-cfg-close").onclick = () => ($("#cfg").hidden = true);
  $("#kelvin").oninput = (e) => { state.kelvin = +e.target.value; applyLight(); };
  $("#ringsize").oninput = (e) => { state.ring = +e.target.value; applyLight(); };
  $("#mirror").onchange = (e) => { state.mirror = e.target.checked; };
  $("#mix").oninput = (e) => { state.mix = +e.target.value / 100; $("#mixval").textContent = e.target.value + "%"; };
  $("#file").onchange = (e) => { importFiles([...e.target.files]); e.target.value = ""; };
  $("#thumb").onclick = () => { if (state.last) $("#result").hidden = false; };
  $("#b-save").onclick = saveLast;
  $("#b-result-close").onclick = () => ($("#result").hidden = true);
  // Segurar o dedo no visor mostra a imagem sem a LUT (antes/depois).
  const cv = $("#view");
  cv.addEventListener("pointerdown", () => (state.compare = true));
  for (const ev of ["pointerup", "pointercancel", "pointerleave"]) cv.addEventListener(ev, () => (state.compare = false));
  window.addEventListener("resize", resizeView);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !TEST && (!stream || !stream.active)) startCamera();
    keepAwake();
  });
}

async function keepAwake() {
  try { if (navigator.wakeLock && document.visibilityState === "visible") await navigator.wakeLock.request("screen"); } catch {}
}

// ---------------------------------------------------------------- teste (?teste=1)
// Usa a amostra no lugar da câmera e compara o shader com o Revelador (Python).
async function verify(id = "raizen-33") {
  await selectLut(id);
  const [w, h] = sourceSize();
  out.resize(w, h);
  out.smooth(false);
  out.draw(testImage, { mirror: false, mix: 1 });
  const gl = out.gl, px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const ref = await new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = "teste/ref-raizen-33.png"; });
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { colorSpace: "srgb" });
  ctx.drawImage(ref, 0, 0);
  const rp = ctx.getImageData(0, 0, w, h).data;
  let diff = 0, maxd = 0, off = 0;
  for (let y = 0; y < h; y++) {
    const gy = h - 1 - y;     // readPixels vem de baixo para cima
    for (let x = 0; x < w; x++) {
      for (let k = 0; k < 3; k++) {
        const d = Math.abs(px[(gy * w + x) * 4 + k] - rp[(y * w + x) * 4 + k]);
        diff += d; if (d > maxd) maxd = d; if (d) off++;
      }
    }
  }
  return { largura: w, altura: h, diferencaMedia: diff / (w * h * 3), diferencaMaxima: maxd,
           canaisDiferentes: off, totalCanais: w * h * 3 };
}

// ---------------------------------------------------------------- início
async function main() {
  view = new Renderer($("#view"), true);
  out = new Renderer(document.createElement("canvas"), false);
  bindUi();
  for (const item of await dbAll()) luts.set(item.id, { ...item, data: Float32Array.from(item.data) });
  let saved = "raizen-33";
  try { saved = localStorage.getItem("lut") || saved; } catch {}
  if (!BUILTIN.find((b) => b.id === saved) && !luts.has(saved)) saved = "raizen-33";
  if (TEST) {
    testImage = await new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = "teste/amostra.png"; });
    window.__verificar = verify;
    window.__estado = state;
  }
  await selectLut(saved);
  applyLight();
  resizeView();
  await startCamera();
  keepAwake();
  frame();
}

main().catch((e) => msg("Erro ao iniciar: " + e.message, 0));
