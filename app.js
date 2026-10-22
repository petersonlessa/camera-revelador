"use strict";
// Câmera Revelador — foto e vídeo com as LUTs .cube do Revelador, aplicadas na GPU.
// A LUT é interpolada no shader com a mesma conta trilinear do Revelador
// (8 leituras exatas da tabela, sem o filtro da placa) e o mesmo arredondamento,
// para a câmera dar o mesmo resultado que o .exe.
// Foto: HDR+ por rajada (vários quadros alinhados e somados, menos ruído),
// Definição e Nitidez. Com os três desligados, a foto é exatamente o .cube.

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

// ---------------------------------------------------------------- GLSL
const VERT = `#version 300 es
in vec2 aPos;
uniform vec4 uCrop;      // x0, y0, largura, altura na textura (0..1)
uniform bool uMirror;
uniform int uOrient;     // orientação EXIF da foto gravada (1 = já de pé)
out vec2 vUv;
// De um ponto da foto de pé (d, y para baixo) para o ponto na foto como foi gravada.
vec2 orient(vec2 d) {
  if (uOrient == 2) return vec2(1.0 - d.x, d.y);
  if (uOrient == 3) return vec2(1.0 - d.x, 1.0 - d.y);
  if (uOrient == 4) return vec2(d.x, 1.0 - d.y);
  if (uOrient == 5) return d.yx;
  if (uOrient == 6) return vec2(d.y, 1.0 - d.x);
  if (uOrient == 7) return vec2(1.0 - d.y, 1.0 - d.x);
  if (uOrient == 8) return vec2(1.0 - d.y, d.x);
  return d;
}
void main() {
  vec2 uv = aPos * 0.5 + 0.5;
  uv.y = 1.0 - uv.y;
  if (uMirror) uv.x = 1.0 - uv.x;
  vUv = uCrop.xy + orient(uv) * uCrop.zw;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

// Trilinear igual ao interp_table do Revelador: interpola em R, depois G, depois B.
const LUT_GLSL = `
uniform highp sampler3D uLut;
uniform int uSize;
vec3 lutAt(ivec3 p) { return texelFetch(uLut, p, 0).rgb; }
vec3 applyLut(vec3 c) {
  if (uSize < 2) return c;
  vec3 x = clamp(c, 0.0, 1.0) * float(uSize - 1);
  ivec3 i = min(ivec3(x), ivec3(uSize - 2));
  vec3 f = x - vec3(i);
  vec3 c000 = lutAt(i), c100 = lutAt(i + ivec3(1, 0, 0));
  vec3 c010 = lutAt(i + ivec3(0, 1, 0)), c110 = lutAt(i + ivec3(1, 1, 0));
  vec3 c001 = lutAt(i + ivec3(0, 0, 1)), c101 = lutAt(i + ivec3(1, 0, 1));
  vec3 c011 = lutAt(i + ivec3(0, 1, 1)), c111 = lutAt(i + ivec3(1, 1, 1));
  vec3 c00 = c000 + (c100 - c000) * f.r;
  vec3 c10 = c010 + (c110 - c010) * f.r;
  vec3 c01 = c001 + (c101 - c001) * f.r;
  vec3 c11 = c011 + (c111 - c011) * f.r;
  vec3 c0 = c00 + (c10 - c00) * f.g;
  vec3 c1 = c01 + (c11 - c01) * f.g;
  return c0 + (c1 - c0) * f.b;
}`;

const FRAG = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uFrame;
uniform float uMix;
${LUT_GLSL}
in vec2 vUv;
out vec4 outColor;
void main() {
  vec3 c = texture(uFrame, vUv).rgb;
  if (uSize < 2 || uMix <= 0.0) { outColor = vec4(c, 1.0); return; }
  // Arredonda para 8 bits do mesmo jeito que o Revelador (piso de v*255 + 0.5).
  vec3 v = clamp(c + (applyLut(c) - c) * uMix, 0.0, 1.0);
  outColor = vec4(floor(v * 255.0 + 0.5) / 255.0, 1.0);
}`;

function compile(gl, vs, fs) {
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(prog, 0, "aPos");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const u = {};
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(prog, i).name;
    u[name] = gl.getUniformLocation(prog, name);
  }
  return { prog, u };
}

function quad(gl) {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
}

function rawPixels(gl) {
  // Valores crus da foto/vídeo, sem gestão de cor do navegador (como o Revelador lê).
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
}

function makeTex(gl, filter) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function makeLutTex(gl) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, t);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_3D, p, gl.NEAREST);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R])
    gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
  return t;
}

function uploadLut(gl, tex, lut) {
  if (!lut) return 0;
  const n = lut.size, rgba = new Float32Array(n * n * n * 4);
  for (let i = 0, j = 0; i < lut.data.length; i += 3, j += 4) {
    rgba[j] = lut.data[i]; rgba[j + 1] = lut.data[i + 1]; rgba[j + 2] = lut.data[i + 2]; rgba[j + 3] = 1;
  }
  gl.bindTexture(gl.TEXTURE_3D, tex);
  // Ordem .cube = R mais rápido, depois G, depois B = x, y, z da textura 3D.
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, n, n, n, 0, gl.RGBA, gl.FLOAT, rgba);
  return n;
}

// ---------------------------------------------------------------- visor e vídeo
class Renderer {
  constructor(canvas, smooth) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false,
      premultipliedAlpha: false, alpha: false });
    if (!gl) throw new Error("Este navegador não tem WebGL2.");
    this.gl = gl;
    this.p = compile(gl, VERT, FRAG);
    gl.useProgram(this.p.prog);
    quad(gl);
    rawPixels(gl);
    gl.activeTexture(gl.TEXTURE0);
    this.frameTex = makeTex(gl, smooth ? gl.LINEAR : gl.NEAREST);
    gl.activeTexture(gl.TEXTURE1);
    this.lutTex = makeLutTex(gl);
    gl.uniform1i(this.p.u.uFrame, 0);
    gl.uniform1i(this.p.u.uLut, 1);
    this.lutSize = 0;
  }

  setLut(lut) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    this.lutSize = uploadLut(gl, this.lutTex, lut);
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

  draw(source, { crop = [0, 0, 1, 1], mirror = false, mix = 1, orient = 1 } = {}) {
    const gl = this.gl;
    gl.useProgram(this.p.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.uniform4f(this.p.u.uCrop, crop[0], crop[1], crop[2], crop[3]);
    gl.uniform1i(this.p.u.uMirror, mirror ? 1 : 0);
    gl.uniform1i(this.p.u.uOrient, orient);
    gl.uniform1i(this.p.u.uSize, this.lutSize);
    gl.uniform1f(this.p.u.uMix, mix);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

// ---------------------------------------------------------------- foto: HDR+, Definição, Nitidez
const QUAD_VS = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

// Soma um quadro alinhado ao acumulador (RGB*peso, peso). O peso cai onde o
// quadro discorda da referência (movimento), para não criar fantasmas.
const ACC_FS = `#version 300 es
precision highp float;
uniform sampler2D uFrame;
uniform sampler2D uRef;
uniform vec2 uShift;
uniform bool uMirror;
uniform bool uFirst;
uniform float uSigma;
in vec2 vUv;
out vec4 o;
vec2 src(vec2 uv) { vec2 t = vec2(uv.x, 1.0 - uv.y); return uMirror ? vec2(1.0 - t.x, t.y) : t; }
void main() {
  vec2 uv = src(vUv);
  vec3 r = texture(uRef, uv).rgb;
  if (uFirst) { o = vec4(r, 1.0); return; }
  vec2 s = uv + uShift;
  if (s.x < 0.0 || s.y < 0.0 || s.x > 1.0 || s.y > 1.0) { o = vec4(0.0); return; }
  vec3 c = texture(uFrame, s).rgb;
  float d = dot(abs(c - r), vec3(0.3, 0.59, 0.11));
  float w = exp(-(d * d) / (2.0 * uSigma * uSigma));
  o = vec4(c * w, w);
}`;

// Normaliza o acumulador e aplica a LUT na intensidade do slider.
const RESOLVE_FS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uAcc;
uniform float uMix;
${LUT_GLSL}
in vec2 vUv;
out vec4 o;
void main() {
  vec4 a = texture(uAcc, vUv);
  vec3 c = clamp(a.rgb / max(a.a, 1e-4), 0.0, 1.0);
  o = vec4(clamp(c + (applyLut(c) - c) * uMix, 0.0, 1.0), 1.0);
}`;

// Luminância reduzida (média de 4x4 pixels) para o desfoque largo da Definição.
const DOWN_FS = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uTexel;
in vec2 vUv;
out vec4 o;
float Y(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
void main() {
  float s = 0.0;
  for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++)
    s += Y(texture(uSrc, vUv + (vec2(float(i), float(j)) - 1.5) * uTexel).rgb);
  o = vec4(s / 16.0, 0.0, 0.0, 1.0);
}`;

const BLUR_FS = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uStep;
uniform float uSigma;
in vec2 vUv;
out vec4 o;
void main() {
  int r = int(ceil(3.0 * uSigma));
  float s = 0.0, ws = 0.0;
  for (int i = -32; i <= 32; i++) {
    if (i < -r || i > r) continue;
    float w = exp(-float(i * i) / (2.0 * uSigma * uSigma));
    s += texture(uSrc, vUv + float(i) * uStep).r * w;
    ws += w;
  }
  o = vec4(s / ws, 0.0, 0.0, 1.0);
}`;

// Definição (contraste de média frequência, limitado) + Nitidez (borda fina,
// com coring e trava de overshoot no mínimo/máximo 3x3): as mesmas contas dos
// ópticos do Revelador, na luminância, somadas igual nos 3 canais (sem mudar a cor).
const FINAL_FS = `#version 300 es
precision highp float;
uniform sampler2D uColor;
uniform sampler2D uBig;
uniform vec2 uTexel;
uniform float uDef;
uniform float uSharp;
in vec2 vUv;
out vec4 o;
float Y(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
void main() {
  vec2 uv = vUv;
  vec3 c = texture(uColor, uv).rgb;
  float y = Y(c);
  float s = 0.0, ws = 0.0, mn = 1.0, mx = 0.0;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    float yy = Y(texture(uColor, uv + vec2(float(i), float(j)) * uTexel).rgb);
    float w = exp(-float(i * i + j * j) / 2.0);
    s += yy * w; ws += w;
    if (abs(i) <= 1 && abs(j) <= 1) { mn = min(mn, yy); mx = max(mx, yy); }
  }
  float fine = y - s / ws;
  float mag = abs(fine);
  fine *= mag / (mag + 0.004);
  float ys = y + 2.5 * uSharp * fine;
  float tol = 0.08 * (mx - mn);
  ys = clamp(ys, mn - tol, mx + tol);
  float big = y - texture(uBig, uv).r;
  ys += 0.9 * uDef * 0.08 * tanh(big / 0.08);
  vec3 v = clamp(c + (ys - y), 0.0, 1.0);
  o = vec4(floor(v * 255.0 + 0.5) / 255.0, 1.0);
}`;

class PhotoProcessor {
  constructor() {
    this.canvas = document.createElement("canvas");
    const gl = this.canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false,
      premultipliedAlpha: false, alpha: false });
    if (!gl) throw new Error("Este navegador não tem WebGL2.");
    this.gl = gl;
    this.float = !!(gl.getExtension("EXT_color_buffer_float") || gl.getExtension("EXT_color_buffer_half_float"));
    this.acc = compile(gl, QUAD_VS, ACC_FS);
    this.resolve = compile(gl, QUAD_VS, RESOLVE_FS);
    this.down = compile(gl, QUAD_VS, DOWN_FS);
    this.blur = compile(gl, QUAD_VS, BLUR_FS);
    this.fin = compile(gl, QUAD_VS, FINAL_FS);
    quad(gl);
    rawPixels(gl);
    this.refTex = makeTex(gl, gl.LINEAR);
    this.frameTex = makeTex(gl, gl.LINEAR);
    this.lutTex = makeLutTex(gl);
    this.lutSize = 0;
    this.fbo = gl.createFramebuffer();
    this.targets = {};
  }

  setLut(lut) { this.lutSize = uploadLut(this.gl, this.lutTex, lut); }

  target(name, w, h, float) {
    const gl = this.gl;
    const key = `${name}`;
    let t = this.targets[key];
    if (!t || t.w !== w || t.h !== h) {
      if (t) gl.deleteTexture(t.tex);
      const tex = makeTex(gl, gl.LINEAR);
      const useFloat = float && this.float;
      gl.texImage2D(gl.TEXTURE_2D, 0, useFloat ? gl.RGBA16F : gl.RGBA8, w, h, 0, gl.RGBA,
                    useFloat ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null);
      t = this.targets[key] = { tex, w, h };
    }
    return t;
  }

  bindOut(t) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t ? this.fbo : null);
    if (t) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
    gl.viewport(0, 0, t ? t.w : this.canvas.width, t ? t.h : this.canvas.height);
  }

  tex(unit, tex, target = this.gl.TEXTURE_2D) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, tex);
  }

  // Começa a rajada com o quadro de referência.
  begin(source, w, h, mirror) {
    const gl = this.gl;
    this.w = w; this.h = h; this.mirror = mirror; this.frames = 1;
    this.tex(0, this.refTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    const acc = this.target("acc", w, h, true);
    this.bindOut(acc);
    gl.disable(gl.BLEND);
    this.drawAcc(true, [0, 0]);
  }

  // Soma mais um quadro, deslocado (dx, dy) em pixels da fonte.
  add(source, dx, dy, sigma) {
    const gl = this.gl;
    this.tex(1, this.frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    this.bindOut(this.targets.acc);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.drawAcc(false, [dx / this.w, dy / this.h], sigma);
    gl.disable(gl.BLEND);
    this.frames++;
  }

  drawAcc(first, shift, sigma = 0.06) {
    const gl = this.gl, u = this.acc.u;
    gl.useProgram(this.acc.prog);
    this.tex(0, this.refTex);
    this.tex(1, this.frameTex);
    gl.uniform1i(u.uRef, 0);
    gl.uniform1i(u.uFrame, 1);
    gl.uniform2f(u.uShift, shift[0], shift[1]);
    gl.uniform1i(u.uMirror, this.mirror ? 1 : 0);
    gl.uniform1i(u.uFirst, first ? 1 : 0);
    gl.uniform1f(u.uSigma, sigma);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // Normaliza, LUT, Definição e Nitidez; o resultado fica no canvas (w x h).
  finish(mix, def, sharp) {
    const gl = this.gl, w = this.w, h = this.h;
    const color = this.target("color", w, h, false);   // 8 bits bastam depois da LUT
    this.bindOut(color);
    gl.useProgram(this.resolve.prog);
    this.tex(0, this.targets.acc.tex);
    this.tex(1, this.lutTex, gl.TEXTURE_3D);
    gl.uniform1i(this.resolve.u.uAcc, 0);
    gl.uniform1i(this.resolve.u.uLut, 1);
    gl.uniform1i(this.resolve.u.uSize, this.lutSize);
    gl.uniform1f(this.resolve.u.uMix, mix);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    const qw = Math.max(1, Math.round(w / 4)), qh = Math.max(1, Math.round(h / 4));
    const q1 = this.target("q1", qw, qh, true), q2 = this.target("q2", qw, qh, true);
    this.bindOut(q1);
    gl.useProgram(this.down.prog);
    this.tex(0, color.tex);
    gl.uniform1i(this.down.u.uSrc, 0);
    gl.uniform2f(this.down.u.uTexel, 1 / w, 1 / h);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // Desfoque largo: sigma 12 px na resolução cheia (3 px na reduzida, 4x).
    const sigma = 3 * Math.max(0.5, Math.max(w, h) / 4032);
    for (const [src, dst, step] of [[q1, q2, [1 / qw, 0]], [q2, q1, [0, 1 / qh]]]) {
      this.bindOut(dst);
      gl.useProgram(this.blur.prog);
      this.tex(0, src.tex);
      gl.uniform1i(this.blur.u.uSrc, 0);
      gl.uniform2f(this.blur.u.uStep, step[0], step[1]);
      gl.uniform1f(this.blur.u.uSigma, Math.min(sigma, 10));
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.bindOut(null);
    gl.useProgram(this.fin.prog);
    this.tex(0, color.tex);
    this.tex(1, q1.tex);
    gl.uniform1i(this.fin.u.uColor, 0);
    gl.uniform1i(this.fin.u.uBig, 1);
    gl.uniform2f(this.fin.u.uTexel, 1 / w, 1 / h);
    gl.uniform1f(this.fin.u.uDef, def);
    gl.uniform1f(this.fin.u.uSharp, sharp);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  release() {
    // Libera a memória dos alvos grandes (12 MP em meia precisão pesam).
    const gl = this.gl;
    for (const t of Object.values(this.targets)) gl.deleteTexture(t.tex);
    this.targets = {};
  }
}

// ---------------------------------------------------------------- alinhamento da rajada
// Deslocamento global entre dois quadros, em pirâmide: 160 px, 640 px e um
// recorte central a 1280 px, com ajuste sub-pixel por parábola.
const alignCanvas = document.createElement("canvas");
const alignCtx = alignCanvas.getContext("2d", { willReadFrequently: true });

function lumaOf(source, sw, sh, w) {
  const h = Math.max(1, Math.round((w * sh) / sw));
  alignCanvas.width = w;
  alignCanvas.height = h;
  alignCtx.drawImage(source, 0, 0, w, h);
  const d = alignCtx.getImageData(0, 0, w, h).data, y = new Float32Array(w * h);
  for (let i = 0, j = 0; i < y.length; i++, j += 4) y[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
  return { y, w, h };
}

// Diferença entre a referência e o quadro deslocado (dx, dy). Nos níveis de
// busca, soma dos absolutos a cada 2 pixels (rápido); no último, soma dos
// quadrados em todos os pixels, que tem fundo de parábola e dá o sub-pixel certo.
function diffAt(a, b, dx, dy, region, fine) {
  const [x0, y0, x1, y1] = region, step = fine ? 1 : 2;
  let s = 0, n = 0;
  for (let y = y0; y < y1; y += step) {
    const yb = y + dy;
    if (yb < 0 || yb >= b.h) continue;
    const ra = y * a.w, rb = yb * b.w;
    for (let x = x0; x < x1; x += step) {
      const xb = x + dx;
      if (xb < 0 || xb >= b.w) continue;
      const d = a.y[ra + x] - b.y[rb + xb];
      s += fine ? d * d : Math.abs(d);
      n++;
    }
  }
  return n ? s / n : 1e9;
}

function search(a, b, cx, cy, rad, region, fine = false) {
  let best = [cx, cy], bs = Infinity;
  const cache = new Map();
  for (let dy = cy - rad; dy <= cy + rad; dy++)
    for (let dx = cx - rad; dx <= cx + rad; dx++) {
      const s = diffAt(a, b, dx, dy, region, fine);
      cache.set(dx + "," + dy, s);
      if (s < bs) { bs = s; best = [dx, dy]; }
    }
  return { best, cache, score: bs };
}

function refine(res, a, b, region) {
  const [bx, by] = res.best;
  const get = (dx, dy) => res.cache.get(dx + "," + dy) ?? diffAt(a, b, dx, dy, region, true);
  const s0 = get(bx, by);
  const para = (m, p) => { const d = m - 2 * s0 + p; return d > 1e-9 ? Math.max(-0.5, Math.min(0.5, (m - p) / (2 * d))) : 0; };
  return [bx + para(get(bx - 1, by), get(bx + 1, by)), by + para(get(bx, by - 1), get(bx, by + 1))];
}

// Acerto fino (Lucas-Kanade): parte do deslocamento inteiro e corrige pelo
// gradiente da referência, lendo o quadro com interpolação, até mudar < 0,01 px.
function lkRefine(a, b, start, region) {
  const [x0, y0, x1, y1] = region.map((v, i) => (i < 2 ? Math.max(v, 1) : Math.min(v, (i === 2 ? a.w : a.h) - 1)));
  const A = a.y, B = b.y, W = a.w;
  let hxx = 0, hxy = 0, hyy = 0;
  const pts = [];
  for (let y = y0; y < y1; y++)
    for (let x = x0; x < x1; x++) {
      const i = y * W + x;
      // Sobel: gradiente com um leve desfoque, menos sensível ao ruído.
      const gx = (A[i - W + 1] + 2 * A[i + 1] + A[i + W + 1] - A[i - W - 1] - 2 * A[i - 1] - A[i + W - 1]) / 8;
      const gy = (A[i + W - 1] + 2 * A[i + W] + A[i + W + 1] - A[i - W - 1] - 2 * A[i - W] - A[i - W + 1]) / 8;
      hxx += gx * gx; hxy += gx * gy; hyy += gy * gy;
      pts.push(x, y, gx, gy, A[i]);
    }
  const det = hxx * hyy - hxy * hxy;
  if (!(det > 1e-6)) return null;
  let [dx, dy] = start;
  for (let it = 0; it < 8; it++) {
    let ex = 0, ey = 0;
    for (let k = 0; k < pts.length; k += 5) {
      const sx = pts[k] + dx, sy = pts[k + 1] + dy;
      const ix = Math.floor(sx), iy = Math.floor(sy);
      if (ix < 0 || iy < 0 || ix >= b.w - 1 || iy >= b.h - 1) continue;
      const fx = sx - ix, fy = sy - iy, j = iy * b.w + ix;
      const v = (B[j] * (1 - fx) + B[j + 1] * fx) * (1 - fy) + (B[j + b.w] * (1 - fx) + B[j + b.w + 1] * fx) * fy;
      const e = v - pts[k + 4];
      ex += pts[k + 2] * e; ey += pts[k + 3] * e;
    }
    const ddx = -(hyy * ex - hxy * ey) / det, ddy = -(hxx * ey - hxy * ex) / det;
    dx += ddx; dy += ddy;
    if (Math.hypot(ddx, ddy) < 0.01) break;
  }
  return Math.hypot(dx - start[0], dy - start[1]) < 1.5 ? [dx, dy] : null;
}

function central(l, frac) {
  const mx = Math.round(l.w * (1 - frac) / 2), my = Math.round(l.h * (1 - frac) / 2);
  return [mx, my, l.w - mx, l.h - my];
}

function makeAligner(refSource, sw, sh) {
  const levels = [160, 640, 1280].map((w) => Math.min(w, sw));
  const refs = levels.map((w, i) => (i < 2 ? lumaOf(refSource, sw, sh, w) : null));
  const patch = Math.min(sw, 1280);
  refs[2] = lumaOf(refSource, sw, sh, patch);
  return (curSource) => {
    const a0 = refs[0], b0 = lumaOf(curSource, sw, sh, levels[0]);
    const r0 = search(a0, b0, 0, 0, 10, central(a0, 0.7));
    const k1 = levels[1] / levels[0];
    const a1 = refs[1], b1 = lumaOf(curSource, sw, sh, levels[1]);
    const r1 = search(a1, b1, Math.round(r0.best[0] * k1), Math.round(r0.best[1] * k1), 5, central(a1, 0.6));
    const k2 = levels[2] / levels[1];
    const a2 = refs[2], b2 = lumaOf(curSource, sw, sh, levels[2]);
    const reg = central(a2, 0.35);
    const r2 = search(a2, b2, Math.round(r1.best[0] * k2), Math.round(r1.best[1] * k2), 2, reg, true);
    const [fx, fy] = lkRefine(a2, b2, r2.best, reg) || refine(r2, a2, b2, reg);
    const k = sw / levels[2];
    return { dx: fx * k, dy: fy * k, erro: r2.score };
  };
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
  mix: 1, lutId: "raizen-33", recording: false, compare: false, last: null, busy: false,
  hdr: 1, def: 0, sharp: 0,
};
const luts = new Map();      // id -> {id, name, size, data, builtin}
let view, out, proc = null, stream = null, recorder = null, chunks = [], recStart = 0, audioStream = null;
const cam = $("#cam");
let testImage = null, testSource = null;

// "prefs2": a versão anterior guardava o HDR+ ligado como padrão, que borrava a foto na mão.
function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem("prefs2") || "{}");
    for (const k of ["kelvin", "ring", "mirror", "hdr", "def", "sharp"]) if (k in p) state[k] = p[k];
  } catch {}
}
function savePrefs() {
  try {
    const { kelvin, ring, mirror, hdr, def, sharp } = state;
    localStorage.setItem("prefs2", JSON.stringify({ kelvin, ring, mirror, hdr, def, sharp }));
  } catch {}
}

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
  if (TEST) { showRes(); return; }
  if (stream) stream.getTracks().forEach((t) => t.stop());
  // O formato 4K 16:9 de vídeo, o mesmo da gravação em 4K. (A versão que pedia o 4:3 de
  // 12 MP deu foto pior no iPhone.) A gravação do visor reduz para 1920 px.
  const video = { facingMode: state.facing, width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 30 } };
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video });
  } catch (e) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: state.facing } });
    } catch (e2) {
      msg("Sem acesso à câmera. Permita a câmera para este site nos Ajustes do Safari. (" + e2.name + ")", 0);
      return;
    }
  }
  cam.srcObject = stream;
  await cam.play().catch(() => {});
  resizeView();
  setTimeout(showRes, 600);
}

function showRes() {
  const [w, h] = sourceSize();
  if (!w) return;
  $("#res").textContent = `${mp(w, h)} MP`;
  $("#res").title = `${w}×${h}`;
}

const mp = (w, h) => (w * h / 1e6).toFixed(1).replace(".", ",");

function sourceSize() {
  if (TEST && testImage) return [testImage.naturalWidth, testImage.naturalHeight];
  return [cam.videoWidth || 0, cam.videoHeight || 0];
}
function source() { return TEST ? (testSource || testImage) : cam; }
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
    if (state.recording) out.draw(source(), { mirror: mirrored(), mix: state.mix });
  }
  if (!TEST && cam.requestVideoFrameCallback) cam.requestVideoFrameCallback(frame);
  else requestAnimationFrame(frame);
}

function nextFrame() {
  return new Promise((ok) => {
    if (!TEST && cam.requestVideoFrameCallback) cam.requestVideoFrameCallback(() => ok());
    else requestAnimationFrame(() => requestAnimationFrame(() => ok()));
  });
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
    if (proc) proc.setLut(lut);
    state.lut = lut;
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

// ---------------------------------------------------------------- foto
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const stamp = () => new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
const enhanced = () => state.hdr > 1 || state.def > 0 || state.sharp > 0;

async function takePhoto() {
  if (!ready() || state.busy) return;
  state.busy = true;
  const flash = state.light && state.facing === "user";
  try {
    if (flash) { $("#flash").classList.add("on"); await sleep(450); }   // a câmera se ajusta à luz da tela
    let canvas;
    if (enhanced() && proc) {
      canvas = await enhancedPhoto(flash);
    } else {
      const [sw, sh] = sourceSize();
      out.resize(sw, sh);
      out.smooth(false);
      out.draw(source(), { mirror: mirrored(), mix: state.mix });
      canvas = out.canvas;
    }
    if (flash) $("#flash").classList.remove("on");
    const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.95));
    if (proc) proc.release();
    showResult(blob, `Revelador_${stamp()}.jpg`, "image/jpeg");
  } catch (e) {
    msg("Falha ao processar a foto: " + e.message, 5000);
  } finally {
    $("#flash").classList.remove("on");
    state.busy = false;
  }
}

async function enhancedPhoto() {
  // HDR+ por rajada: o primeiro quadro é a referência; os outros são alinhados
  // a ele e somados com peso (o que se mexeu entra menos). Depois LUT, Definição
  // e Nitidez. A intensidade da LUT é a do slider (o shader mistura na resolução).
  const [sw, sh] = sourceSize();
  const n = Math.max(1, state.hdr);
  if (n > 1 && !proc.float) msg("Este iPhone/navegador não soma quadros em ponto flutuante: HDR+ desligado.", 3500);
  const frames = proc.float ? n : 1;
  msg(frames > 1 ? `HDR+: segure firme… (${frames} quadros)` : "Processando…", 0);
  const ref = await grab();
  proc.begin(ref, sw, sh, mirrored());
  const align = frames > 1 ? makeAligner(ref, sw, sh) : null;
  const shifts = [];
  for (let i = 1; i < frames; i++) {
    await nextFrame();
    const cur = await grab(i);
    const { dx, dy } = align(cur);
    shifts.push([dx, dy]);
    proc.add(cur, dx, dy, 0.06);
  }
  proc.finish(state.mix, state.def, state.sharp);
  $("#msg").hidden = true;
  enhancedPhoto.last = { frames, shifts };
  return proc.canvas;
}

// Quadro atual da câmera (no teste, um quadro sintético da rajada).
async function grab(i = 0) {
  if (!TEST) return cam;
  return testBurstFrame(i);
}

// ---------------------------------------------------------------- MÁXIMA e galeria
// A foto vem da câmera do próprio iPhone (foco, HDR, Deep Fusion e o flash dela)
// ou da galeria; a LUT entra depois, pixel a pixel, na resolução cheia.
const MAX_PIXELS = 16777216;   // maior canvas que o Safari do iPhone aceita (16,7 MP)

// JPEG mínimo 16x8 (metade esquerda vermelha, direita azul) com EXIF 6: gerado
// pelo gerar_recursos.py para descobrir se o navegador já gira a foto sozinho.
const ORIENT_TEST = "/9j/4AAQSkZJRgABAQAAAQABAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAYAAAAAAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAAIABADAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8X6/ynP8Av4PyHr/1UD/zXz//2Q==";

async function loadImage(src) {
  const img = new Image();
  img.src = src;
  await img.decode();
  return img;
}

// O navegador já entrega a foto de pé quando ela vai para a GPU? Desenha a imagem
// de teste num quadro 2x2: vermelho no canto de cima à direita = já girou.
let rotatesP = null;
function browserRotates() {
  return rotatesP || (rotatesP = (async () => {
    const img = await loadImage("data:image/jpeg;base64," + ORIENT_TEST);
    out.resize(2, 2);
    out.smooth(false);
    out.draw(img, { mix: 0 });
    const px = new Uint8Array(4);
    out.gl.readPixels(1, 1, 1, 1, out.gl.RGBA, out.gl.UNSIGNED_BYTE, px);
    return px[0] > px[2];
  })());
}

// Orientação EXIF e tamanho gravado de um JPEG (null se não for JPEG).
async function jpegInfo(blob) {
  const v = new DataView(await blob.slice(0, 262144).arrayBuffer());
  if (v.byteLength < 4 || v.getUint16(0) !== 0xffd8) return null;
  let orient = 1, p = 2;
  while (p + 4 <= v.byteLength && v.getUint8(p) === 0xff) {
    const m = v.getUint8(p + 1);
    if (m === 0xff) { p++; continue; }
    if (m === 0xd9 || m === 0xda) break;          // fim ou início dos dados: o tamanho vem antes
    const len = v.getUint16(p + 2);
    if (m === 0xe1 && p + 10 <= v.byteLength && v.getUint32(p + 4) === 0x45786966) orient = exifOrient(v, p + 10);
    const sof = m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc;
    if (sof && p + 9 <= v.byteLength) return { orient, w: v.getUint16(p + 7), h: v.getUint16(p + 5) };
    p += 2 + len;
  }
  return null;
}

function exifOrient(v, t) {
  if (t + 8 > v.byteLength) return 1;
  const le = v.getUint16(t) === 0x4949;
  const ifd = t + v.getUint32(t + 4, le);
  if (ifd + 2 > v.byteLength) return 1;
  for (let i = 0, n = v.getUint16(ifd, le); i < n; i++) {
    const e = ifd + 2 + i * 12;
    if (e + 12 > v.byteLength) break;
    if (v.getUint16(e, le) === 0x0112) {
      const o = v.getUint16(e + 8, le);
      return o >= 1 && o <= 8 ? o : 1;
    }
  }
  return 1;
}

// A foto (arquivo) de pé e com a LUT, no canvas de saída. Devolve o canvas e o tamanho.
async function lutPhoto(blob, { forceOrient = 0 } = {}) {
  const info = await jpegInfo(blob);
  const url = URL.createObjectURL(blob);
  try {
    const img = await loadImage(url);
    let w = img.naturalWidth, h = img.naturalHeight, orient = 1;
    if (info) {
      const turn = forceOrient || info.orient;
      [w, h] = turn >= 5 ? [info.h, info.w] : [info.w, info.h];
      orient = forceOrient || ((await browserRotates()) ? 1 : info.orient);
    }
    const k = Math.min(1, Math.sqrt(MAX_PIXELS / (w * h)));
    const ow = Math.floor(w * k), oh = Math.floor(h * k);
    out.resize(ow, oh);
    out.smooth(k < 1);
    out.draw(img, { orient, mix: state.mix });
    return { canvas: out.canvas, w: ow, h: oh, original: [w, h] };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function processPhotoFile(file) {
  if (!file || state.busy) return;
  state.busy = true;
  msg("Aplicando a LUT na foto…", 0);
  try {
    const r = await lutPhoto(file);
    const blob = await new Promise((ok) => r.canvas.toBlob(ok, "image/jpeg", 0.95));
    $("#msg").hidden = true;
    showResult(blob, `Revelador_${stamp()}.jpg`, "image/jpeg");
    const reduzida = r.w !== r.original[0] ? " (reduzida: o Safari não passa de 16,7 MP)" : "";
    msg(`Foto de ${mp(r.w, r.h)} MP com a LUT${reduzida}.`, 3500);
  } catch (e) {
    msg("Não deu para abrir essa foto: " + e.message, 5000);
  } finally {
    state.busy = false;
    ensureCamera();
  }
}

function openNative() {
  if (state.busy) return;
  const inp = $("#nativa");
  inp.setAttribute("capture", state.facing);
  inp.click();
}

// Volta o visor depois que a câmera do iPhone foi usada.
function ensureCamera() {
  if (TEST) return;
  const t = stream && stream.getVideoTracks()[0];
  if (!t || t.readyState !== "live" || t.muted) startCamera();
}

// ---------------------------------------------------------------- vídeo
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
function syncPhotoUi() {
  $("#hdr").value = String(state.hdr);
  $("#def").value = Math.round(state.def * 100);
  $("#sharp").value = Math.round(state.sharp * 100);
  $("#dval").textContent = Math.round(state.def * 100) + "%";
  $("#sval").textContent = Math.round(state.sharp * 100) + "%";
  $("#kelvin").value = state.kelvin;
  $("#ringsize").value = state.ring;
  $("#mirror").checked = state.mirror;
}

function bindUi() {
  $("#shutter").onclick = () => {
    if (state.mode === "foto") takePhoto();
    else if (state.mode === "maxima") openNative();
    else if (state.recording) stopVideo();
    else startVideo();
  };
  document.querySelectorAll(".mode").forEach((b) => (b.onclick = () => {
    if (state.recording || state.busy || state.mode === b.dataset.mode) return;
    state.mode = b.dataset.mode;
    document.querySelectorAll(".mode").forEach((x) => x.classList.toggle("on", x === b));
    $("#shutter").classList.toggle("video", state.mode === "video");
    $("#shutter").classList.toggle("maxima", state.mode === "maxima");
    applyLight();
    if (state.mode === "maxima")
      msg("MÁXIMA: o disparo abre a câmera do iPhone (foco, HDR e flash dela). A LUT entra logo depois, na resolução cheia.", 5000);
  }));
  $("#b-gal").onclick = () => { if (!state.busy) $("#galeria").click(); };
  for (const id of ["#nativa", "#galeria"])
    $(id).onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; processPhotoFile(f); };
  $("#b-flip").onclick = async () => {
    if (state.recording || state.busy) return;
    state.facing = state.facing === "user" ? "environment" : "user";
    applyLight();
    await startCamera();
  };
  $("#b-flash").onclick = () => {
    state.light = !state.light;
    applyLight();
    if (state.light && state.mode === "maxima") msg("Na MÁXIMA, use o flash da própria câmera do iPhone (o raio no canto).");
    else if (state.light && state.facing !== "user") msg("A luz da tela serve para a câmera frontal.");
    else if (state.light) msg("Luz ligada. Brilho do iPhone no máximo!");
  };
  $("#b-cfg").onclick = () => ($("#cfg").hidden = false);
  $("#b-cfg-close").onclick = () => ($("#cfg").hidden = true);
  $("#kelvin").oninput = (e) => { state.kelvin = +e.target.value; applyLight(); savePrefs(); };
  $("#ringsize").oninput = (e) => { state.ring = +e.target.value; applyLight(); savePrefs(); };
  $("#mirror").onchange = (e) => { state.mirror = e.target.checked; savePrefs(); };
  $("#hdr").onchange = (e) => { state.hdr = +e.target.value; syncPhotoUi(); savePrefs(); };
  $("#def").oninput = (e) => { state.def = +e.target.value / 100; syncPhotoUi(); savePrefs(); };
  $("#sharp").oninput = (e) => { state.sharp = +e.target.value / 100; syncPhotoUi(); savePrefs(); };
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
    if (document.visibilityState === "visible") ensureCamera();
    keepAwake();
  });
}

async function keepAwake() {
  try { if (navigator.wakeLock && document.visibilityState === "visible") await navigator.wakeLock.request("screen"); } catch {}
}

// ---------------------------------------------------------------- teste (?teste=1)
// A amostra no lugar da câmera: confere o shader com o Revelador (Python) e
// simula uma rajada tremida e com ruído para testar o HDR+.
function testBurstFrame(i) {
  const w = testImage.naturalWidth, h = testImage.naturalHeight;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const shifts = testBurstFrame.shifts || (testBurstFrame.shifts = [[0, 0], [3.4, -2.2], [-4.1, 1.6], [1.8, 4.3],
    [-2.6, -3.7], [5.2, 0.9], [-0.7, -5.1], [2.9, 2.6]]);
  const [dx, dy] = shifts[i % shifts.length];
  ctx.drawImage(testImage, dx, dy);         // o conteúdo aparece deslocado de (dx, dy)
  if (testBurstFrame.noise) {
    const img = ctx.getImageData(0, 0, w, h), d = img.data;
    let seed = 1234 + i * 977;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    for (let k = 0; k < d.length; k += 4) {
      const g = (rnd() + rnd() + rnd() - 1.5) * 2 * testBurstFrame.noise;   // ~gaussiano
      d[k] += g; d[k + 1] += g * 0.9; d[k + 2] += g * 1.1;
    }
    ctx.putImageData(img, 0, 0);
  }
  return c;
}

function readCanvas(canvas) {
  const c = document.createElement("canvas");
  c.width = canvas.width; c.height = canvas.height;
  const ctx = c.getContext("2d");
  ctx.drawImage(canvas, 0, 0);
  return ctx.getImageData(0, 0, c.width, c.height).data;
}

async function verify(id = "raizen-33") {
  await selectLut(id);
  const [w, h] = sourceSize();
  out.resize(w, h);
  out.smooth(false);
  out.draw(testImage, { mirror: false, mix: 1 });
  const gl = out.gl, px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const ref = await refPixels(w, h);
  let diff = 0, maxd = 0, off = 0;
  for (let y = 0; y < h; y++) {
    const gy = h - 1 - y;     // readPixels vem de baixo para cima
    for (let x = 0; x < w; x++) {
      for (let k = 0; k < 3; k++) {
        const d = Math.abs(px[(gy * w + x) * 4 + k] - ref[(y * w + x) * 4 + k]);
        diff += d; if (d > maxd) maxd = d; if (d) off++;
      }
    }
  }
  return { largura: w, altura: h, diferencaMedia: diff / (w * h * 3), diferencaMaxima: maxd,
           canaisDiferentes: off, totalCanais: w * h * 3 };
}

async function refPixels(w, h) {
  const ref = await new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = "teste/ref-raizen-33.png"; });
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(ref, 0, 0);
  return ctx.getImageData(0, 0, w, h).data;
}

// HDR+: rajada de 8 quadros tremidos com ruído -> alinhamento e ruído medidos
// contra a referência exata (a amostra pela LUT, sem ruído).
async function verifyHdr(frames = 8, noise = 12) {
  await selectLut("raizen-33");
  const keep = { hdr: state.hdr, def: state.def, sharp: state.sharp, mirror: state.mirror };
  Object.assign(state, { def: 0, sharp: 0, mirror: false });
  testBurstFrame.noise = noise;
  const [w, h] = sourceSize(), ref = await refPixels(w, h);
  const err = (data) => {
    // erro médio na região central (as bordas recebem menos quadros)
    let s = 0, n = 0;
    for (let y = Math.round(h * 0.1); y < h * 0.9; y++)
      for (let x = Math.round(w * 0.1); x < w * 0.9; x++)
        for (let k = 0; k < 3; k++) { s += Math.abs(data[(y * w + x) * 4 + k] - ref[(y * w + x) * 4 + k]); n++; }
    return s / n;
  };
  state.hdr = 1;
  const single = err(readCanvas(await enhancedPhoto()));
  state.hdr = frames;
  const merged = err(readCanvas(await enhancedPhoto()));
  const real = testBurstFrame.shifts.slice(1, frames);
  const found = enhancedPhoto.last.shifts;
  const alignErr = Math.max(...found.map(([dx, dy], i) => Math.hypot(dx - real[i][0], dy - real[i][1])));
  testBurstFrame.noise = 0;
  Object.assign(state, keep);
  proc.release();
  return { erroUmQuadro: +single.toFixed(2), erroHdr: +merged.toFixed(2),
           reducaoRuido: +(single / merged).toFixed(2), erroAlinhamentoPx: +alignErr.toFixed(2),
           deslocamentos: found.map(([a, b]) => [+a.toFixed(2), +b.toFixed(2)]) };
}

// Sem HDR+ e sem Definição/Nitidez a foto tem de ser o .cube exato; com eles,
// medimos o quanto mudam a imagem.
async function verifyEnhanced() {
  await selectLut("raizen-33");
  const keep = { hdr: state.hdr, def: state.def, sharp: state.sharp, mirror: state.mirror };
  testBurstFrame.noise = 0;
  const [w, h] = sourceSize(), ref = await refPixels(w, h);
  const diff = (data) => {
    let s = 0, m = 0;
    for (let k = 0; k < data.length; k += 4)
      for (let c = 0; c < 3; c++) { const d = Math.abs(data[k + c] - ref[k + c]); s += d; if (d > m) m = d; }
    return [+(s / (w * h * 3)).toFixed(3), m];
  };
  Object.assign(state, { hdr: 1, def: 0, sharp: 0, mirror: false });
  const plain = diff(readCanvas(await enhancedPhoto()));
  Object.assign(state, { def: 0.5, sharp: 0.5 });
  const detail = diff(readCanvas(await enhancedPhoto()));
  Object.assign(state, keep);
  proc.release();
  return { semAjustesProcessador: plain, comDefinicaoENitidez50: detail };
}

// MÁXIMA: a foto deitada com EXIF 6 tem de sair de pé e igual à LUT do Revelador,
// pelo caminho do navegador e pelo giro no shader (com o EXIF arrancado do arquivo).
async function verifyNative() {
  await selectLut("raizen-33");
  const keep = state.mix;
  state.mix = 1;
  const jpg = await (await fetch("teste/amostra-o6.jpg")).blob();
  const ref = await loadImage("teste/ref-o6.png");
  const cmp = (r) => {
    if (r.w !== ref.naturalWidth || r.h !== ref.naturalHeight)
      return { erro: `saiu ${r.w}×${r.h}, esperava ${ref.naturalWidth}×${ref.naturalHeight}` };
    const c = document.createElement("canvas");
    c.width = r.w; c.height = r.h;
    const ctx = c.getContext("2d");
    ctx.drawImage(ref, 0, 0);
    const a = ctx.getImageData(0, 0, r.w, r.h).data, b = readCanvas(r.canvas);
    let s = 0, m = 0, big = 0;
    for (let k = 0; k < a.length; k += 4)
      for (let ch = 0; ch < 3; ch++) {
        const d = Math.abs(a[k + ch] - b[k + ch]);
        s += d; if (d > m) m = d; if (d > 4) big++;
      }
    return { largura: r.w, altura: r.h, diferencaMedia: +(s / (r.w * r.h * 3)).toFixed(3),
             diferencaMaxima: m, canaisAcimaDe4: big };
  };
  const pelaGaleria = cmp(await lutPhoto(jpg));
  const giroNoShader = cmp(await lutPhoto(await stripExif(jpg), { forceOrient: 6 }));
  state.mix = keep;
  return { navegadorGira: await browserRotates(), pelaGaleria, giroNoShader };
}

async function stripExif(blob) {
  const b = new Uint8Array(await blob.arrayBuffer()), v = new DataView(b.buffer);
  const parts = [b.subarray(0, 2)];
  let p = 2;
  while (p + 4 <= b.length && b[p] === 0xff && b[p + 1] !== 0xda) {
    const len = v.getUint16(p + 2);
    if (b[p + 1] !== 0xe1) parts.push(b.subarray(p, p + 2 + len));
    p += 2 + len;
  }
  parts.push(b.subarray(p));
  return new Blob(parts, { type: "image/jpeg" });
}

// ---------------------------------------------------------------- início
async function main() {
  loadPrefs();
  view = new Renderer($("#view"), true);
  out = new Renderer(document.createElement("canvas"), false);
  try { proc = new PhotoProcessor(); } catch (e) { proc = null; }
  bindUi();
  syncPhotoUi();
  for (const item of await dbAll()) luts.set(item.id, { ...item, data: Float32Array.from(item.data) });
  let saved = "raizen-33";
  try { saved = localStorage.getItem("lut") || saved; } catch {}
  if (!BUILTIN.find((b) => b.id === saved) && !luts.has(saved)) saved = "raizen-33";
  if (TEST) {
    testImage = await new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = "teste/amostra.png"; });
    Object.assign(window, { __verificar: verify, __testarHdr: verifyHdr, __testarFoto: verifyEnhanced,
                            __testarMaxima: verifyNative, __estado: state });
  }
  await selectLut(saved);
  applyLight();
  resizeView();
  await startCamera();
  keepAwake();
  frame();
}

main().catch((e) => msg("Erro ao iniciar: " + e.message, 0));
