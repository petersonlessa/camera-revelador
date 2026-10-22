"use strict";
// Câmera Revelador — foto e vídeo com as LUTs .cube do Revelador, aplicadas na GPU.
// A LUT é interpolada no shader com a mesma conta trilinear do Revelador
// (8 leituras exatas da tabela, sem o filtro da placa) e o mesmo arredondamento,
// para a câmera dar o mesmo resultado que o .exe.
// Foto: o motor Raizen (rajada, anti-lavado, ruído, tom local, reflexo e nitidez)
// roda antes da LUT. No modo SÓ LUT, a foto é exatamente o .cube.

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
// De um ponto da foto de pé (d, y para baixo) para o ponto na foto como foi
// gravada. uOrient é a orientação EXIF (1 = já de pé).
const ORIENT_GLSL = `
vec2 orient(vec2 d) {
  if (uOrient == 2) return vec2(1.0 - d.x, d.y);
  if (uOrient == 3) return vec2(1.0 - d.x, 1.0 - d.y);
  if (uOrient == 4) return vec2(d.x, 1.0 - d.y);
  if (uOrient == 5) return d.yx;
  if (uOrient == 6) return vec2(d.y, 1.0 - d.x);
  if (uOrient == 7) return vec2(1.0 - d.y, 1.0 - d.x);
  if (uOrient == 8) return vec2(1.0 - d.y, d.x);
  return d;
}`;

const VERT = `#version 300 es
in vec2 aPos;
uniform vec4 uCrop;      // x0, y0, largura, altura na textura (0..1)
uniform bool uMirror;
uniform int uOrient;
out vec2 vUv;
${ORIENT_GLSL}
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

function srcDims(s) {
  if (s instanceof HTMLVideoElement) return [s.videoWidth, s.videoHeight];
  if (s instanceof HTMLImageElement) return [s.naturalWidth, s.naturalHeight];
  return [s.width, s.height];
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

// ---------------------------------------------------------------- motor Raizen (GPU)
// Etapas: 1 rajada alinhada e somada · 2 anti-lavado (ponto preto + névoa) ·
// 3 ruído (luminância aqui; croma na baixa resolução) · 4 tom local (fusão de 3
// exposições + contraste local) e reflexo do flash na pele · 5 nitidez em duas
// bandas · 6 a LUT. A curva Raizen, a densidade subtrativa e o roll-off da pele
// já estão dentro do .cube Raizen: o motor só os aplica sem LUT.
const QUAD_VS = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const LUMA_GLSL = `float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }`;

// 1. Soma um quadro alinhado ao acumulador (RGB*peso, peso). A comparação com a
// referência é feita na média 3x3 (sem o ruído): onde o quadro discorda, por
// movimento ou alinhamento ruim, o peso cai a quase zero e a borda não borra.
const ACC_FS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uRef;
uniform sampler2D uFrame;
uniform vec2 uShift;
uniform vec2 uTexel;
uniform bool uMirror;
uniform int uOrient;
uniform bool uFirst;
uniform float uSigma;
in vec2 vUv;
out vec4 o;
${ORIENT_GLSL}
vec3 box(sampler2D t, vec2 p) {
  vec3 s = vec3(0.0);
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) s += texture(t, p + vec2(i, j) * uTexel).rgb;
  return s / 9.0;
}
void main() {
  vec2 d = vec2(vUv.x, 1.0 - vUv.y);
  if (uMirror) d.x = 1.0 - d.x;
  vec2 uv = orient(d);
  vec3 r = texture(uRef, uv).rgb;
  if (uFirst) { o = vec4(r, 1.0); return; }
  vec2 s = uv + uShift;
  if (s.x < 0.0 || s.y < 0.0 || s.x > 1.0 || s.y > 1.0) { o = vec4(0.0); return; }
  vec3 dd = abs(box(uFrame, s) - box(uRef, uv));
  float dl = max(dd.r, max(dd.g, dd.b));
  float w = exp(-(dl * dl) / (2.0 * uSigma * uSigma));
  o = vec4(texture(uFrame, s).rgb * w, w);
}`;

// Média de blocos para a baixa resolução, lida pelo JS: Y em 16 bits (R e G),
// Cb e Cr em 8 bits (B e A).
const STATS_FS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uSrc;
uniform bool uAcc;
uniform vec2 uStep;
uniform int uN;
in vec2 vUv;
out vec4 o;
void main() {
  vec4 s = vec4(0.0);
  vec2 base = vUv - uStep * (float(uN) - 1.0) * 0.5;
  for (int j = 0; j < 8; j++) {
    if (j >= uN) break;
    for (int i = 0; i < 8; i++) {
      if (i >= uN) break;
      s += texture(uSrc, base + vec2(i, j) * uStep);
    }
  }
  vec3 c = clamp(uAcc ? s.rgb / max(s.a, 1e-4) : s.rgb / float(uN * uN), 0.0, 1.0);
  float y = dot(c, vec3(0.299, 0.587, 0.114));
  float y16 = floor(y * 65535.0 + 0.5);
  float hi = floor(y16 / 256.0);
  o = vec4(hi / 255.0, (y16 - hi * 256.0) / 255.0, (c.b - y) * 0.564 + 0.5, (c.r - y) * 0.713 + 0.5);
}`;

// 2. Anti-lavado: tira o véu cinza do preto (pé quadrático suave e reescala) e a
// névoa do flash pelo canal escuro, só nas sombras e meios-tons baixos.
const PREP_FS = `#version 300 es
precision highp float;
uniform sampler2D uAcc;
uniform float uBlack;
uniform float uHaze;
in vec2 vUv;
out vec4 o;
${LUMA_GLSL}
vec3 toe(vec3 x, float k) {
  if (k <= 0.0) return x;
  return mix(x * x / (4.0 * k), x - k, step(2.0 * k, x)) / (1.0 - k);
}
void main() {
  vec4 a = texture(uAcc, vUv);
  vec3 c = toe(clamp(a.rgb / max(a.a, 1e-4), 0.0, 1.0), uBlack);
  float dk = min(c.r, min(c.g, c.b));
  float t = clamp(1.0 - 0.32 * dk, 0.35, 1.0);
  vec3 j = (c - (1.0 - t)) / t;
  float w = uHaze * (1.0 - smoothstep(0.45, 0.62, luma(c)));
  o = vec4(clamp(mix(c, j, w), 0.0, 1.0), 1.0);
}`;

// 3. Ruído de luminância: bilateral 5x5 só onde a região é lisa (parede, fundo,
// sombra); em olho, cílio, barba e cabelo a imagem passa intacta.
const DENOISE_FS = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uAmount;
uniform float uRange;
in vec2 vUv;
out vec4 o;
${LUMA_GLSL}
void main() {
  vec3 c = texture(uSrc, vUv).rgb;
  float y = luma(c);
  float s = 0.0, ws = 0.0, mn = 1.0, mx = 0.0;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    float yy = luma(texture(uSrc, vUv + vec2(i, j) * uTexel).rgb);
    float d = yy - y;
    float w = exp(-float(i * i + j * j) / 4.5 - d * d / (2.0 * uRange * uRange));
    s += yy * w; ws += w;
    mn = min(mn, yy); mx = max(mx, yy);
  }
  float liso = 1.0 - smoothstep(0.06, 0.16, mx - mn);
  float yd = mix(y, s / ws, uAmount * liso);
  o = vec4(clamp(c + (yd - y), 0.0, 1.0), 1.0);
}`;

// Desfoque gaussiano separável do Y (para a banda média da nitidez).
const BLUR_FS = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform bool uRgb;
uniform vec2 uStep;
uniform float uSigma;
in vec2 vUv;
out vec4 o;
${LUMA_GLSL}
void main() {
  int r = int(ceil(3.0 * uSigma));
  float s = 0.0, ws = 0.0;
  for (int i = -12; i <= 12; i++) {
    if (i < -r || i > r) continue;
    float w = exp(-float(i * i) / (2.0 * uSigma * uSigma));
    vec4 t = texture(uSrc, vUv + float(i) * uStep);
    s += (uRgb ? luma(t.rgb) : t.r) * w;
    ws += w;
  }
  o = vec4(s / ws, 0.0, 0.0, 1.0);
}`;

// 4 a 6. Ganho do tom local e croma limpo vindos da baixa resolução (interpolação
// guiada pelo Y daqui: não vaza de um lado da borda para o outro), reflexo do
// flash na pele, nitidez em duas bandas e a LUT.
const FINAL_FS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uColor;
uniform sampler2D uBlur;
uniform highp sampler2D uMap;
uniform highp sampler2D uSkin;
uniform ivec2 uLow;
uniform vec2 uTexel;
uniform float uChroma;
uniform float uSharp;
uniform float uSpec;
uniform float uMix;
uniform bool uRaizen;
${LUT_GLSL}
in vec2 vUv;
out vec4 o;
${LUMA_GLSL}
vec4 low(highp sampler2D t, ivec2 p) { return texelFetch(t, clamp(p, ivec2(0), uLow - 1), 0); }
// Curva Raizen do Revelador (gamma 2,3, sombras -11 em 0..80 com offset 15,
// meios-tons +2, altas-luzes +4), na escala 0..255.
float raizen(float v) {
  float g = 255.0 * pow(max(v, 0.0), 2.3 / 2.2);
  float s = sin(3.14159265 * g / 80.0);
  float sh = g < 80.0 ? s * s * smoothstep(0.0, 15.0, g) : 0.0;
  float m = sin(3.14159265 * (g - 80.0) / 110.0);
  float md = g >= 80.0 && g < 190.0 ? m * m : 0.0;
  float h = sin(3.14159265 * (g - 190.0) / 65.0);
  float hi = g >= 190.0 ? h * h : 0.0;
  return clamp(g - 11.0 * sh + 2.0 * md + 4.0 * hi, 0.0, 255.0) / 255.0;
}
void main() {
  vec3 c = texture(uColor, vUv).rgb;
  float y = luma(c);
  vec2 ch = vec2((c.b - y) * 0.564, (c.r - y) * 0.713);

  vec2 p = vUv * vec2(uLow) - 0.5;
  ivec2 i0 = ivec2(floor(p));
  vec2 f = p - floor(p);
  float ws = 0.0, gain = 0.0;
  vec2 cd = vec2(0.0);
  vec4 sk = vec4(0.0);
  for (int j = 0; j <= 1; j++) for (int i = 0; i <= 1; i++) {
    ivec2 q = i0 + ivec2(i, j);
    float b = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
    vec4 mp = low(uMap, q);
    float dy = y - mp.y;
    float w = b * (exp(-dy * dy / 0.0098) + 0.001);
    ws += w; gain += w * mp.x; cd += w * (mp.zw - 0.5);
    sk += b * low(uSkin, q);
  }
  gain /= ws;
  cd /= ws;
  ch = mix(ch, cd, uChroma);

  // Reflexo do flash: brilho alto e sem cor, com pele em volta -> -15% de brilho
  // e a cor média da pele saudável de volta.
  float sat = length(ch);
  float m = smoothstep(0.78, 0.92, y) * (1.0 - smoothstep(0.03, 0.10, sat))
          * smoothstep(0.15, 0.40, sk.x) * uSpec;
  ch = mix(ch, sk.yz - 0.5, 0.7 * m);

  // Nitidez: banda fina (sigma 0,9) e média (0,9 a 2,2), com portão de ruído,
  // freio nas sombras (Y < 25) e nos reflexos (Y > 215) e trava de contorno.
  float s09 = 0.0, w09 = 0.0, mn = 1.0, mx = 0.0;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    float yy = luma(texture(uColor, vUv + vec2(i, j) * uTexel).rgb);
    float w = exp(-float(i * i + j * j) / 1.62);
    s09 += yy * w; w09 += w;
    if (abs(i) <= 1 && abs(j) <= 1) { mn = min(mn, yy); mx = max(mx, yy); }
  }
  float g09 = s09 / w09;
  float fine = y - g09, mid = g09 - texture(uBlur, vUv).r;
  fine *= abs(fine) / (abs(fine) + 0.004);
  float gate = smoothstep(0.004, 0.02, abs(mid));
  float brake = smoothstep(0.08, 0.13, y) * (1.0 - smoothstep(0.80, 0.86, y));
  float tol = 0.08 * (mx - mn);
  float det = clamp((1.2 * fine + 0.6 * mid) * uSharp * gate * brake, mn - tol - y, mx + tol - y);

  float yo = (y * (1.0 - 0.15 * m) + det) * gain;
  vec2 co = ch * clamp(gain, 0.5, 2.0);       // a cor acompanha o tom (não desbota)
  if (uRaizen) {                              // sem LUT: vivacidade +14%, densidade -5%
    float s = length(co);
    co *= 1.0 + 0.14 * (1.0 - smoothstep(0.0, 0.25, s));
    yo *= 1.0 - 0.05 * smoothstep(0.08, 0.25, s);
  }
  vec3 rgb = clamp(vec3(yo + 1.402 * co.y, yo - 0.344136 * co.x - 0.714136 * co.y, yo + 1.772 * co.x), 0.0, 1.0);
  if (uRaizen) rgb = vec3(raizen(rgb.r), raizen(rgb.g), raizen(rgb.b));
  vec3 v = clamp(rgb + (applyLut(rgb) - rgb) * uMix, 0.0, 1.0);
  o = vec4(floor(v * 255.0 + 0.5) / 255.0, 1.0);
}`;

// Força de cada etapa: Noite (ou com a Luz) e Dia.
const MOTOR = {
  noite: { quadros: 12, preto: 1.0, nevoa: 1.0, ruidoY: 0.9, ruidoC: 0.9, fusao: 0.5, volume: 0.9, nitidez: 1.0, reflexo: 1.0 },
  dia: { quadros: 1, preto: 0.7, nevoa: 0.45, ruidoY: 0.35, ruidoC: 0.55, fusao: 0.3, volume: 0.7, nitidez: 0.8, reflexo: 0.6 },
};
const BAIXA = 1024;   // lado maior da baixa resolução (tom local, croma, pele)

class Motor {
  constructor() {
    this.canvas = document.createElement("canvas");
    const gl = this.canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false,
      premultipliedAlpha: false, alpha: false });
    if (!gl) throw new Error("Este navegador não tem WebGL2.");
    this.gl = gl;
    this.float = !!(gl.getExtension("EXT_color_buffer_float") || gl.getExtension("EXT_color_buffer_half_float"));
    this.p = {
      acc: compile(gl, QUAD_VS, ACC_FS), stats: compile(gl, QUAD_VS, STATS_FS),
      prep: compile(gl, QUAD_VS, PREP_FS), denoise: compile(gl, QUAD_VS, DENOISE_FS),
      blur: compile(gl, QUAD_VS, BLUR_FS), fin: compile(gl, QUAD_VS, FINAL_FS),
    };
    quad(gl);
    rawPixels(gl);
    this.refTex = makeTex(gl, gl.LINEAR);
    this.frameTex = makeTex(gl, gl.LINEAR);
    this.mapTex = makeTex(gl, gl.NEAREST);
    this.skinTex = makeTex(gl, gl.NEAREST);
    this.lutTex = makeLutTex(gl);
    this.lutSize = 0;
    this.fbo = gl.createFramebuffer();
    this.targets = {};
  }

  setLut(lut) {
    this.gl.activeTexture(this.gl.TEXTURE4);
    this.lutSize = uploadLut(this.gl, this.lutTex, lut);
  }

  target(name, w, h, fmt) {
    const gl = this.gl;
    let t = this.targets[name];
    if (!t || t.w !== w || t.h !== h) {
      if (t) gl.deleteTexture(t.tex);
      const [inner, format, type] = {
        rgba16f: [gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT], r16f: [gl.R16F, gl.RED, gl.HALF_FLOAT],
        rgba8: [gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE],
      }[fmt];
      const tex = makeTex(gl, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, inner, w, h, 0, format, type, null);
      t = this.targets[name] = { tex, w, h };
    }
    return t;
  }

  tex(unit, tex, target = this.gl.TEXTURE_2D) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, tex);
  }

  run(p, out, set) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, out ? this.fbo : null);
    if (out) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out.tex, 0);
    gl.viewport(0, 0, out ? out.w : this.canvas.width, out ? out.h : this.canvas.height);
    gl.useProgram(p.prog);
    set(p.u);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // Começa a foto (w x h, já de pé) com o quadro de referência.
  begin(source, w, h, { mirror = false, orient = 1 } = {}) {
    const gl = this.gl;
    Object.assign(this, { w, h, mirror, orient, frames: 1 });
    const [sw, sh] = srcDims(source);
    this.texel = [1 / sw, 1 / sh];
    this.tex(0, this.refTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.disable(gl.BLEND);
    this.accumulate(true, [0, 0]);
  }

  // Soma mais um quadro, deslocado (dx, dy) em pixels da fonte.
  add(source, dx, dy) {
    const gl = this.gl;
    this.tex(1, this.frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.accumulate(false, [dx * this.texel[0], dy * this.texel[1]]);
    gl.disable(gl.BLEND);
    this.frames++;
  }

  accumulate(first, shift) {
    const gl = this.gl;
    this.run(this.p.acc, this.target("t0", this.w, this.h, "rgba16f"), (u) => {
      this.tex(0, this.refTex);
      this.tex(1, this.frameTex);
      gl.uniform1i(u.uRef, 0);
      gl.uniform1i(u.uFrame, 1);
      gl.uniform2f(u.uShift, shift[0], shift[1]);
      gl.uniform2f(u.uTexel, this.texel[0], this.texel[1]);
      gl.uniform1i(u.uMirror, this.mirror ? 1 : 0);
      gl.uniform1i(u.uOrient, this.orient);
      gl.uniform1i(u.uFirst, first ? 1 : 0);
      gl.uniform1f(u.uSigma, 0.03);
    });
  }

  // Lê a baixa resolução de uma etapa (média de blocos).
  readSmall(src, small, acc) {
    const gl = this.gl, n = Math.min(8, Math.ceil(this.w / small.w));
    this.run(this.p.stats, small, (u) => {
      this.tex(0, src.tex);
      gl.uniform1i(u.uSrc, 0);
      gl.uniform1i(u.uAcc, acc ? 1 : 0);
      gl.uniform1i(u.uN, n);
      gl.uniform2f(u.uStep, 1 / (small.w * n), 1 / (small.h * n));
    });
    const px = new Uint8Array(small.w * small.h * 4);
    gl.readPixels(0, 0, small.w, small.h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return px;
  }

  upload(tex, data, w, h) {
    const gl = this.gl;
    this.tex(5, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
  }

  // Revela a foto somada: etapas 2 a 6. noite = true/false, ou null para decidir
  // pelo brilho da cena. O resultado fica no canvas (w x h).
  async develop({ noite = null, preto = 0.75, mix = 1, raizen = false } = {}) {
    await sleep(0);   // deixa a mensagem aparecer antes da conta pesada
    const gl = this.gl, w = this.w, h = this.h;
    gl.disable(gl.BLEND);
    const k = Math.min(1, BAIXA / Math.max(w, h));
    const lw = Math.max(1, Math.round(w * k)), lh = Math.max(1, Math.round(h * k));
    const t0 = this.targets.t0, t1 = this.target("t1", w, h, "rgba16f"), small = this.target("small", lw, lh, "rgba8");

    const est = estatistica(this.readSmall(t0, small, true));
    const noturno = noite === null ? est.media < 0.22 : noite;
    const P = MOTOR[noturno ? "noite" : "dia"];
    const black = Math.min(0.25, est.p015 * 1.17 * preto * P.preto);
    const haze = Math.min(1, P.nevoa * preto / 0.75);

    this.run(this.p.prep, t1, (u) => {
      this.tex(0, t0.tex);
      gl.uniform1i(u.uAcc, 0);
      gl.uniform1f(u.uBlack, black);
      gl.uniform1f(u.uHaze, haze);
    });
    this.run(this.p.denoise, t0, (u) => {
      this.tex(0, t1.tex);
      gl.uniform1i(u.uSrc, 0);
      gl.uniform2f(u.uTexel, 1 / w, 1 / h);
      gl.uniform1f(u.uAmount, P.ruidoY);
      gl.uniform1f(u.uRange, 0.03 + 0.03 * P.ruidoY);
    });

    const mapas = mapasBaixos(this.readSmall(t0, small, false), lw, lh, P);
    this.upload(this.mapTex, mapas.map, lw, lh);
    this.upload(this.skinTex, mapas.pele, lw, lh);

    const y1 = this.target("y1", w, h, "r16f"), y2 = this.target("y2", w, h, "r16f");
    for (const [src, dst, rgb, step] of [[t0, y1, true, [1 / w, 0]], [y1, y2, false, [0, 1 / h]]])
      this.run(this.p.blur, dst, (u) => {
        this.tex(0, src.tex);
        gl.uniform1i(u.uSrc, 0);
        gl.uniform1i(u.uRgb, rgb ? 1 : 0);
        gl.uniform2f(u.uStep, step[0], step[1]);
        gl.uniform1f(u.uSigma, 2.2);
      });

    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.run(this.p.fin, null, (u) => {
      this.tex(0, t0.tex);
      this.tex(1, y2.tex);
      this.tex(2, this.mapTex);
      this.tex(3, this.skinTex);
      this.tex(4, this.lutTex, gl.TEXTURE_3D);
      gl.uniform1i(u.uColor, 0);
      gl.uniform1i(u.uBlur, 1);
      gl.uniform1i(u.uMap, 2);
      gl.uniform1i(u.uSkin, 3);
      gl.uniform1i(u.uLut, 4);
      gl.uniform1i(u.uSize, this.lutSize);
      gl.uniform2i(u.uLow, lw, lh);
      gl.uniform2f(u.uTexel, 1 / w, 1 / h);
      gl.uniform1f(u.uChroma, P.ruidoC);
      gl.uniform1f(u.uSharp, P.nitidez);
      gl.uniform1f(u.uSpec, P.reflexo);
      gl.uniform1f(u.uMix, mix);
      gl.uniform1i(u.uRaizen, raizen ? 1 : 0);
    });
    return { canvas: this.canvas, info: { noite: noturno, frames: this.frames, pretoAntes: est.p015, preto: black, media: est.media } };
  }

  release() {
    // Libera a memória dos alvos grandes (12 MP em meia precisão pesam).
    const gl = this.gl;
    for (const t of Object.values(this.targets)) gl.deleteTexture(t.tex);
    this.targets = {};
  }
}

// ---------------------------------------------------------------- motor: baixa resolução (JS, Float32)
const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const janela = (v, a, b, e) => sstep(a - e, a + e, v) * (1 - sstep(b - e, b + e, v));

// Ponto preto (percentil 1,5% do Y) e brilho médio.
function estatistica(px) {
  const hist = new Uint32Array(1024), n = px.length / 4;
  let soma = 0;
  for (let i = 0; i < px.length; i += 4) {
    const y = (px[i] * 256 + px[i + 1]) / 65535;
    soma += y;
    hist[Math.min(1023, (y * 1024) | 0)]++;
  }
  let acc = 0, p = 0;
  for (; p < 1023; p++) { acc += hist[p]; if (acc >= n * 0.015) break; }
  return { p015: (p + 0.5) / 1024, media: soma / n };
}

// Média móvel (2r+1) na horizontal e na vertical; 3 passadas ≈ gaussiana.
function boxBlur(a, w, h, r) {
  const t = new Float32Array(w * h), o = new Float32Array(w * h), k = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let s = 0;
    for (let i = -r; i <= r; i++) s += a[row + Math.min(w - 1, Math.max(0, i))];
    for (let x = 0; x < w; x++) {
      t[row + x] = s * k;
      s += a[row + Math.min(w - 1, x + r + 1)] - a[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -r; i <= r; i++) s += t[Math.min(h - 1, Math.max(0, i)) * w + x];
    for (let y = 0; y < h; y++) {
      o[y * w + x] = s * k;
      s += t[Math.min(h - 1, y + r + 1) * w + x] - t[Math.max(0, y - r) * w + x];
    }
  }
  return o;
}

function gaussAprox(a, w, h, sigma) {
  const r = Math.max(1, Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2));
  return boxBlur(boxBlur(boxBlur(a, w, h, r), w, h, r), w, h, r);
}

// Croma (Cb e Cr) desfocado e guiado pelo Y: a cor não atravessa as bordas.
function bilateralCroma(cb, cr, Y, w, h, sigma, sr) {
  const R = Math.ceil(2.5 * sigma), gs = new Float32Array(2 * R + 1), rw = new Float32Array(256);
  for (let k = -R; k <= R; k++) gs[k + R] = Math.exp(-k * k / (2 * sigma * sigma));
  for (let q = 0; q < 256; q++) rw[q] = Math.exp(-((q / 255) ** 2) / (2 * sr * sr));
  const passo = (a1, a2, horiz) => {
    const o1 = new Float32Array(w * h), o2 = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x, yi = Y[i];
        let s1 = 0, s2 = 0, ws = 0;
        for (let k = -R; k <= R; k++) {
          const j = horiz ? y * w + Math.min(w - 1, Math.max(0, x + k)) : Math.min(h - 1, Math.max(0, y + k)) * w + x;
          const wt = gs[k + R] * rw[Math.min(255, (Math.abs(Y[j] - yi) * 255) | 0)];
          s1 += wt * a1[j]; s2 += wt * a2[j]; ws += wt;
        }
        o1[i] = s1 / ws; o2[i] = s2 / ws;
      }
    return [o1, o2];
  };
  const [h1, h2] = passo(cb, cr, true);
  return passo(h1, h2, false);
}

// Pirâmide: filtro 1-4-6-4-1 e metade do tamanho; a volta é bilinear.
function down2(a, w, h) {
  const w2 = (w + 1) >> 1, h2 = (h + 1) >> 1, t = new Float32Array(w2 * h), o = new Float32Array(w2 * h2);
  const K = [1, 4, 6, 4, 1];
  for (let y = 0; y < h; y++)
    for (let x2 = 0; x2 < w2; x2++) {
      let s = 0;
      for (let k = 0; k < 5; k++) s += K[k] * a[y * w + Math.min(w - 1, Math.max(0, 2 * x2 + k - 2))];
      t[y * w2 + x2] = s / 16;
    }
  for (let y2 = 0; y2 < h2; y2++)
    for (let x2 = 0; x2 < w2; x2++) {
      let s = 0;
      for (let k = 0; k < 5; k++) s += K[k] * t[Math.min(h - 1, Math.max(0, 2 * y2 + k - 2)) * w2 + x2];
      o[y2 * w2 + x2] = s / 16;
    }
  return { a: o, w: w2, h: h2 };
}

function up2(a, w2, h2, w, h) {
  const o = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(h2 - 1, y / 2), y0 = Math.floor(fy), y1 = Math.min(h2 - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(w2 - 1, x / 2), x0 = Math.floor(fx), x1 = Math.min(w2 - 1, x0 + 1), tx = fx - x0;
      o[y * w + x] = (a[y0 * w2 + x0] * (1 - tx) + a[y0 * w2 + x1] * tx) * (1 - ty)
                   + (a[y1 * w2 + x0] * (1 - tx) + a[y1 * w2 + x1] * tx) * ty;
    }
  }
  return o;
}

// Fusão de exposições (Mertens): 3 exposições virtuais do Y (gamma 0,65, 1 e
// 1,55), pesadas por contraste local e boa exposição (gaussiana em 0,52) e
// misturadas em pirâmide de Laplace (sem costura nem halo).
function fusao(Y, w, h) {
  const n = w * h;
  const E = [0.65, 1, 1.55].map((g) => {
    if (g === 1) return Y;
    const e = new Float32Array(n);
    for (let i = 0; i < n; i++) e[i] = Math.pow(Y[i], g);
    return e;
  });
  const W = E.map((e) => {
    const wt = new Float32Array(n);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const lap = 4 * e[i] - e[y * w + Math.max(0, x - 1)] - e[y * w + Math.min(w - 1, x + 1)]
                  - e[Math.max(0, y - 1) * w + x] - e[Math.min(h - 1, y + 1) * w + x];
        const d = e[i] - 0.52;
        wt[i] = (Math.abs(lap) + 0.001) * Math.exp(-d * d / 0.08) + 1e-12;
      }
    return wt;
  });
  for (let i = 0; i < n; i++) {
    const s = W[0][i] + W[1][i] + W[2][i];
    W[0][i] /= s; W[1][i] /= s; W[2][i] /= s;
  }
  const niveis = Math.max(1, Math.floor(Math.log2(Math.min(w, h) / 8)));
  const piramide = (a) => {
    const p = [{ a, w, h }];
    for (let k = 1; k < niveis; k++) { const q = p[k - 1]; p.push(down2(q.a, q.w, q.h)); }
    return p;
  };
  const GE = E.map(piramide), GW = W.map(piramide);
  let out = null;
  for (let k = niveis - 1; k >= 0; k--) {
    const lw = GE[0][k].w, lh = GE[0][k].h, m = lw * lh, nivel = new Float32Array(m);
    for (let e = 0; e < 3; e++) {
      const g = GE[e][k].a, wk = GW[e][k].a;
      const prox = k < niveis - 1 ? up2(GE[e][k + 1].a, GE[e][k + 1].w, GE[e][k + 1].h, lw, lh) : null;
      for (let i = 0; i < m; i++) nivel[i] += wk[i] * (prox ? g[i] - prox[i] : g[i]);
    }
    if (out) {
      const u = up2(out.a, out.w, out.h, lw, lh);
      for (let i = 0; i < m; i++) u[i] += nivel[i];
      out = { a: u, w: lw, h: lh };
    } else out = { a: nivel, w: lw, h: lh };
  }
  return out.a;
}

// Os mapas que a GPU usa: [ganho do tom, Y, Cb limpo, Cr limpo] e
// [pele em volta, Cb e Cr médios da pele saudável].
function mapasBaixos(px, w, h, P) {
  const n = w * h, Y = new Float32Array(n), cb = new Float32Array(n), cr = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    Y[i] = (px[j] * 256 + px[j + 1]) / 65535;
    cb[i] = px[j + 2] / 255;
    cr[i] = px[j + 3] / 255;
  }
  // Ruído de croma em duas escalas (equivalem a uns 6 e 18 px na foto cheia);
  // a escala grande roda em meia resolução, onde custa 8 vezes menos.
  const [cb1, cr1] = bilateralCroma(cb, cr, Y, w, h, 1.5, 0.08);
  const Yh = down2(Y, w, h), cbh = down2(cb, w, h), crh = down2(cr, w, h);
  const [cb2h, cr2h] = bilateralCroma(cbh.a, crh.a, Yh.a, Yh.w, Yh.h, 2.25, 0.08);
  const cb2 = up2(cb2h, Yh.w, Yh.h, w, h), cr2 = up2(cr2h, Yh.w, Yh.h, w, h);
  // Tom local: fusão misturada ao original e contraste local limitado (tipo CLAHE).
  const Yf = fusao(Y, w, h), Ym = new Float32Array(n);
  for (let i = 0; i < n; i++) Ym[i] = Y[i] + P.fusao * (Yf[i] - Y[i]);
  const grande = gaussAprox(Ym, w, h, Math.max(w, h) / 40);
  const map = new Float32Array(n * 4), pele = new Float32Array(n * 4);
  const sw = new Float32Array(n), swb = new Float32Array(n), swr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const yl = Ym[i] + P.volume * 0.8 * 0.08 * Math.tanh((Ym[i] - grande[i]) / 0.08);
    const cbD = 0.5 * (cb1[i] + cb2[i]), crD = 0.5 * (cr1[i] + cr2[i]);
    map[i * 4] = Math.min(4, Math.max(0.25, yl / Math.max(Y[i], 0.004)));
    map[i * 4 + 1] = Y[i];
    map[i * 4 + 2] = cbD;
    map[i * 4 + 3] = crD;
    // Pele: faixa clássica de Cb/Cr (77-127 e 133-173) em meios-tons.
    const s = janela(cbD, 0.30, 0.50, 0.03) * janela(crD, 0.52, 0.68, 0.03) * janela(Y[i], 0.18, 0.80, 0.06);
    sw[i] = s; swb[i] = s * cbD; swr[i] = s * crD;
  }
  const sig = Math.max(w, h) / 30;
  const A = gaussAprox(sw, w, h, sig), B = gaussAprox(swb, w, h, sig), C = gaussAprox(swr, w, h, sig);
  for (let i = 0; i < n; i++) {
    const tem = A[i] > 1e-4;
    pele[i * 4] = A[i];
    pele[i * 4 + 1] = tem ? B[i] / A[i] : 0.5;
    pele[i * 4 + 2] = tem ? C[i] / A[i] : 0.5;
  }
  return { map, pele };
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
  const refs = levels.map((w) => lumaOf(refSource, sw, sh, w));
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
const MODOS = ["auto", "noite", "dia", "lut"];
const ROTULO = { auto: "AUTO", noite: "🌙 NOITE", dia: "☀️ DIA", lut: "SÓ LUT" };
const AVISO = {
  auto: "Auto: o app escolhe Noite (cena escura ou com a Luz) ou Dia.",
  noite: "🌙 Noite / Flash Raizen 6.1 (anti-lavado): rajada de 12 quadros, pretos cravados, ruído limpo e volume.",
  dia: "☀️ Dia / luz natural: pretos, limpeza leve, volume e nitidez.",
  lut: "Só LUT: a foto sai idêntica ao .cube, sem o motor.",
};
const state = {
  facing: "user", mode: "foto", light: false, kelvin: 5600, ring: 18, mirror: true,
  mix: 1, lutId: "raizen-33", recording: false, compare: false, last: null, busy: false,
  modo: "auto", preto: 0.75,
};
const luts = new Map();      // id -> {id, name, size, data, builtin}
let view, out, motor = null, stream = null, recorder = null, chunks = [], recStart = 0, audioStream = null;
const cam = $("#cam");
let testImage = null;

// "prefs3": as versões anteriores guardavam o HDR+ e a Definição/Nitidez antigos.
function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem("prefs3") || "{}");
    for (const k of ["kelvin", "ring", "mirror", "modo", "preto"]) if (k in p) state[k] = p[k];
    if (!MODOS.includes(state.modo)) state.modo = "auto";
  } catch {}
}
function savePrefs() {
  try {
    const { kelvin, ring, mirror, modo, preto } = state;
    localStorage.setItem("prefs3", JSON.stringify({ kelvin, ring, mirror, modo, preto }));
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
  // 5600 K ≈ #FFEFE1, o branco quente que não deixa a pele pálida.
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
    if (state.recording) out.draw(source(), { mirror: mirrored(), mix: state.mix });
  }
  if (!TEST && cam.requestVideoFrameCallback) cam.requestVideoFrameCallback(frame);
  else requestAnimationFrame(frame);
}

// Próximo quadro da câmera, com limite de 120 ms: se a câmera engasgar (ou a aba
// ficar escondida), a rajada não fica esperando para sempre.
function nextFrame() {
  return new Promise((ok) => {
    const t = setTimeout(ok, 120);
    const done = () => { clearTimeout(t); ok(); };
    if (!TEST && cam.requestVideoFrameCallback) cam.requestVideoFrameCallback(done);
    else requestAnimationFrame(() => requestAnimationFrame(done));
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
    if (motor) motor.setLut(lut);
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
const motorAtivo = () => state.modo !== "lut" && !!motor && motor.float;
const opcoesMotor = () => ({ preto: state.preto, mix: state.mix, raizen: state.lutId === "none" });

function notaMotor(info) {
  const q = info.frames > 1 ? ` · ${info.frames} quadros` : "";
  return `${info.noite ? "🌙 Noite" : "☀️ Dia"}${q} · preto ${Math.round(info.pretoAntes * 255)} → 0`;
}

async function takePhoto() {
  if (!ready() || state.busy) return;
  state.busy = true;
  const flash = state.light && state.facing === "user";
  try {
    // Pré-flash: a tela acende 600 ms antes, para a câmera baixar o ISO e acertar
    // a exposição com a luz da tela (sem rosto estourado).
    if (flash) { $("#flash").classList.add("on"); await sleep(600); }
    let canvas, nota = "";
    if (motorAtivo()) {
      const r = await fotoMotor(flash);
      canvas = r.canvas;
      nota = notaMotor(r.info);
    } else {
      const [sw, sh] = sourceSize();
      out.resize(sw, sh);
      out.smooth(false);
      out.draw(source(), { mirror: mirrored(), mix: state.mix });
      canvas = out.canvas;
    }
    $("#flash").classList.remove("on");
    const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.95));
    if (motor) motor.release();
    $("#msg").hidden = true;
    showResult(blob, `Revelador_${stamp()}.jpg`, "image/jpeg");
    if (nota) msg(nota, 3000);
  } catch (e) {
    msg("Falha ao processar a foto: " + e.message, 5000);
  } finally {
    $("#flash").classList.remove("on");
    state.busy = false;
  }
}

// A cena do visor está escura? (decide o Noite no modo Auto)
function escuroAgora() {
  const [sw, sh] = sourceSize();
  const l = lumaOf(source(), sw, sh, 160);
  let s = 0;
  for (const v of l.y) s += v;
  return s / l.y.length < 0.22 * 255;
}

// Visor: com a Luz ou no Noite, rajada de 12 quadros alinhados e somados com a
// tela acesa; depois o motor revela.
async function fotoMotor(flash) {
  const [sw, sh] = sourceSize();
  const noite = state.modo === "noite" || (state.modo === "auto" && (flash || escuroAgora()));
  const n = noite ? MOTOR.noite.quadros : MOTOR.dia.quadros;
  msg(n > 1 ? `Segure firme… (${n} quadros)` : "Revelando…", 0);
  const ref = await grab();
  motor.begin(ref, sw, sh, { mirror: mirrored() });
  const align = n > 1 ? makeAligner(ref, sw, sh) : null;
  const shifts = [];
  for (let i = 1; i < n; i++) {
    await nextFrame();
    const cur = await grab(i);
    const { dx, dy } = align(cur);
    shifts.push([dx, dy]);
    motor.add(cur, dx, dy);
  }
  $("#flash").classList.remove("on");
  msg("Revelando…", 0);
  const r = await motor.develop({ noite, ...opcoesMotor() });
  fotoMotor.last = { shifts, ...r.info };
  return r;
}

// Quadro atual da câmera (no teste, um quadro sintético da rajada).
async function grab(i = 0) {
  if (!TEST) return cam;
  return testBurstFrame(i);
}

// ---------------------------------------------------------------- MÁXIMA e galeria
// A foto vem da câmera do próprio iPhone (foco, HDR, Deep Fusion e o flash dela)
// ou da galeria; o motor e a LUT entram depois, na resolução cheia.
const MAX_PIXELS = 16777216;   // maior canvas que o Safari do iPhone aceita (16,7 MP)

// JPEG mínimo 16x8 (metade esquerda vermelha, direita azul) com EXIF 6: gerado
// pelo gerar_recursos.py para descobrir se o navegador já gira a foto sozinho.
const ORIENT_TEST = "/9j/4AAQSkZJRgABAQAAAQABAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAYAAAAAAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAAIABADAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8X6/ynP8Av4PyHr/1UD/zXz//2Q==";

// onload em vez de img.decode(): o decode() às vezes não responde (visto no Chromium).
function loadImage(src) {
  return new Promise((ok, fail) => {
    const img = new Image();
    img.onload = () => ok(img);
    img.onerror = () => fail(new Error("a imagem não abriu"));
    img.src = src;
  });
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

// A foto (arquivo) aberta, com o tamanho de pé e a orientação que falta aplicar.
async function fotoDePe(blob, { forceOrient = 0 } = {}) {
  const info = await jpegInfo(blob);
  const url = URL.createObjectURL(blob);
  try {
    const img = await loadImage(url);
    let w = img.naturalWidth, h = img.naturalHeight, orient = 1;
    if (info) {
      [w, h] = (forceOrient || info.orient) >= 5 ? [info.h, info.w] : [info.w, info.h];
      orient = forceOrient || ((await browserRotates()) ? 1 : info.orient);
    }
    const k = Math.min(1, Math.sqrt(MAX_PIXELS / (w * h)));
    return { img, orient, w: Math.floor(w * k), h: Math.floor(h * k), original: [w, h] };
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Só a LUT, pixel a pixel (modo SÓ LUT).
async function lutPhoto(blob, opts) {
  const f = await fotoDePe(blob, opts);
  out.resize(f.w, f.h);
  out.smooth(f.w !== f.original[0]);
  out.draw(f.img, { orient: f.orient, mix: state.mix });
  return { canvas: out.canvas, w: f.w, h: f.h, original: f.original };
}

// O motor inteiro (sem rajada: a câmera do iPhone já juntou os quadros dela).
async function motorPhoto(blob) {
  const f = await fotoDePe(blob);
  motor.begin(f.img, f.w, f.h, { orient: f.orient });
  const noite = state.modo === "auto" ? null : state.modo === "noite";
  const r = await motor.develop({ noite, ...opcoesMotor() });
  return { canvas: r.canvas, w: f.w, h: f.h, original: f.original, info: r.info };
}

async function processPhotoFile(file) {
  if (!file || state.busy) return;
  state.busy = true;
  msg(motorAtivo() ? "Revelando a foto…" : "Aplicando a LUT na foto…", 0);
  try {
    const r = motorAtivo() ? await motorPhoto(file) : await lutPhoto(file);
    const blob = await new Promise((ok) => r.canvas.toBlob(ok, "image/jpeg", 0.95));
    if (motor) motor.release();
    $("#msg").hidden = true;
    showResult(blob, `Revelador_${stamp()}.jpg`, "image/jpeg");
    const reduzida = r.w !== r.original[0] ? " (reduzida: o Safari não passa de 16,7 MP)" : "";
    msg(`Foto de ${mp(r.w, r.h)} MP · ${r.info ? notaMotor(r.info) : "só a LUT"}${reduzida}.`, 4000);
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
function syncUi() {
  $("#kelvin").value = state.kelvin;
  $("#ringsize").value = state.ring;
  $("#mirror").checked = state.mirror;
  $("#preto").value = Math.round(state.preto * 100);
  $("#pretoval").textContent = Math.round(state.preto * 100) + "%";
  $("#b-modo").textContent = ROTULO[state.modo];
  $("#b-modo").classList.toggle("on", state.modo !== "lut");
  $("#pretorow").hidden = state.modo === "lut";
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
      msg("MÁXIMA: o disparo abre a câmera do iPhone (foco, HDR e flash dela). O motor e a LUT entram logo depois, na resolução cheia.", 5000);
  }));
  $("#b-modo").onclick = () => {
    if (state.busy) return;
    state.modo = MODOS[(MODOS.indexOf(state.modo) + 1) % MODOS.length];
    syncUi();
    savePrefs();
    if (state.modo !== "lut" && !(motor && motor.float)) msg("Este navegador não tem a GPU em ponto flutuante: só a LUT funciona aqui.", 4000);
    else msg(AVISO[state.modo], 4000);
  };
  $("#preto").oninput = (e) => { state.preto = +e.target.value / 100; syncUi(); savePrefs(); };
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
// A amostra no lugar da câmera: confere o shader com o Revelador (Python), a
// MÁXIMA com foto deitada (EXIF 6) e o motor numa cena de dia e de noite.
function testBurstFrame(i) {
  const w = testImage.naturalWidth, h = testImage.naturalHeight;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const shifts = testBurstFrame.shifts || (testBurstFrame.shifts = [[0, 0], [3.4, -2.2], [-4.1, 1.6], [1.8, 4.3],
    [-2.6, -3.7], [5.2, 0.9], [-0.7, -5.1], [2.9, 2.6], [-3.3, -1.4], [1.1, -3.9], [4.4, 3.1], [-1.9, 2.2]]);
  const [dx, dy] = shifts[i % shifts.length];
  ctx.drawImage(testImage, dx, dy);         // o conteúdo aparece deslocado de (dx, dy)
  const veu = testBurstFrame.veu, ruido = testBurstFrame.noise;
  if (veu || ruido) {
    // Véu: a cena escura e acinzentada, como o iPhone entrega com o flash da tela.
    const img = ctx.getImageData(0, 0, w, h), d = img.data;
    let seed = 1234 + i * 977;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    for (let k = 0; k < d.length; k += 4) {
      const g = ruido ? (rnd() + rnd() + rnd() - 1.5) * 2 * ruido : 0;   // ~gaussiano
      for (let c3 = 0; c3 < 3; c3++) {
        const v = veu ? veu.lift + veu.ganho * d[k + c3] : d[k + c3];
        d[k + c3] = v + g * (c3 === 1 ? 0.9 : c3 === 2 ? 1.1 : 1);
      }
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
  const ref = await loadImage("teste/ref-raizen-33.png");
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(ref, 0, 0);
  return ctx.getImageData(0, 0, w, h).data;
}

// MÁXIMA (só LUT): a foto deitada com EXIF 6 tem de sair de pé e igual à LUT do
// Revelador, pelo caminho do navegador e pelo giro no shader (sem o EXIF).
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

// Luminância (0..255): ponto preto (1,5%), média e contraste (desvio padrão).
function lumaStats(data) {
  const hist = new Uint32Array(256);
  let s = 0, s2 = 0, n = 0;
  for (let k = 0; k < data.length; k += 4) {
    const y = 0.299 * data[k] + 0.587 * data[k + 1] + 0.114 * data[k + 2];
    hist[Math.round(y)]++; s += y; s2 += y * y; n++;
  }
  let acc = 0, p = 0;
  for (; p < 255; p++) { acc += hist[p]; if (acc >= n * 0.015) break; }
  const m = s / n;
  return { preto: p, media: +m.toFixed(1), contraste: +Math.sqrt(s2 / n - m * m).toFixed(1) };
}

// Uma foto sintética pelo motor: n quadros da rajada de teste, alinhados e somados.
async function runMotor(n, noite) {
  const [w, h] = sourceSize(), ref = testBurstFrame(0);
  motor.begin(ref, w, h, {});
  const align = n > 1 ? makeAligner(ref, w, h) : null;
  for (let i = 1; i < n; i++) { const cur = testBurstFrame(i); const { dx, dy } = align(cur); motor.add(cur, dx, dy); }
  const t0 = performance.now();
  const r = await motor.develop({ noite, preto: 0.75, mix: 1, raizen: false });
  return { data: readCanvas(r.canvas), info: r.info, ms: Math.round(performance.now() - t0) };
}

async function verifyMotor() {
  await selectLut("raizen-33");
  const erro = (a, b) => {   // diferença média na região central (as bordas recebem menos quadros)
    const [w, h] = sourceSize();
    let s = 0, n = 0;
    for (let y = Math.round(h * 0.1); y < h * 0.9; y++)
      for (let x = Math.round(w * 0.1); x < w * 0.9; x++)
        for (let k = 0; k < 3; k++) { s += Math.abs(a[(y * w + x) * 4 + k] - b[(y * w + x) * 4 + k]); n++; }
    return +(s / n).toFixed(2);
  };
  const [w, h] = sourceSize();
  Object.assign(testBurstFrame, { noise: 0, veu: null });
  const soLut = await refPixels(w, h);
  const dia = await runMotor(1, false);
  // Noite: a mesma cena com o preto levantado para 34 (véu cinza), como o iPhone
  // entrega com o flash da tela; o branco continua perto de 255.
  testBurstFrame.veu = { lift: 34, ganho: 0.85 };
  const entrada = readCanvas(testBurstFrame(0));
  const limpa = await runMotor(1, true);
  testBurstFrame.noise = 10;
  const um = await runMotor(1, true);
  const doze = await runMotor(12, true);
  Object.assign(testBurstFrame, { noise: 0, veu: null });
  motor.release();
  return {
    dia: { ms: dia.ms, soLut: lumaStats(soLut), motor: lumaStats(dia.data) },
    noite: { entrada: lumaStats(entrada), motor: lumaStats(limpa.data), ms: limpa.ms, info: limpa.info },
    rajada: { erroUmQuadro: erro(um.data, limpa.data), erroDozeQuadros: erro(doze.data, limpa.data) },
  };
}

// ---------------------------------------------------------------- início
async function main() {
  loadPrefs();
  view = new Renderer($("#view"), true);
  out = new Renderer(document.createElement("canvas"), false);
  try { motor = new Motor(); } catch (e) { motor = null; }
  bindUi();
  syncUi();
  for (const item of await dbAll()) luts.set(item.id, { ...item, data: Float32Array.from(item.data) });
  let saved = "raizen-33";
  try { saved = localStorage.getItem("lut") || saved; } catch {}
  if (!BUILTIN.find((b) => b.id === saved) && !luts.has(saved)) saved = "raizen-33";
  if (TEST) {
    testImage = await loadImage("teste/amostra.png");
    Object.assign(window, { __verificar: verify, __testarMaxima: verifyNative, __testarMotor: verifyMotor,
                            __estado: state });
  }
  await selectLut(saved);
  applyLight();
  resizeView();
  await startCamera();
  keepAwake();
  frame();
}

main().catch((e) => msg("Erro ao iniciar: " + e.message, 0));
