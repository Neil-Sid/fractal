// WebGL2 renderer: compute pass (perturbation, temporal reuse) + display pass (palette).
'use strict';

const REF_TEX_W = 2048;

class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      antialias: false, depth: false, stencil: false, alpha: false,
      preserveDrawingBuffer: false, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('This browser cannot render to float textures (EXT_color_buffer_float missing).');
    }

    this.progCompute = this._program(SHADER_VS, SHADER_COMPUTE_FS);
    this.progDisplay = this._program(SHADER_VS, SHADER_DISPLAY_FS);
    this.uC = this._uniforms(this.progCompute, [
      'uRef', 'uRefLen', 'uRefEscaped', 'uPrev', 'uHavePrev', 'uRes', 'uOffsetPx', 'uPm', 'uEc',
      'uMaxIter', 'uFrame', 'uFraction', 'uPrevScale', 'uPrevShift', 'uRefGrew',
    ]);
    this.uD = this._uniforms(this.progDisplay, ['uMu', 'uPal', 'uDensityInv', 'uPhase', 'uInterior']);
    this.progAnalyze = this._program(SHADER_VS, SHADER_ANALYZE_FS);
    this.uA = this._uniforms(this.progAnalyze, ['uMu', 'uRes', 'uCells']);
    this.anaTex = null;
    this.anaFbo = null;
    this.anaCells = [0, 0];
    this.anaBuf = null;

    this.vao = gl.createVertexArray();

    this.refTex = gl.createTexture();
    this.refRows = 0;
    this._setupTex(this.refTex);

    this.palTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.palTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.mu = [null, null];
    this.fbo = [null, null];
    this.cur = 0;
    this.width = 0;
    this.height = 0;
    this.frame = 0;
  }

  _setupTex(tex) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  _shader(type, src) {
    const gl = this.gl;
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error('Shader compile failed: ' + gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  _program(vs, fs) {
    const gl = this.gl;
    const p = gl.createProgram();
    gl.attachShader(p, this._shader(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, this._shader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('Program link failed: ' + gl.getProgramInfoLog(p));
    }
    return p;
  }

  _uniforms(prog, names) {
    const out = {};
    for (const n of names) out[n] = this.gl.getUniformLocation(prog, n);
    return out;
  }

  resize(w, h) {
    if (w === this.width && h === this.height) return false;
    const gl = this.gl;
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;
    for (let i = 0; i < 2; i++) {
      if (this.mu[i]) { gl.deleteTexture(this.mu[i]); gl.deleteFramebuffer(this.fbo[i]); }
      const tex = gl.createTexture();
      this._setupTex(tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, null);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Float framebuffer is not complete on this GPU.');
      }
      this.mu[i] = tex;
      this.fbo[i] = fbo;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return true;
  }

  // orbit: Float32Array of interleaved (re, im) pairs; len: number of valid pairs.
  setReference(orbit, len) {
    const gl = this.gl;
    const rows = Math.max(1, Math.ceil(len / REF_TEX_W));
    gl.bindTexture(gl.TEXTURE_2D, this.refTex);
    if (rows !== this.refRows) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, REF_TEX_W, rows, 0, gl.RG, gl.FLOAT, null);
      this.refRows = rows;
    }
    const fullRows = Math.floor(len / REF_TEX_W);
    if (fullRows > 0) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, REF_TEX_W, fullRows, gl.RG, gl.FLOAT,
        orbit.subarray(0, fullRows * REF_TEX_W * 2));
    }
    const rem = len - fullRows * REF_TEX_W;
    if (rem > 0) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, fullRows, rem, 1, gl.RG, gl.FLOAT,
        orbit.subarray(fullRows * REF_TEX_W * 2, len * 2));
    }
  }

  setPalette(rgba, n) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.palTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, n, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  }

  // Runs the compute pass into the next buffer (in horizontal bands so no single draw call
  // runs long enough to trip a GPU watchdog), then the display pass to the canvas.
  render(p) {
    const gl = this.gl;
    const w = this.width, h = this.height;
    const next = 1 - this.cur;

    gl.bindVertexArray(this.vao);
    gl.useProgram(this.progCompute);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[next]);
    gl.viewport(0, 0, w, h);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.refTex);
    gl.uniform1i(this.uC.uRef, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.mu[this.cur]);
    gl.uniform1i(this.uC.uPrev, 1);

    gl.uniform1i(this.uC.uRefLen, p.refLen);
    gl.uniform1i(this.uC.uRefEscaped, p.refEscaped ? 1 : 0);
    gl.uniform1i(this.uC.uHavePrev, p.havePrev ? 1 : 0);
    gl.uniform2f(this.uC.uRes, w, h);
    gl.uniform2f(this.uC.uOffsetPx, p.offsetPx[0], p.offsetPx[1]);
    gl.uniform1f(this.uC.uPm, p.pm);
    gl.uniform1i(this.uC.uEc, p.ec);
    gl.uniform1i(this.uC.uMaxIter, p.maxIter);
    gl.uniform1i(this.uC.uFrame, this.frame);
    gl.uniform1f(this.uC.uFraction, p.fraction);
    gl.uniform1f(this.uC.uPrevScale, p.prevScale);
    gl.uniform2f(this.uC.uPrevShift, p.prevShift[0], p.prevShift[1]);
    gl.uniform1i(this.uC.uRefGrew, p.refGrew ? 1 : 0);

    const bands = p.bands || 1;
    gl.enable(gl.SCISSOR_TEST);
    for (let b = 0; b < bands; b++) {
      const y0 = Math.floor((b * h) / bands);
      const y1 = Math.floor(((b + 1) * h) / bands);
      gl.scissor(0, y0, w, y1 - y0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.disable(gl.SCISSOR_TEST);

    this.cur = next;
    this.frame++;
    this.present(p);
  }

  present(p) {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.useProgram(this.progDisplay);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.mu[this.cur]);
    gl.uniform1i(this.uD.uMu, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.palTex);
    gl.uniform1i(this.uD.uPal, 1);
    gl.uniform1f(this.uD.uDensityInv, 1 / p.density);
    gl.uniform1f(this.uD.uPhase, p.phase);
    gl.uniform3f(this.uD.uInterior, p.interior[0], p.interior[1], p.interior[2]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
