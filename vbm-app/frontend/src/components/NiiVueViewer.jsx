import { useEffect, useRef, useState } from 'react';
import * as nifti from 'nifti-reader-js';
import pako from 'pako';
import { useT } from '../i18n/LanguageContext.jsx';

// ─── 2D NIfTI viewer (axial / coronal / sagittal) ───────────────────────────
//
// Replacement for NiiVue: we use nifti-reader-js + canvas2d for full control
// over the render — no WebGL, no 3D library overhead, exactly the 3
// anatomical views with a red overlay of the binary mask.
//
// Works in spatial RAS (x=L→R, y=P→A, z=I→S). Assumes images are in standard
// orientation (raw T1 without reorientation + nnUNet mask in the same space,
// which is what the pipeline produces).
//
// Performance: per-slice render ≈ 8-15ms on a 512×512 canvas.

const VIEWS = [
  { key: 'axial',    axis: 2 },   // slices on the Z axis (I-S)
  { key: 'coronal',  axis: 1 },   // slices on the Y axis (P-A)
  { key: 'sagittal', axis: 0 },   // slices on the X axis (L-R)
];

/** Download + decompress + parse a .nii.gz → { header, data, dims } */
async function fetchVolume(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  const buf = await res.arrayBuffer();

  // .nii.gz → decompress; .nii → use directly
  let raw;
  if (nifti.isCompressed(buf)) {
    raw = pako.inflate(new Uint8Array(buf)).buffer;
  } else {
    raw = buf;
  }
  if (!nifti.isNIFTI(raw)) {
    throw new Error('Archivo no es NIfTI válido');
  }
  const header = nifti.readHeader(raw);
  const image  = nifti.readImage(header, raw);

  // Type the array according to datatypeCode
  const data = toTypedArray(image, header.datatypeCode);
  const dims = [header.dims[1], header.dims[2], header.dims[3]];
  return { header, data, dims };
}

function toTypedArray(buffer, code) {
  switch (code) {
    case nifti.NIFTI1.TYPE_UINT8:   return new Uint8Array(buffer);
    case nifti.NIFTI1.TYPE_INT16:   return new Int16Array(buffer);
    case nifti.NIFTI1.TYPE_INT32:   return new Int32Array(buffer);
    case nifti.NIFTI1.TYPE_FLOAT32: return new Float32Array(buffer);
    case nifti.NIFTI1.TYPE_FLOAT64: return new Float64Array(buffer);
    case nifti.NIFTI1.TYPE_INT8:    return new Int8Array(buffer);
    case nifti.NIFTI1.TYPE_UINT16:  return new Uint16Array(buffer);
    case nifti.NIFTI1.TYPE_UINT32:  return new Uint32Array(buffer);
    default: return new Float32Array(buffer);
  }
}

/** Extract a 2D slice from a 3D volume along the given axis. Returns {w, h, pixels}. */
function sliceVolume(volume, axis, sliceIdx) {
  const [nx, ny, nz] = volume.dims;
  const data = volume.data;
  let w, h, out;

  if (axis === 2) {           // AXIAL (fixed Z) → XY plane
    w = nx; h = ny;
    out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        out[y * w + x] = data[sliceIdx * nx * ny + y * nx + x];
      }
    }
  } else if (axis === 1) {    // CORONAL (fixed Y) → XZ plane
    w = nx; h = nz;
    out = new Float32Array(w * h);
    for (let z = 0; z < h; z++) {
      for (let x = 0; x < w; x++) {
        out[z * w + x] = data[z * nx * ny + sliceIdx * nx + x];
      }
    }
  } else {                    // SAGITTAL (fixed X) → YZ plane
    w = ny; h = nz;
    out = new Float32Array(w * h);
    for (let z = 0; z < h; z++) {
      for (let y = 0; y < w; y++) {
        out[z * w + y] = data[z * nx * ny + y * nx + sliceIdx];
      }
    }
  }
  return { w, h, pixels: out };
}

/**
 * Compute the brain's 3D bounding box in the full volume (min/max indices
 * where intensity > threshold). It is computed ONCE per volume and reused
 * on every render — no per-slice cost.
 *
 * Returns [[xMin, xMax], [yMin, yMax], [zMin, zMax]] with a 4-voxel margin.
 */
function computeBoundingBox(volume) {
  const [nx, ny, nz] = volume.dims;
  const data = volume.data;
  // Robust threshold: 5% of the 99th percentile (filters background noise)
  const sample = new Float32Array(Math.min(30000, data.length));
  for (let i = 0; i < sample.length; i++) sample[i] = data[(Math.random() * data.length) | 0];
  sample.sort();
  const p99 = sample[(sample.length * 0.99) | 0];
  const thr = p99 * 0.05;

  let xMin = nx, xMax = 0, yMin = ny, yMax = 0, zMin = nz, zMax = 0;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if (data[z * nx * ny + y * nx + x] > thr) {
          if (x < xMin) xMin = x; if (x > xMax) xMax = x;
          if (y < yMin) yMin = y; if (y > yMax) yMax = y;
          if (z < zMin) zMin = z; if (z > zMax) zMax = z;
        }
      }
    }
  }
  const margin = 4;
  return [
    [Math.max(0, xMin - margin), Math.min(nx - 1, xMax + margin)],
    [Math.max(0, yMin - margin), Math.min(ny - 1, yMax + margin)],
    [Math.max(0, zMin - margin), Math.min(nz - 1, zMax + margin)],
  ];
}

/** Crop a 2D slice to a rectangular region [[xMin,xMax],[yMin,yMax]] */
function cropSlice(slice, bbox) {
  const { w, h, pixels } = slice;
  const [[x0, x1], [y0, y1]] = bbox;
  const cw = x1 - x0 + 1;
  const ch = y1 - y0 + 1;
  const out = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      out[y * cw + x] = pixels[(y + y0) * w + (x + x0)];
    }
  }
  return { w: cw, h: ch, pixels: out };
}

/** 2D bounding box corresponding to a given view (axis ∈ {0,1,2}). */
function sliceBboxFor(axis, bbox3d) {
  const [[x0, x1], [y0, y1], [z0, z1]] = bbox3d;
  if (axis === 2) return [[x0, x1], [y0, y1]];   // axial: XY
  if (axis === 1) return [[x0, x1], [z0, z1]];   // coronal: XZ
  return [[y0, y1], [z0, z1]];                    // sagittal: YZ
}

/** Draw T1 in grayscale + red mask onto a canvas. */
function renderSlice(canvas, t1Slice, maskSlice, showMask, opacity, t1Min, t1Max) {
  const { w, h, pixels: t1 } = t1Slice;
  canvas.width  = w;
  canvas.height = h;

  const ctx     = canvas.getContext('2d');
  const imgData = ctx.createImageData(w, h);
  const range   = (t1Max - t1Min) || 1;

  for (let i = 0; i < t1.length; i++) {
    // Normalize T1 to 0-255
    let v = (t1[i] - t1Min) / range;
    v = Math.max(0, Math.min(1, v));
    let r = (v * 255) | 0;
    let g = r, b = r;

    // Red overlay of the mask (alpha-blending)
    if (showMask && maskSlice) {
      const m = maskSlice.pixels[i];
      if (m > 0) {
        // Blend: pixel = (1-α)·gray + α·red
        r = (r * (1 - opacity) + 255 * opacity) | 0;
        g = (g * (1 - opacity)) | 0;
        b = (b * (1 - opacity)) | 0;
      }
    }

    const j = i * 4;
    imgData.data[j]     = r;
    imgData.data[j + 1] = g;
    imgData.data[j + 2] = b;
    imgData.data[j + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

/** min/max stats ignoring outliers (1st-99th percentile for better contrast) */
function computeIntensityRange(data) {
  // Sampling: 50k random voxels are enough for stable percentiles
  const n = data.length;
  const sample = new Float32Array(Math.min(50000, n));
  for (let i = 0; i < sample.length; i++) {
    sample[i] = data[(Math.random() * n) | 0];
  }
  sample.sort();
  const lo = sample[(sample.length * 0.01) | 0];
  const hi = sample[(sample.length * 0.99) | 0];
  return [lo, hi];
}

const NiiVueViewer = ({ t1Url, maskUrl, onError }) => {
  const t = useT();
  const canvasRef = useRef(null);
  const t1Ref     = useRef(null);
  const maskRef   = useRef(null);
  const rangeRef  = useRef([0, 1]);
  const bboxRef   = useRef(null);   // brain 3D bounding box (in voxels)

  const [viewKey,     setViewKey]     = useState('axial');
  const [sliceIdx,    setSliceIdx]    = useState(0);
  const [maxSlice,    setMaxSlice]    = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity,     setOpacity]     = useState(0.5);
  const [loading,     setLoading]     = useState(true);
  const [errorMsg,    setErrorMsg]    = useState(null);

  // ── Initial load of both volumes in parallel ──────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        setErrorMsg(null);
        const [t1Vol, maskVol] = await Promise.all([
          fetchVolume(t1Url),
          fetchVolume(maskUrl).catch((e) => {
            console.warn('[Viewer] máscara no cargada:', e);
            return null;
          }),
        ]);
        if (cancelled) return;

        t1Ref.current   = t1Vol;
        maskRef.current = maskVol;
        rangeRef.current = computeIntensityRange(t1Vol.data);
        bboxRef.current  = computeBoundingBox(t1Vol);

        // Default to the central axial slice
        const axis = VIEWS.find((v) => v.key === viewKey).axis;
        const sliceCount = t1Vol.dims[axis];
        setMaxSlice(sliceCount - 1);
        setSliceIdx(Math.floor(sliceCount / 2));
        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        console.error('[Viewer] Error:', err);
        setErrorMsg(err.message || String(err));
        setLoading(false);
        onError?.(err.message || String(err));
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [t1Url, maskUrl]);

  // ── View change: reset slice to the center of the new axis ──────────
  useEffect(() => {
    if (!t1Ref.current) return;
    const axis = VIEWS.find((v) => v.key === viewKey).axis;
    const count = t1Ref.current.dims[axis];
    setMaxSlice(count - 1);
    setSliceIdx(Math.floor(count / 2));
  }, [viewKey]);

  // ── Reactive render whenever slice/overlay/opacity changes ──────────
  useEffect(() => {
    if (!t1Ref.current || !canvasRef.current) return;
    const axis = VIEWS.find((v) => v.key === viewKey).axis;
    let t1Slice   = sliceVolume(t1Ref.current, axis, sliceIdx);
    let maskSlice = maskRef.current ? sliceVolume(maskRef.current, axis, sliceIdx) : null;

    // Auto-crop to the brain bounding box — removes the empty FOV and the
    // image fills the whole canvas (≈ 2× larger visually).
    if (bboxRef.current) {
      const slice2dBbox = sliceBboxFor(axis, bboxRef.current);
      t1Slice = cropSlice(t1Slice, slice2dBbox);
      if (maskSlice) maskSlice = cropSlice(maskSlice, slice2dBbox);
    }

    const [lo, hi]  = rangeRef.current;
    renderSlice(canvasRef.current, t1Slice, maskSlice, showOverlay, opacity, lo, hi);
  }, [viewKey, sliceIdx, showOverlay, opacity, loading]);

  return (
    <div className="niivue-wrap">
      <div className="niivue-toolbar">
        <label className="niivue-field">
          <span className="niivue-label">{t('viewer.view')}</span>
          <select
            value={viewKey}
            onChange={(e) => setViewKey(e.target.value)}
            className="niivue-select"
          >
            {VIEWS.map((v) => (
              <option key={v.key} value={v.key}>{t(`viewer.slice.${v.key}`)}</option>
            ))}
          </select>
        </label>

        <label className="niivue-field niivue-slider">
          <span className="niivue-label">{t('viewer.slice')}</span>
          <input
            type="range"
            min="0"
            max={maxSlice}
            value={sliceIdx}
            onChange={(e) => setSliceIdx(parseInt(e.target.value, 10))}
            disabled={loading || !!errorMsg}
          />
          <span className="niivue-value">{sliceIdx} / {maxSlice}</span>
        </label>

        <label className="niivue-field niivue-toggle">
          <input
            type="checkbox"
            checked={showOverlay}
            disabled={!maskRef.current}
            onChange={(e) => setShowOverlay(e.target.checked)}
          />
          <span>{t('viewer.showMask')}</span>
        </label>

        <label className="niivue-field niivue-slider">
          <span className="niivue-label">{t('viewer.opacity')}</span>
          <input
            type="range"
            min="0" max="1" step="0.05"
            value={opacity}
            disabled={!showOverlay || !maskRef.current}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
          />
          <span className="niivue-value">{Math.round(opacity * 100)}%</span>
        </label>
      </div>

      <div className="niivue-canvas-wrap">
        {loading && <div className="niivue-loading">{t('viewer.loading')}</div>}
        {errorMsg && (
          <div className="niivue-loading" style={{ color: '#f87171' }}>
            ⚠ {errorMsg}
          </div>
        )}
        <canvas ref={canvasRef} className="niivue-canvas" />
      </div>

      <div className="niivue-hint">{t('viewer.hint')}</div>
    </div>
  );
};

export default NiiVueViewer;
