import { useCallback, useEffect, useRef, useState } from "react";
import { extractLaneFrame, getLaneRois, saveLaneRois, testLanes } from "./api";

// Draw lane ROIs + a physical gate line on a real frame from each gate camera.
// Draw once, save, and every future job reuses it. Back camera is excluded.
const CAMERAS = ["front", "left", "right"];
const LANE_COLORS = { 1: "#1d9e75", 2: "#e24b4a" };
const GATE_COLOR = "#378add";

const emptyCam = () => ({
  imageUrl: "",
  w: 0,
  h: 0,
  lanes: { 1: [], 2: [] },
  gate: [], // [[x,y],[x,y]]
  savedAt: "",
  videoFile: null, // kept so we can run the lane test
  testTrucks: null // [{x1,y1,x2,y2,type,confidence,lane}] from the last test
});

// Map a mouse event to NATIVE image pixels. The SVG viewBox is the native size,
// so getScreenCTM().inverse() gives native coords regardless of display scale.
function svgPoint(svg, evt) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  return [Math.round(p.x), Math.round(p.y)];
}

export default function LaneSetup({ onClose }) {
  const [camera, setCamera] = useState("front");
  const [mode, setMode] = useState("1"); // "1" | "2" | "gate"
  const [data, setData] = useState(() => ({
    front: emptyCam(),
    left: emptyCam(),
    right: emptyCam()
  }));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const svgRef = useRef(null);

  const cur = data[camera];

  const patchCam = useCallback((cam, patch) => {
    setData((prev) => ({ ...prev, [cam]: { ...prev[cam], ...patch } }));
  }, []);

  // Prefill any previously-saved ROI when switching cameras.
  useEffect(() => {
    let cancelled = false;
    getLaneRois(camera)
      .then((res) => {
        if (cancelled || !res?.exists || !res.roi) return;
        const roi = res.roi;
        patchCam(camera, {
          w: roi.image_width || 0,
          h: roi.image_height || 0,
          lanes: {
            1: (roi.lanes?.["1"] || []).map((p) => [p[0], p[1]]),
            2: (roi.lanes?.["2"] || []).map((p) => [p[0], p[1]])
          },
          gate: (roi.gate_line || []).map((p) => [p[0], p[1]]),
          savedAt: "previously saved"
        });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [camera, patchCam]);

  async function onPickVideo(evt) {
    const file = evt.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const url = await extractLaneFrame(file, camera);
      patchCam(camera, { imageUrl: url, videoFile: file, testTrucks: null });
    } catch (e) {
      setError(`Could not extract a frame: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setBusy(false);
    }
  }

  function onImageLoad(evt) {
    patchCam(camera, { w: evt.target.naturalWidth, h: evt.target.naturalHeight });
  }

  function onCanvasClick(evt) {
    if (!cur.imageUrl || !svgRef.current) return;
    const [x, y] = svgPoint(svgRef.current, evt);
    if (mode === "gate") {
      const gate = cur.gate.length >= 2 ? [[x, y]] : [...cur.gate, [x, y]];
      patchCam(camera, { gate });
    } else {
      const lanes = { ...cur.lanes, [mode]: [...cur.lanes[mode], [x, y]] };
      patchCam(camera, { lanes });
    }
  }

  function undo() {
    if (mode === "gate") {
      patchCam(camera, { gate: cur.gate.slice(0, -1) });
    } else {
      patchCam(camera, { lanes: { ...cur.lanes, [mode]: cur.lanes[mode].slice(0, -1) } });
    }
  }

  function resetCurrent() {
    if (mode === "gate") patchCam(camera, { gate: [] });
    else patchCam(camera, { lanes: { ...cur.lanes, [mode]: [] } });
  }

  async function runTest() {
    setError("");
    if (!cur.videoFile) {
      setError("Upload the camera video first, then draw the lanes, then test.");
      return;
    }
    if (cur.lanes[1].length < 3 || cur.lanes[2].length < 3) {
      setError("Draw at least 3 points for BOTH Lane 1 and Lane 2 before testing.");
      return;
    }
    setBusy(true);
    try {
      const res = await testLanes(cur.videoFile, camera, { 1: cur.lanes[1], 2: cur.lanes[2] });
      const patch = { testTrucks: res.trucks || [] };
      if (res.frame_jpeg_base64) {
        patch.imageUrl = `data:image/jpeg;base64,${res.frame_jpeg_base64}`;
        if (res.image_width) patch.w = res.image_width;
        if (res.image_height) patch.h = res.image_height;
      }
      patchCam(camera, patch);
    } catch (e) {
      setError(`Lane test failed: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    setError("");
    if (cur.lanes[1].length < 3 || cur.lanes[2].length < 3) {
      setError("Draw at least 3 points for BOTH Lane 1 and Lane 2 before saving.");
      return;
    }
    setBusy(true);
    try {
      await saveLaneRois({
        camera,
        image_width: cur.w,
        image_height: cur.h,
        lanes: { 1: cur.lanes[1], 2: cur.lanes[2] },
        gate_line: cur.gate.length === 2 ? cur.gate : null
      });
      patchCam(camera, { savedAt: new Date().toLocaleTimeString() });
    } catch (e) {
      setError(`Save failed: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setBusy(false);
    }
  }

  const polyPoints = (pts) => pts.map((p) => p.join(",")).join(" ");

  return (
    <div className="lane-setup">
      <div className="lane-setup-head">
        <h2>Lane &amp; Gate Setup</h2>
        <button className="toggle-btn" onClick={onClose}>Close</button>
      </div>

      <p className="lane-setup-hint">
        Draw <b>Lane 1</b> and <b>Lane 2</b> polygons and the <b>gate line</b> on each
        camera. Saved once, reused for every future job. (Back camera is excluded.)
      </p>

      <div className="lane-tabs">
        {CAMERAS.map((c) => (
          <button
            key={c}
            className={c === camera ? "toggle-btn active" : "toggle-btn"}
            onClick={() => setCamera(c)}
          >
            {c.toUpperCase()}
            {data[c].savedAt ? " ✓" : ""}
          </button>
        ))}
      </div>

      <div className="lane-controls">
        <label className="file-btn">
          {cur.imageUrl ? "Replace frame" : "Upload video → grab frame"}
          <input type="file" accept="video/*" onChange={onPickVideo} hidden />
        </label>
        <span className="spacer" />
        <button className={mode === "1" ? "toggle-btn active" : "toggle-btn"} onClick={() => setMode("1")}>
          Lane 1 ({cur.lanes[1].length})
        </button>
        <button className={mode === "2" ? "toggle-btn active" : "toggle-btn"} onClick={() => setMode("2")}>
          Lane 2 ({cur.lanes[2].length})
        </button>
        <button className={mode === "gate" ? "toggle-btn active" : "toggle-btn"} onClick={() => setMode("gate")}>
          Gate line ({cur.gate.length}/2)
        </button>
        <button className="toggle-btn" onClick={undo} disabled={busy}>Undo</button>
        <button className="toggle-btn" onClick={resetCurrent} disabled={busy}>Reset</button>
        <button className="toggle-btn" onClick={runTest} disabled={busy}>Test lanes</button>
        <button className="toggle-btn active" onClick={save} disabled={busy}>Save {camera}</button>
      </div>

      {cur.testTrucks && (
        <div className="message-banner">
          Test: detected {cur.testTrucks.length} truck(s) —{" "}
          {[1, 2].map((l) => `Lane ${l}: ${cur.testTrucks.filter((t) => t.lane === l).length}`).join(", ")}
          {cur.testTrucks.some((t) => t.lane === null || t.lane === undefined)
            ? `, unassigned: ${cur.testTrucks.filter((t) => t.lane === null || t.lane === undefined).length}`
            : ""}
          . Each box is coloured by the lane it landed in — confirm it matches reality before saving.
        </div>
      )}

      {error && <div className="message-banner">{error}</div>}
      {cur.savedAt && !error && (
        <div className="message-banner">Saved {camera} lanes ({cur.savedAt}). It will be used on the next run.</div>
      )}

      <div className="lane-canvas">
        {!cur.imageUrl ? (
          <div className="lane-placeholder">
            {busy ? "Extracting frame…" : "Upload the camera video to get a frame to draw on."}
          </div>
        ) : (
          <div style={{ position: "relative", width: "100%" }}>
            <img
              src={cur.imageUrl}
              alt={`${camera} frame`}
              onLoad={onImageLoad}
              style={{ width: "100%", display: "block" }}
            />
            {cur.w > 0 && (
              <svg
                ref={svgRef}
                viewBox={`0 0 ${cur.w} ${cur.h}`}
                onClick={onCanvasClick}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  cursor: "crosshair"
                }}
              >
                {[1, 2].map((lane) => (
                  <g key={lane}>
                    <polygon
                      points={polyPoints(cur.lanes[lane])}
                      fill={LANE_COLORS[lane]}
                      fillOpacity="0.18"
                      stroke={LANE_COLORS[lane]}
                      strokeWidth="2"
                    />
                    {cur.lanes[lane].map((p, i) => (
                      <circle key={i} cx={p[0]} cy={p[1]} r="5" fill={LANE_COLORS[lane]} />
                    ))}
                  </g>
                ))}
                {cur.gate.length === 2 && (
                  <line
                    x1={cur.gate[0][0]}
                    y1={cur.gate[0][1]}
                    x2={cur.gate[1][0]}
                    y2={cur.gate[1][1]}
                    stroke={GATE_COLOR}
                    strokeWidth="3"
                    strokeDasharray="10 6"
                  />
                )}
                {cur.gate.map((p, i) => (
                  <circle key={`g${i}`} cx={p[0]} cy={p[1]} r="5" fill={GATE_COLOR} />
                ))}
                {(cur.testTrucks || []).map((t, i) => {
                  const stroke = t.lane ? LANE_COLORS[t.lane] : "#f0b000";
                  return (
                    <g key={`t${i}`}>
                      <rect
                        x={t.x1}
                        y={t.y1}
                        width={t.x2 - t.x1}
                        height={t.y2 - t.y1}
                        fill="none"
                        stroke={stroke}
                        strokeWidth="3"
                      />
                      <text x={t.x1 + 4} y={t.y1 + 22} fill={stroke} fontSize="20" fontWeight="700">
                        {t.lane ? `L${t.lane}` : "?"}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
