# PlateFlow Cross-Camera Aggregator — Problem & How I Want It Resolved

## Context

PlateFlow is a multi-camera truck detection + OCR pipeline (YOLO + ByteTrack + FastAPI).
Each camera video is processed as a **separate job** by the single-camera engine in
`main.py`. The engine does ROI lane-gating and emits per-truck records with:

- `camera`
- `lane_id` (from ROI gating)
- `track_id` (random, from ByteTrack)
- `first_seen_time_sec` (truck enters)
- `last_seen_time_sec` (truck leaves)
- `associated_info` (OCR fields + confidences)
- for gate cameras only: a **main class** — `truck_with_container` / `truck_without_container`

There are four cameras: **front**, **right**, **left** (the gate, outside) and **back**
(just inside the gate door). I want a separate aggregator module/endpoint — `main.py`
must not be touched.

---

## 1. Physical layout

- **FRONT** — at the gate door, narrow field of view, covers **one lane only**. The truck nose stops exactly here.
- **RIGHT** — beside FRONT, covers **both lanes**.
- **LEFT** — beside RIGHT, covers **both lanes** from the rear angle. The truck's last position (as it pulls away) is under LEFT.
- FRONT + RIGHT + LEFT are **concurrent** views of the same truck — same time window.
- **BACK** — mounted **just inside the gate door**. It sees the truck a few seconds after it crosses through.

<svg width="100%" viewBox="0 0 680 470" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <defs>
    <marker id="ar1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <text x="170" y="44" text-anchor="middle" font-size="12" fill="#888780">OUTSIDE</text>
  <text x="560" y="44" text-anchor="middle" font-size="12" fill="#888780">INSIDE</text>

  <rect x="318" y="64" width="14" height="306" rx="3" fill="#888780" opacity="0.6"/>
  <text x="325" y="58" text-anchor="middle" font-size="13" fill="#5F5E5A">gate door</text>

  <rect x="40" y="200" width="600" height="130" fill="#F1EFE8"/>
  <line x1="40" y1="265" x2="640" y2="265" stroke="#B4B2A9" stroke-dasharray="14 8" stroke-width="1"/>
  <text x="54" y="195" font-size="12" fill="#888780">Lane 1</text>
  <text x="54" y="284" font-size="12" fill="#888780">Lane 2</text>

  <line x1="290" y1="230" x2="100" y2="230" stroke="#B4B2A9" stroke-width="1" marker-end="url(#ar1)"/>
  <text x="295" y="194" text-anchor="end" font-size="12" fill="#888780">&#8592; truck enters</text>

  <rect x="136" y="208" width="180" height="44" rx="5" fill="#D3D1C7" opacity="0.5" stroke="#888780" stroke-width="1"/>
  <rect x="294" y="200" width="22" height="20" rx="3" fill="#B4B2A9" opacity="0.6" stroke="#888780" stroke-width="0.6"/>
  <text x="226" y="234" text-anchor="middle" font-size="12" fill="#5F5E5A">truck (stopped)</text>
  <line x1="318" y1="200" x2="318" y2="258" stroke="#639922" stroke-width="1.5" stroke-dasharray="3 2"/>
  <text x="312" y="197" text-anchor="end" font-size="12" fill="#3B6D11">nose at gate</text>

  <rect x="240" y="88" width="90" height="54" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.8"/>
  <text x="285" y="110" text-anchor="middle" font-size="14" font-weight="500" fill="#0F6E56">FRONT</text>
  <text x="285" y="128" text-anchor="middle" font-size="12" fill="#0F6E56">1 lane</text>

  <rect x="148" y="88" width="88" height="54" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.8"/>
  <text x="192" y="110" text-anchor="middle" font-size="14" font-weight="500" fill="#0F6E56">RIGHT</text>
  <text x="192" y="128" text-anchor="middle" font-size="12" fill="#0F6E56">both lanes</text>

  <rect x="50" y="88" width="88" height="54" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.8"/>
  <text x="94" y="110" text-anchor="middle" font-size="14" font-weight="500" fill="#0F6E56">LEFT</text>
  <text x="94" y="128" text-anchor="middle" font-size="12" fill="#0F6E56">both lanes, rear</text>

  <rect x="340" y="88" width="110" height="54" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.8"/>
  <text x="395" y="110" text-anchor="middle" font-size="14" font-weight="500" fill="#854F0B">BACK</text>
  <text x="395" y="128" text-anchor="middle" font-size="12" fill="#854F0B">just inside door</text>
  <text x="395" y="160" text-anchor="middle" font-size="12" fill="#BA7517">fires 10&#8211;15 s after truck clears gate</text>

  <line x1="40" y1="430" x2="640" y2="430" stroke="#D3D1C7" stroke-width="0.6"/>
  <rect x="40" y="440" width="12" height="12" rx="2" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.6"/>
  <text x="58" y="450" font-size="12" fill="#2C2C2A">Gate cameras — outside (concurrent)</text>
  <rect x="320" y="440" width="12" height="12" rx="2" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.6"/>
  <text x="338" y="450" font-size="12" fill="#2C2C2A">Back camera — inside door (deferred)</text>
</svg>

---

## 2. Timing model

- A truck enters the gate cameras at **~t = 0 s**, leaves at **~t = 120 s** (2 min), with up to **±3 s variance** between the three cameras.
- The same truck then appears in **BACK starting ~t = 120 s** and stays for a **maximum of ~10 s**.
- So the back-attach window = `gate_event.last_seen` → `last_seen + 10 s`.
- Each camera is its own video with its own timeline, aligned to the same real-world clock (or aligned by a per-camera offset in config).

<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#888780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <rect x="100" y="40" width="200" height="24" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.6"/>
  <text x="200" y="56" text-anchor="middle" font-size="12" fill="#0C447C">FRONT / RIGHT / LEFT detect truck</text>

  <rect x="300" y="40" width="200" height="24" rx="4" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.6"/>
  <text x="400" y="56" text-anchor="middle" font-size="12" fill="#854F0B">BACK detects same truck</text>

  <rect x="298" y="38" width="5" height="28" rx="1" fill="#639922"/>
  <text x="305" y="32" font-size="12" fill="#3B6D11">attach point</text>

  <line x1="60" y1="110" x2="620" y2="110" stroke="#B4B2A9" stroke-width="0.8" marker-end="url(#ar2)"/>
  <text x="625" y="114" font-size="12" fill="#888780">t</text>
  <line x1="100" y1="105" x2="100" y2="115" stroke="#888780" stroke-width="0.8"/>
  <text x="100" y="130" text-anchor="middle" font-size="12" fill="#888780">0 s</text>
  <line x1="300" y1="105" x2="300" y2="115" stroke="#888780" stroke-width="0.8"/>
  <text x="300" y="130" text-anchor="middle" font-size="12" fill="#888780">~120 s</text>
  <line x1="500" y1="105" x2="500" y2="115" stroke="#888780" stroke-width="0.8"/>
  <text x="500" y="130" text-anchor="middle" font-size="12" fill="#888780">~130 s</text>

  <line x1="96" y1="76" x2="304" y2="76" stroke="#B4B2A9" stroke-width="0.6"/>
  <text x="200" y="92" text-anchor="middle" font-size="12" fill="#5F5E5A">enters ~0 s  |  leaves ~120 s  |  &#177;3 s between cameras</text>
  <line x1="296" y1="150" x2="504" y2="150" stroke="#BA7517" stroke-width="0.6"/>
  <text x="400" y="166" text-anchor="middle" font-size="12" fill="#BA7517">enters ~120 s  |  stays max 10 s  &#8594;  leaves ~130 s</text>
</svg>

---

## 3. The core problem — cross-lane contamination

RIGHT and LEFT both cover **both lanes**. When two trucks sit in the two lanes at the
**same time**, RIGHT and LEFT each detect **both** trucks at **overlapping timestamps**.

If records are grouped on time alone, the OCR fields of Truck A (lane 1) and Truck B
(lane 2) get mixed into one consolidated record. **That is the bug.** Timestamps cannot
separate them — only `lane_id` / horizontal position can.

Secondary rule: **BACK must not contribute a main class.** The
`truck_with_container` / `truck_without_container` label comes only from the gate cameras,
even if BACK happens to emit one.

<svg width="100%" viewBox="0 0 680 240" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <rect x="330" y="20" width="10" height="180" rx="2" fill="#888780" opacity="0.6"/>
  <text x="335" y="16" text-anchor="middle" font-size="12" fill="#5F5E5A">gate door</text>

  <rect x="40" y="20" width="288" height="84" fill="#F1EFE8"/>
  <text x="56" y="36" font-size="12" fill="#888780">Lane 1</text>
  <rect x="120" y="44" width="150" height="34" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.8"/>
  <text x="195" y="65" text-anchor="middle" font-size="12" fill="#0C447C">Truck A &#183; plate MU-A</text>

  <rect x="40" y="112" width="288" height="84" fill="#F1EFE8"/>
  <text x="56" y="128" font-size="12" fill="#888780">Lane 2</text>
  <rect x="120" y="136" width="150" height="34" rx="4" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.8"/>
  <text x="195" y="157" text-anchor="middle" font-size="12" fill="#A32D2D">Truck B &#183; plate MU-B</text>

  <rect x="42" y="78" width="60" height="32" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.7"/>
  <text x="72" y="98" text-anchor="middle" font-size="13" font-weight="500" fill="#0F6E56">LEFT</text>
  <rect x="42" y="120" width="60" height="32" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.7"/>
  <text x="72" y="140" text-anchor="middle" font-size="13" font-weight="500" fill="#0F6E56">RIGHT</text>
  <rect x="250" y="98" width="64" height="32" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.7"/>
  <text x="282" y="118" text-anchor="middle" font-size="13" font-weight="500" fill="#0F6E56">FRONT</text>

  <rect x="430" y="22" width="232" height="170" rx="10" fill="none" stroke="#D3D1C7" stroke-width="0.8"/>
  <text x="546" y="44" text-anchor="middle" font-size="14" font-weight="500" fill="#A32D2D">&#9888; Cross-lane problem</text>
  <text x="446" y="68" font-size="12" fill="#2C2C2A">RIGHT covers both lanes.</text>
  <text x="446" y="88" font-size="12" fill="#2C2C2A">LEFT covers both lanes.</text>
  <text x="446" y="108" font-size="12" fill="#2C2C2A">At the same moment they see</text>
  <text x="446" y="128" font-size="12" fill="#2C2C2A">Truck A AND Truck B.</text>
  <text x="446" y="152" font-size="12" fill="#5F5E5A">Time alone cannot say which</text>
  <text x="446" y="170" font-size="12" fill="#5F5E5A">plate belongs to which truck.</text>
</svg>

---

## 4. How I want it resolved

1. **Gate grouping (lane-aware).** Group FRONT/RIGHT/LEFT records into one gate event by
   **`lane_id` + concurrent time window**. **Never merge records with different `lane_id`s**,
   even if their timestamps overlap. Use the ByteTrack `track_id` only to dedupe a single
   camera's re-detections of the same truck. A time gap larger than the sync tolerance
   starts a new event (= the next truck).

2. **Back attachment (deferred).** Attach a BACK record to the **most recent** gate event
   whose `last_seen` is within `BACK_ATTACH_WINDOW_SEC` (default 10 s) before the back
   record's `first_seen`. If nothing qualifies, log it as an orphan — never spawn a
   phantom truck.

3. **Field provenance (role-enforced).**
   - `container_no`, `company_logo`, `side_no`, `license_plate` → **gate cameras only**
   - `truck_no`, `truck_company` → **back only**
   - `main_class` → **gate only** (ignore BACK's, even if present)
   - When several gate cameras read the same field, keep the **highest-confidence**
     reading; break ties by **most-frequent** value. Out-of-role fields are ignored.

4. **No hardcoding.** `BACK_ATTACH_WINDOW_SEC`, gate sync tolerance, per-camera time
   offsets, the camera→role map, and the field→role map are all config. Another gate must
   work by changing config only.

5. **Output.** One row **per physical truck per lane**: `consolidated_id`, `lane_id`,
   `camera_source` (e.g. `FRONT+RIGHT+LEFT+BACK`), and every merged field with its value,
   confidence, and **source camera**. Plus a validation report flagging
   (a) the same container/identity under two IDs, and
   (b) one ID carrying two conflicting identities.

---

## 5. Acceptance criteria

- Two trucks in two lanes at the same time → **two clean rows, no field bleed**
  (lane 1 keeps its plate, lane 2 keeps its plate).
- BACK attaches to the correct gate event and never creates an extra truck.
- `main_class` is never sourced from BACK.
- The validation report catches duplicate IDs and conflicting identities.
- Re-targeting a different gate requires **config changes only** — no code edits, and no
  changes to `main.py`.
