/* =============================================================================
 * Symphony of the Stare — MVP
 *
 * Pipeline:
 *   webcam → MediaPipe Face Mesh (478 landmarks incl. iris)
 *          → raw gaze metrics (head yaw, head pitch, iris X/Y, eye openness)
 *          → minus a calibrated "neutral" baseline
 *          → per-metric score 0..1 → combined targetFocus
 *          → asymmetric smoothing → focusLevel (0..1)
 *          → Tone.js parameter mapping (filter, bitcrush, distortion, detune…)
 *
 * Page Visibility: when the tab is hidden, the music is muted immediately and
 * replaced by pure white noise.
 * ========================================================================== */

/* -----------------------------------------------------------------------------
 * 1. TUNING — everything you will want to tweak lives here.
 *
 * How "looking away" is measured
 * ------------------------------
 * Every metric is a dimensionless RATIO computed from landmark positions, so it
 * does not depend on how far you sit from the camera:
 *
 *   headYaw    = where the nose tip sits between the two cheek edges.
 *                (nose.x - leftCheek.x) / (rightCheek.x - leftCheek.x) - 0.5
 *                0 = facing camera, about ±0.5 = full profile.
 *
 *   headPitch  = where the nose tip sits between forehead and chin.
 *                (nose.y - forehead.y) / (chin.y - forehead.y)
 *                Grows when you look down, shrinks when you look up.
 *
 *   irisX      = where the iris centre sits between the two eye corners,
 *                averaged over both eyes. 0.5 = centred, about 0.3/0.7 = hard sideways.
 *
 *   irisY      = where the iris centre sits between upper and lower eyelid.
 *                Noisy (eyelids follow the eyes), so it gets a lower weight.
 *
 *   eyeOpen    = eyelid gap / eye width ("eye aspect ratio"). Closed eyes = not looking.
 *
 * Calibration: for CALIBRATION_MS after start you stare at the dot and we
 * record the average of each metric. That average becomes "zero". This matters
 * because the webcam is usually above the screen, so looking at the screen
 * centre is NOT the same as looking into the lens.
 *
 * Each metric's deviation |value - baseline| is converted to a score with a
 * dead zone and a falloff:
 *
 *        score
 *        1.0 ┤━━━━━━━╮
 *            │        ╲             dev <= deadZone          → 1 (fully focused)
 *            │         ╲            deadZone < dev < limit   → smooth falloff
 *        0.0 ┤          ╰━━━━━━━    dev >= limit             → 0 (looking away)
 *            └───────┬───┬────── |deviation|
 *                deadZone limit
 *
 * The scores are MULTIPLIED (each raised to its weight), so failing any single
 * test is enough to break focus.
 *
 * To tune: press "D" to see live values. Look at the dot, then at the screen
 * edge you consider "away", and put deadZone/limit between those numbers.
 * Smaller numbers = stricter.
 * -------------------------------------------------------------------------- */
const FOCUS = {
  headYaw:   { deadZone: 0.04, limit: 0.14, weight: 1.0 },  // turning head left/right
  headPitch: { deadZone: 0.04, limit: 0.12, weight: 1.0 },  // nodding up/down
  irisX:     { deadZone: 0.04, limit: 0.11, weight: 1.0 },  // eyes left/right (most telling)
  irisY:     { deadZone: 0.07, limit: 0.20, weight: 0.6 },  // eyes up/down (noisy, softer)

  // Eye aspect ratio below CLOSED = eyes shut (score 0); above OPEN = fine (score 1).
  // Normal open eyes sit around 0.25–0.32. Measured in pixels, so aspect-correct.
  eyeClosed: 0.11,
  eyeOpen:   0.17,
};

// Smoothing time constants, in seconds. Asymmetric on purpose:
// distraction is punished almost instantly, focus has to be "earned back".
const SMOOTHING = {
  dropTau: 0.06,   // how fast focusLevel falls when you look away (lower = snappier)
  riseTau: 0.45,   // how fast it recovers when you look back
};

const CALIBRATION_MS = 2500;          // time spent learning your neutral gaze
const NO_FACE_FOCUS = 0;              // focus target when no face is detected
const DISTRACTED_BELOW = 0.5;         // UI "distracted" state threshold
const TREAT_WINDOW_BLUR_AS_HIDDEN = false; // true = clicking another app also triggers white noise

// Audio levels (dB / linear). Keep the noise sane; it is meant to be unpleasant, not dangerous.
const AUDIO = {
  musicDb: -8,          // overall music level
  hiddenNoiseGain: 0.18, // white noise level when the tab is hidden (linear 0..1)
  maxGrimeNoiseGain: 0.12, // extra noise blended in at focusLevel 0 (linear)
  filterMaxHz: 16000,   // low-pass cutoff when focused
  filterMinHz: 220,     // low-pass cutoff when distracted
  maxDetuneCents: -65,  // pad detune at focus 0 (≈ a sour quarter-tone flat vs. the drone)
  maxVibratoDepth: 0.9, // pitch wobble at focus 0
  reverbFocused: 0.55,  // reverb wet when focused
  reverbDistracted: 0.08,
  rampSec: 0.04,        // per-frame parameter ramp; small = responsive, avoids zipper clicks
};

/* -----------------------------------------------------------------------------
 * 2. FACE MESH LANDMARK INDICES (MediaPipe 468 + 10 iris points)
 * "Left/right" below means IMAGE left/right, not the subject's.
 * -------------------------------------------------------------------------- */
const LM = {
  noseTip: 1,
  forehead: 10,
  chin: 152,
  cheekA: 234,          // image-left face edge
  cheekB: 454,          // image-right face edge
  // Eye 1 (subject's right eye, image-left)
  e1Outer: 33, e1Inner: 133, e1Top: 159, e1Bottom: 145, e1Iris: 468,
  // Eye 2 (subject's left eye, image-right)
  e2Inner: 362, e2Outer: 263, e2Top: 386, e2Bottom: 374, e2Iris: 473,
};

/* -----------------------------------------------------------------------------
 * 3. STATE
 * -------------------------------------------------------------------------- */
let focusLevel = 0;        // the smoothed, public value (0..1)
let targetFocus = 0;       // the raw per-detection value
let lastMetrics = null;    // latest raw metrics, for the debug panel
let lastScores = null;
let baseline = null;       // calibrated neutral metrics
let calibrationSamples = [];
let calibratingUntil = 0;
let faceVisible = false;
let audio = null;          // filled by buildAudio()
let lastFrameTime = performance.now();
let lastDetuneSent = 0;

const $ = (id) => document.getElementById(id);
const video = $("cam");

/* -----------------------------------------------------------------------------
 * 4. MATH HELPERS
 * -------------------------------------------------------------------------- */
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (e0, e1, x) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};

/** Deviation → score using the dead-zone/limit curve described above. */
function deviationScore(dev, cfg) {
  return 1 - smoothstep(cfg.deadZone, cfg.limit, Math.abs(dev));
}

/* -----------------------------------------------------------------------------
 * 5. VISION: landmarks → metrics → focus target
 * -------------------------------------------------------------------------- */
function computeMetrics(lms, w, h) {
  // Work in pixel space so x and y ratios are aspect-correct.
  const p = (i) => ({ x: lms[i].x * w, y: lms[i].y * h });

  const nose = p(LM.noseTip);
  const cheekA = p(LM.cheekA), cheekB = p(LM.cheekB);
  const top = p(LM.forehead), chin = p(LM.chin);

  const headYaw = (nose.x - cheekA.x) / (cheekB.x - cheekA.x) - 0.5;
  const headPitch = (nose.y - top.y) / (chin.y - top.y);

  // Iris position inside each eye, measured from the image-left corner,
  // so both eyes move in the same direction when you glance sideways.
  function eye(cornerL, cornerR, lidTop, lidBottom, iris) {
    const a = p(cornerL), b = p(cornerR), t = p(lidTop), bt = p(lidBottom), c = p(iris);
    const width = Math.hypot(b.x - a.x, b.y - a.y);
    const gap = Math.hypot(bt.x - t.x, bt.y - t.y);
    return {
      x: (c.x - a.x) / (b.x - a.x),
      y: gap > 1e-3 ? (c.y - t.y) / (bt.y - t.y) : 0.5,
      ear: gap / width,
    };
  }
  const e1 = eye(LM.e1Outer, LM.e1Inner, LM.e1Top, LM.e1Bottom, LM.e1Iris);
  const e2 = eye(LM.e2Inner, LM.e2Outer, LM.e2Top, LM.e2Bottom, LM.e2Iris);

  return {
    headYaw,
    headPitch,
    irisX: (e1.x + e2.x) / 2,
    irisY: (e1.y + e2.y) / 2,
    eyeOpen: (e1.ear + e2.ear) / 2,
  };
}

function scoreMetrics(m, base) {
  const s = {
    headYaw:   deviationScore(m.headYaw - base.headYaw, FOCUS.headYaw),
    headPitch: deviationScore(m.headPitch - base.headPitch, FOCUS.headPitch),
    irisX:     deviationScore(m.irisX - base.irisX, FOCUS.irisX),
    irisY:     deviationScore(m.irisY - base.irisY, FOCUS.irisY),
    eyeOpen:   smoothstep(FOCUS.eyeClosed, FOCUS.eyeOpen, m.eyeOpen),
  };
  s.total =
    Math.pow(s.headYaw, FOCUS.headYaw.weight) *
    Math.pow(s.headPitch, FOCUS.headPitch.weight) *
    Math.pow(s.irisX, FOCUS.irisX.weight) *
    Math.pow(s.irisY, FOCUS.irisY.weight) *
    s.eyeOpen;
  return s;
}

function onFaceResults(results) {
  const faces = results.multiFaceLandmarks;
  faceVisible = !!(faces && faces.length && faces[0].length >= 478);
  if (!faceVisible) {
    targetFocus = NO_FACE_FOCUS;
    lastMetrics = lastScores = null;
    return;
  }

  const m = computeMetrics(faces[0], video.videoWidth || 640, video.videoHeight || 480);
  lastMetrics = m;

  // Collect calibration samples while the prompt is visible.
  if (performance.now() < calibratingUntil) {
    if (m.eyeOpen > FOCUS.eyeOpen) calibrationSamples.push(m); // skip blinks
    targetFocus = 1; // let the music bloom while calibrating
    return;
  }
  if (!baseline) finishCalibration();

  lastScores = scoreMetrics(m, baseline);
  targetFocus = lastScores.total;
}

function startCalibration() {
  baseline = null;
  calibrationSamples = [];
  calibratingUntil = performance.now() + CALIBRATION_MS;
  $("calibrate").hidden = false;
  setTimeout(() => { $("calibrate").hidden = true; }, CALIBRATION_MS);
}

function finishCalibration() {
  const keys = ["headYaw", "headPitch", "irisX", "irisY"];
  baseline = {};
  for (const k of keys) {
    const vals = calibrationSamples.map((s) => s[k]).sort((a, b) => a - b);
    // Median is robust against a stray glance during calibration.
    baseline[k] = vals.length
      ? vals[Math.floor(vals.length / 2)]
      : { headYaw: 0, headPitch: 0.55, irisX: 0.5, irisY: 0.5 }[k]; // fallback defaults
  }
}

/* -----------------------------------------------------------------------------
 * 6. CAMERA + FACE MESH LOOP
 * -------------------------------------------------------------------------- */
async function startVision() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  const faceMesh = new FaceMesh({
    locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${f}`,
  });
  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,   // REQUIRED: adds iris landmarks 468–477
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  faceMesh.onResults(onFaceResults);
  await faceMesh.initialize();

  // Send frames one at a time; never queue a new frame while one is in flight.
  let busy = false;
  const pump = async () => {
    if (!busy && video.readyState >= 2 && !document.hidden) {
      busy = true;
      try { await faceMesh.send({ image: video }); }
      catch (e) { console.warn("FaceMesh frame failed", e); }
      busy = false;
    }
    // requestVideoFrameCallback fires once per new camera frame where supported.
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(pump);
    else requestAnimationFrame(pump);
  };
  pump();
}

/* -----------------------------------------------------------------------------
 * 7. AUDIO: generative ambient + degradation chain
 *
 *   pad ─┐
 *   drone┼→ musicBus → vibrato → lowpass → distortion → bitcrusher → reverb → master ┐
 *   bell ┘   (gain)    (detune   (cutoff)   (wet)        (wet)        (wet)            │
 *                       wobble)                                                       ├→ limiter → out
 *   grimeNoise → grimeGain ───────────────────────────────────────────────────────────┤
 *   tabNoise (white) → tabGain  (Page Visibility) ────────────────────────────────────┘
 * -------------------------------------------------------------------------- */
async function buildAudio() {
  const limiter = new Tone.Limiter(-2).toDestination();
  const master = new Tone.Volume(AUDIO.musicDb).connect(limiter);

  const reverb = new Tone.Reverb({ decay: 9, preDelay: 0.03, wet: AUDIO.reverbFocused });
  await reverb.generate();
  const crusher = new Tone.BitCrusher(4);           // 4 bits = crunchy
  crusher.wet.value = 0;
  const distortion = new Tone.Distortion({ distortion: 0.95, oversample: "2x", wet: 0 });
  const filter = new Tone.Filter({ type: "lowpass", frequency: AUDIO.filterMaxHz, Q: 0.7, rolloff: -24 });
  const vibrato = new Tone.Vibrato({ frequency: 5.5, depth: 0, wet: 1 });
  const musicBus = new Tone.Gain(1);

  musicBus.chain(vibrato, filter, distortion, crusher, reverb, master);

  // --- Pad: slow, wide chords -------------------------------------------------
  const pad = new Tone.PolySynth(Tone.Synth, {
    maxPolyphony: 16,
    oscillator: { type: "fattriangle", count: 3, spread: 24 },
    envelope: { attack: 2.5, decay: 1.5, sustain: 0.8, release: 6 },
    volume: -14,
  }).connect(musicBus);

  // --- Drone: steady root. It never detunes, so a detuned pad clashes with it.
  const drone = new Tone.FatOscillator({ frequency: "D2", type: "sine", count: 2, spread: 8, volume: -16 })
    .connect(musicBus);

  // --- Bells: sparse random pentatonic notes through a soft delay --------------
  const delay = new Tone.FeedbackDelay({ delayTime: "4n.", feedback: 0.45, wet: 0.35 }).connect(musicBus);
  const bell = new Tone.FMSynth({
    harmonicity: 3.01,
    modulationIndex: 6,
    envelope: { attack: 0.005, decay: 1.8, sustain: 0, release: 2 },
    modulationEnvelope: { attack: 0.002, decay: 0.4, sustain: 0, release: 0.5 },
    volume: -24,
  }).connect(delay);

  // --- Noise layers -------------------------------------------------------------
  const grimeGain = new Tone.Gain(0).connect(limiter);
  const grimeFilter = new Tone.Filter({ type: "bandpass", frequency: 2500, Q: 0.8 }).connect(grimeGain);
  const grimeNoise = new Tone.Noise("pink").connect(grimeFilter);

  const tabGain = new Tone.Gain(0).connect(limiter);
  const tabNoise = new Tone.Noise("white").connect(tabGain);

  // --- Generative score -------------------------------------------------------------
  const chords = [
    ["D3", "A3", "C#4", "E4", "F#4"],   // Dmaj9
    ["B2", "F#3", "A3", "D4", "E4"],    // Bm11
    ["G2", "D3", "F#3", "A3", "B3"],    // Gmaj9
    ["A2", "E3", "G3", "B3", "D4"],     // A9sus
  ];
  const pentatonic = ["D5", "E5", "F#5", "A5", "B5", "D6", "E6"];

  Tone.Transport.bpm.value = 60;
  let chordIndex = 0;
  new Tone.Loop((time) => {
    pad.triggerAttackRelease(chords[chordIndex % chords.length], "1m", time);
    chordIndex++;
  }, "2m").start(0);

  new Tone.Loop((time) => {
    if (Math.random() < 0.35) {
      const note = pentatonic[Math.floor(Math.random() * pentatonic.length)];
      bell.triggerAttackRelease(note, "8n", time + Math.random() * 0.1, 0.4 + Math.random() * 0.4);
    }
  }, "4n").start("1m");

  drone.start();
  grimeNoise.start();
  tabNoise.start();
  Tone.Transport.start("+0.1");

  return { musicBus, vibrato, filter, distortion, crusher, reverb, pad, grimeGain, tabGain };
}

/**
 * The heart of the piece: focusLevel → sound.
 * d ("distraction") = 1 - focusLevel. Curves are chosen so even a small
 * glance away is clearly audible, and full distraction is truly ugly.
 */
function applyFocusToAudio(f) {
  if (!audio) return;
  const d = 1 - f;
  const r = AUDIO.rampSec;

  // Low-pass cutoff: exponential sweep so it sounds even across the range.
  const cutoff = AUDIO.filterMinHz * Math.pow(AUDIO.filterMaxHz / AUDIO.filterMinHz, f);
  audio.filter.frequency.rampTo(cutoff, r);

  // Distortion kicks in early (sqrt curve → noticeable even at d = 0.1).
  audio.distortion.wet.rampTo(Math.min(1, Math.sqrt(d) * 1.1), r);

  // Bitcrusher arrives a little later, then dominates.
  audio.crusher.wet.rampTo(smoothstep(0.15, 0.7, d), r);

  // Pitch: seasick vibrato and a sour detune against the drone.
  audio.vibrato.depth.rampTo(AUDIO.maxVibratoDepth * d * d, r);
  const detune = AUDIO.maxDetuneCents * d;
  if (Math.abs(detune - lastDetuneSent) > 1.5) {   // avoid hammering every voice every frame
    audio.pad.set({ detune });
    lastDetuneSent = detune;
  }

  // Reverb: lush when focused, dry and exposed when not.
  audio.reverb.wet.rampTo(lerp(AUDIO.reverbDistracted, AUDIO.reverbFocused, f), r);

  // Extra noise bed creeping in underneath.
  audio.grimeGain.gain.rampTo(AUDIO.maxGrimeNoiseGain * d * d, r);
}

/* -----------------------------------------------------------------------------
 * 8. PAGE VISIBILITY: tab hidden → music off, pure white noise on.
 * Hidden tabs throttle requestAnimationFrame, so this is done directly in the
 * event handler with immediate (not ramped) value changes.
 * -------------------------------------------------------------------------- */
function setHidden(hidden) {
  if (!audio) return;
  const now = Tone.now();
  audio.musicBus.gain.cancelScheduledValues(now);
  audio.tabGain.gain.cancelScheduledValues(now);
  audio.musicBus.gain.setValueAtTime(hidden ? 0 : 1, now);
  audio.tabGain.gain.setValueAtTime(hidden ? AUDIO.hiddenNoiseGain : 0, now);
  if (hidden) {
    // Coming back should start from zero focus, so the music must be re-earned.
    focusLevel = 0;
    targetFocus = 0;
  }
}

document.addEventListener("visibilitychange", () => setHidden(document.hidden));
if (TREAT_WINDOW_BLUR_AS_HIDDEN) {
  window.addEventListener("blur", () => setHidden(true));
  window.addEventListener("focus", () => setHidden(document.hidden));
}

/* -----------------------------------------------------------------------------
 * 9. MAIN LOOP: smoothing + visuals + audio mapping (runs every display frame)
 * -------------------------------------------------------------------------- */
function frame(now) {
  const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
  lastFrameTime = now;

  // Exponential smoothing with a frame-rate independent coefficient.
  const tau = targetFocus < focusLevel ? SMOOTHING.dropTau : SMOOTHING.riseTau;
  focusLevel += (targetFocus - focusLevel) * (1 - Math.exp(-dt / tau));
  focusLevel = clamp01(focusLevel);

  // Visuals
  document.documentElement.style.setProperty("--focus", focusLevel.toFixed(3));
  document.body.classList.toggle("distracted", focusLevel < DISTRACTED_BELOW);
  $("status").textContent = !faceVisible ? "where are you?"
    : focusLevel > 0.85 ? "listening"
    : focusLevel > DISTRACTED_BELOW ? "drifting"
    : "lost";

  if (!document.hidden) applyFocusToAudio(focusLevel);
  if (!$("debug").hidden) renderDebug();

  requestAnimationFrame(frame);
}

function renderDebug() {
  const f = (v) => (v === undefined || v === null ? "  –  " : v.toFixed(3).padStart(6));
  const m = lastMetrics, s = lastScores, b = baseline;
  const row = (k) =>
    `${k.padEnd(10)} raw ${f(m && m[k])}  base ${f(b && b[k])}  dev ${f(m && b ? m[k] - b[k] : null)}  score ${f(s && s[k])}`;
  $("debugText").textContent = [
    `focusLevel ${f(focusLevel)}   target ${f(targetFocus)}   face ${faceVisible ? "yes" : "no"}`,
    performance.now() < calibratingUntil ? "CALIBRATING…" : "",
    row("headYaw"), row("headPitch"), row("irisX"), row("irisY"),
    `eyeOpen    raw ${f(m && m.eyeOpen)}  closed<${FOCUS.eyeClosed} open>${FOCUS.eyeOpen}  score ${f(s && s.eyeOpen)}`,
  ].join("\n");
}

/* -----------------------------------------------------------------------------
 * 10. START
 * -------------------------------------------------------------------------- */
$("start").addEventListener("click", async () => {
  const btn = $("start");
  btn.disabled = true;
  btn.textContent = "Loading…";
  $("error").textContent = "";
  try {
    await Tone.start();                 // must happen inside the click gesture
    audio = await buildAudio();
    applyFocusToAudio(0);
    await startVision();
    $("intro").classList.add("gone");
    startCalibration();
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    btn.textContent = "Start Experience";
    $("error").textContent =
      err && err.name === "NotAllowedError"
        ? "Camera access was blocked. Allow it in the address bar and try again."
        : `Could not start: ${err && err.message ? err.message : err}`;
  }
});

window.addEventListener("keydown", (e) => {
  if (e.key === "d" || e.key === "D") {
    $("debug").hidden = !$("debug").hidden;
    document.body.classList.toggle("debug", !$("debug").hidden);
  }
  if ((e.key === "c" || e.key === "C") && audio) startCalibration();
});
