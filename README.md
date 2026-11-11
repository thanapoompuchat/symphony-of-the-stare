# Symphony of the Stare (MVP)

Files: `index.html`, `style.css`, `main.js`. Libraries load from jsDelivr (Tone.js 14.8.49, MediaPipe Face Mesh 0.4.1633559619).

## Run locally
The webcam only works on `localhost` or HTTPS, so don't open index.html as a file. From this folder:

    python3 -m http.server 8000      # or: npx serve .

Then open http://localhost:8000, click **Start Experience**, allow the camera, and stare at the dot for the 2.5 s calibration.

## Keys
- **D**: debug panel with live raw values, baseline, deviation and score for each metric, plus a camera preview.
- **C**: recalibrate (stare at the dot while it runs).

## Tuning
All knobs are at the top of `main.js`: `FOCUS` (dead zone / limit / weight per metric), `SMOOTHING` (drop vs. recovery speed), `AUDIO` (levels and mapping ranges), `TREAT_WINDOW_BLUR_AS_HIDDEN`.
Workflow: open the debug panel, look at the dot, then at whatever you consider "away", and set `deadZone` and `limit` between the two `dev` values.
