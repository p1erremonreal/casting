# Casting

Hold your hands up to your webcam and conjure a glowing 4D tesseract, like casting a spell.

- **One hand:** the tesseract floats just off your palm and spins on its own. Open your hand to grow it and close it to shrink it.
- **Two hands:** the tesseract hovers between your palms and stops spinning. Turn your hands to rotate it, including through the fourth dimension.

## How it works

| Part | Language | Job |
|---|---|---|
| `tracker/` | Python | Reads the webcam with OpenCV (MJPEG, in a separate capture process) and detects hands with MediaPipe at the camera's frame rate. |
| `engine/` | Rust | Smooths and predicts hand positions at 60 fps, works out palm orientation, then rotates and projects the 4D tesseract. |
| `app/` | Electron | Starts both processes, passes data between them, and draws the camera feed and tesseract in its own window. |

## Requirements

- [Node.js](https://nodejs.org) 18 or newer
- [Python](https://www.python.org) 3.12 or 3.13
- [Rust](https://rustup.rs) (stable)
- A webcam

## Setup

Run these from the project folder.

1. Create the Python environment and install the tracker's packages:

   ```bash
   python -m venv tracker/.venv
   ```

   Windows:

   ```bash
   tracker\.venv\Scripts\python -m pip install -r tracker/requirements.txt
   ```

   macOS / Linux:

   ```bash
   tracker/.venv/bin/python -m pip install -r tracker/requirements.txt
   ```

2. Install Electron:

   ```bash
   npm install
   ```

The MediaPipe hand model (about 7.5 MB) downloads automatically on the first run.

## Run

```bash
npm start
```

This builds the Rust engine if needed, opens the window and starts the camera.

**Controls:** `C` toggles the camera feed, `L` toggles the fingertip markers, and `F11` toggles fullscreen.

**If the camera won't start**, close any other app that's using it (Discord, Teams, a browser tab) and run `npm start` again. To pick a specific camera, pass its index:

```bash
npm start -- --camera 1
```
