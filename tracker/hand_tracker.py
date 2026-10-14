"""Realtime hand tracker (two processes).

    capture process                         main process (this one)
    ---------------                         -----------------------
    camera (MJPEG) -> decode -> flip        wait for newest slot
      -> RGB + preview JPEG                 copy it out (seqlock-checked)
      -> write into shared-memory ring  ->  MediaPipe hand landmarker
      -> announce (slot, seq) on queue      -> JSON line on stdout

The capture process never waits on detection: it keeps ingesting frames into
a ring of SLOTS buffers, and the main process always jumps to the newest one.
Camera I/O, MJPEG decode, color conversion and preview encoding all happen
off the detection process.

Each stdout line:

    {"t": seconds, "aspect": w/h, "w": w, "h": h,
     "hands": [{"tips": [[x, y, z] * 5], "wrist": [x, y, z], "index_mcp": [x, y, z],
                "middle_mcp": [x, y, z], "pinky_mcp": [x, y, z],
                "handedness": "Left" | "Right", "score": float}],
     "stats": {"cap_fps", "proc_fps", "dropped", "lag_ms"},
     "jpg": "<base64 jpeg of the mirrored frame>"}

Coordinates are normalized 0..1 in the mirrored (selfie) image, origin top-left.
Logs and errors go to stderr so stdout stays a clean data stream.
"""

import argparse
import base64
import json
import multiprocessing as mp
import queue
import sys
import time
import urllib.request
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# MediaPipe hand landmark indices.
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky

SLOTS = 6  # ring buffer depth
MAX_JPG = 2 << 20  # bytes reserved per slot for the preview JPEG

# Per-slot metadata columns (float64).
META_SEQ, META_T, META_JPG_LEN = 0, 1, 2
WRITING = -1.0


def log(*args):
    print("[tracker]", *args, file=sys.stderr, flush=True)


def ensure_model():
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading hand model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ---- capture process --------------------------------------------------------


def open_camera(index, width, height, fps):
    """Open a real camera in MJPEG mode.

    Windows uses DirectShow only: MSMF (the default backend) can hang
    indefinitely on open/read, e.g. when Discord or Teams holds the camera.
    MJPEG must be requested as an open parameter; setting it afterwards is
    ignored by DirectShow and the camera stays in slow uncompressed YUY2.
    Virtual cameras (OBS, VEGAS...) are skipped: they repeat a static image.
    """
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    params = [
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
        cv2.CAP_PROP_FRAME_WIDTH, width,
        cv2.CAP_PROP_FRAME_HEIGHT, height,
        cv2.CAP_PROP_FPS, fps,
    ]
    for i in [index] if index >= 0 else [0, 1, 2, 3]:
        cap = cv2.VideoCapture(i, backend, params)
        if not cap.isOpened():
            cap.release()
            continue
        frames = [f for ok, f in (cap.read() for _ in range(6)) if ok]
        live = len(frames) >= 2 and any(
            cv2.absdiff(a, b).any() for a, b in zip(frames, frames[1:])
        )
        if live:
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join(chr((code >> 8 * k) & 0xFF) for k in range(4))
            return i, fourcc, cap
        log(f"skipping camera index {i} (no live frames - virtual camera?)")
        cap.release()
    return None, None, None


def capture_main(args, ready_q, frame_q, stop):
    cam, fourcc, cap = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        ready_q.put(("error", "could not open a camera"))
        return

    ok, frame = cap.read()
    cam_h, cam_w = frame.shape[:2]
    # Downscale wide captures for detection + preview (keeps the wide field of view).
    if args.process_width and cam_w > args.process_width:
        w = args.process_width
        h = round(cam_h * w / cam_w)
    else:
        w, h = cam_w, cam_h

    shm_frames = shared_memory.SharedMemory(create=True, size=SLOTS * h * w * 3)
    shm_jpg = shared_memory.SharedMemory(create=True, size=SLOTS * MAX_JPG)
    shm_meta = shared_memory.SharedMemory(create=True, size=SLOTS * 3 * 8)
    frames = np.ndarray((SLOTS, h, w, 3), np.uint8, shm_frames.buf)
    jpgs = np.ndarray((SLOTS, MAX_JPG), np.uint8, shm_jpg.buf)
    meta = np.ndarray((SLOTS, 3), np.float64, shm_meta.buf)
    meta[:] = 0

    ready_q.put(("ok", {
        "camera": cam, "fourcc": fourcc, "w": w, "h": h, "cam_w": cam_w, "cam_h": cam_h,
        "frames": shm_frames.name, "jpg": shm_jpg.name, "meta": shm_meta.name,
    }))

    parent = mp.parent_process()
    seq = 0
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                log("camera read failed")
                break
            t = time.monotonic()
            seq += 1
            slot = seq % SLOTS

            # No image processing beyond a lossless mirror: what Logi Options+ sets on
            # the camera (exposure, white balance...) is exactly what you see.
            frame = cv2.flip(frame, 1)  # mirror: your right hand moves right on screen
            jpg = None
            if not args.no_frames:
                # Full-resolution preview; re-encoded once because OpenCV's DirectShow
                # backend always decodes MJPEG and can't hand us the camera's own bytes.
                ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if ok and enc.size <= MAX_JPG:
                    jpg = enc.reshape(-1)
            # Only the detection copy is downscaled.
            small = frame if frame.shape[1] == w else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            # Seqlock write: mark slot as in-progress, fill it, then publish seq.
            meta[slot, META_SEQ] = WRITING
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB, dst=frames[slot])
            if jpg is not None:
                jpgs[slot, : jpg.size] = jpg
            meta[slot, META_T] = t
            meta[slot, META_JPG_LEN] = 0 if jpg is None else jpg.size
            meta[slot, META_SEQ] = seq

            try:
                frame_q.put_nowait((slot, seq))
            except queue.Full:
                pass  # consumer only wants the newest anyway

            # If the parent was killed (Electron closed), release the camera.
            if seq % 30 == 0 and parent is not None and not parent.is_alive():
                break
    finally:
        cap.release()
        del frames, jpgs, meta
        for shm in (shm_frames, shm_jpg, shm_meta):
            shm.close()
            shm.unlink()


# ---- main (detection) process -----------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=-1, help="camera index; -1 = first live camera")
    # 16:9 capture uses the camera's full horizontal field of view (4:3 crops the sides).
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--process-width", type=int, default=640, help="downscale to this width; 0 = full size")
    parser.add_argument("--fps", type=int, default=60, help="requested; the camera may cap it (Brio 100: 30)")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--no-frames", action="store_true", help="omit the camera image")
    args = parser.parse_args()

    ensure_model()

    ready_q = mp.Queue()
    frame_q = mp.Queue(maxsize=SLOTS * 4)
    stop = mp.Event()
    capture = mp.Process(target=capture_main, args=(args, ready_q, frame_q, stop), name="capture", daemon=True)
    capture.start()

    # Load the model while the capture process opens the camera.
    import mediapipe as mediapipe
    from mediapipe.tasks.python import BaseOptions, vision

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    try:
        kind, info = ready_q.get(timeout=30)
    except queue.Empty:
        kind, info = "error", "camera open timed out"
    if kind != "ok":
        log(info)
        stop.set()
        sys.exit(2)

    w, h = info["w"], info["h"]
    log(f"capture process: camera {info['camera']} {info['cam_w']}x{info['cam_h']} {info['fourcc']} -> {w}x{h}")
    shm_frames = shared_memory.SharedMemory(name=info["frames"])
    shm_jpg = shared_memory.SharedMemory(name=info["jpg"])
    shm_meta = shared_memory.SharedMemory(name=info["meta"])
    frames = np.ndarray((SLOTS, h, w, 3), np.uint8, shm_frames.buf)
    jpgs = np.ndarray((SLOTS, MAX_JPG), np.uint8, shm_jpg.buf)
    meta = np.ndarray((SLOTS, 3), np.float64, shm_meta.buf)

    rgb = np.empty((h, w, 3), np.uint8)
    start = time.monotonic()
    last_ts_ms = -1
    last_seq = 0
    dropped = 0
    cap_times, proc_times = [], []
    out = sys.stdout

    try:
        while True:
            # Block for the next announcement, then drain to the newest one.
            try:
                slot, seq = frame_q.get(timeout=5)
            except queue.Empty:
                if not capture.is_alive():
                    log("capture process stopped")
                    break
                continue
            while True:
                try:
                    slot, seq = frame_q.get_nowait()
                except queue.Empty:
                    break
            if seq <= last_seq:
                continue

            # Seqlock read: copy out, then confirm the slot wasn't overwritten meanwhile.
            if meta[slot, META_SEQ] != seq:
                dropped += 1
                continue
            np.copyto(rgb, frames[slot])
            t_cap = float(meta[slot, META_T])
            jpg_len = int(meta[slot, META_JPG_LEN])
            jpg = bytes(jpgs[slot, :jpg_len]) if jpg_len else None
            if meta[slot, META_SEQ] != seq:
                dropped += 1
                continue

            dropped += max(0, seq - last_seq - 1)
            last_seq = seq

            t = t_cap - start
            ts_ms = max(int(t * 1000), last_ts_ms + 1)  # must be strictly increasing
            last_ts_ms = ts_ms
            result = landmarker.detect_for_video(
                mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb), ts_ms
            )

            hands = []
            for i, lms in enumerate(result.hand_landmarks):
                # z is MediaPipe's relative depth (same scale as x; smaller = closer).
                pt = lambda j: [round(lms[j].x, 5), round(lms[j].y, 5), round(lms[j].z, 5)]
                cat = result.handedness[i][0] if i < len(result.handedness) else None
                hands.append(
                    {
                        "tips": [pt(j) for j in FINGERTIPS],
                        "wrist": pt(WRIST),
                        "index_mcp": pt(INDEX_MCP),
                        "middle_mcp": pt(MIDDLE_MCP),
                        "pinky_mcp": pt(PINKY_MCP),
                        "handedness": cat.category_name if cat else "",
                        "score": round(cat.score, 3) if cat else 0.0,
                    }
                )

            now = time.monotonic()
            cap_times = [x for x in cap_times if now - x < 1.0] + [t_cap]
            proc_times = [x for x in proc_times if now - x < 1.0] + [now]
            msg = {
                "t": round(t, 4),
                "aspect": w / h,
                "w": w,
                "h": h,
                "hands": hands,
                "stats": {
                    "cap_fps": len(cap_times),
                    "proc_fps": len(proc_times),
                    "dropped": dropped,
                    "lag_ms": round((now - t_cap) * 1000, 1),
                },
            }
            if jpg:
                msg["jpg"] = base64.b64encode(jpg).decode("ascii")

            out.write(json.dumps(msg, separators=(",", ":")))
            out.write("\n")
            out.flush()
    except (BrokenPipeError, KeyboardInterrupt, OSError):
        pass
    finally:
        stop.set()
        capture.join(timeout=2)
        if capture.is_alive():
            capture.terminate()
        landmarker.close()
        del frames, jpgs, meta
        for shm in (shm_frames, shm_jpg, shm_meta):
            shm.close()


if __name__ == "__main__":
    main()
