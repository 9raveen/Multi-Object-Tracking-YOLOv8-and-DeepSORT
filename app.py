import gradio as gr
import cv2
import os
import tempfile
import hashlib
import torch
import glob

from src.detector import YOLODetector
from src.tracker import Tracker
from src.team_classifier import TeamClassifier
from src.scene_detector import SceneChangeDetector

# ── CPU optimization ──────────────────────────────────────────────────────
torch.set_num_threads(4)

# ── Constants ─────────────────────────────────────────────────────────────
MAX_DETECTIONS  = 20
MAX_FRAMES      = 375   # 15 seconds at 25fps
PROCESS_EVERY_N = 2     # process every 2nd frame
RESIZE_WIDTH    = 640   # downscale for CPU speed

# ── Find demo clip robustly ───────────────────────────────────────────────
def find_demo_clip():
    """Try all known HF Spaces paths to find demo_clip.mp4"""
    candidates = [
        "demo_clip.mp4",
        "./demo_clip.mp4",
        "/home/user/app/demo_clip.mp4",
        "/app/demo_clip.mp4",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_clip.mp4"),
    ]
    # Also search recursively
    found = glob.glob("**/demo_clip.mp4", recursive=True)
    candidates += found

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

DEMO_CLIP_PATH = find_demo_clip()

# ── Helpers ───────────────────────────────────────────────────────────────
def get_fallback_color(track_id):
    h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    r = max(100, (h & 0xFF0000) >> 16)
    g = max(100, (h & 0x00FF00) >> 8)
    b = max(100, (h & 0x0000FF))
    return (b, g, r)

def resize_frame(frame, target_width=RESIZE_WIDTH):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame, 1.0
    scale   = target_width / w
    new_h   = int(h * scale)
    resized = cv2.resize(frame, (target_width, new_h))
    return resized, scale

def scale_tracks_up(tracks, scale):
    if scale == 1.0:
        return tracks
    scaled = []
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        scaled.append([
            int(x1 / scale), int(y1 / scale),
            int(x2 / scale), int(y2 / scale),
            track_id
        ])
    return scaled

# ── Draw function ─────────────────────────────────────────────────────────
def draw(frame, tracks, track_history, team_classifier,
         frame_idx, total, scene_cuts):
    annotated = frame.copy()

    for track in tracks:
        x1, y1, x2, y2, track_id = track

        if x2 <= x1 or y2 <= y1:
            continue

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w > 200 or box_h > 250 or box_w < 15:
            continue

        if track_id not in team_classifier.team_assignments:
            color = get_fallback_color(track_id)
            team  = "Detecting..."
        else:
            color = team_classifier.get_team_color(track_id)
            team  = team_classifier.team_assignments[track_id]

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"ID {track_id} | {team}"
        (lw, lh), base = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated,
                      (x1, y1 - lh - base - 4),
                      (x1 + lw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Trajectory
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)
        for i in range(1, len(track_history[track_id])):
            cv2.line(annotated,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     color, 2)

    # Overlay
    cv2.putText(annotated,
                f"Frame {frame_idx}/{total} | "
                f"Players: {len(tracks)} | Cuts: {scene_cuts}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    return annotated

# ── Main pipeline ─────────────────────────────────────────────────────────
def run_tracking(video_path, conf_threshold, max_det,
                 progress=gr.Progress()):

    if video_path is None:
        return None, "❌ Please upload a video first."

    if not os.path.exists(video_path):
        return None, f"❌ Video not found at: {video_path}"

    detector        = YOLODetector(model_path="yolov8n.pt",
                                   conf_threshold=float(conf_threshold))
    tracker         = Tracker(max_age=60, n_init=5)
    team_classifier = TeamClassifier(update_interval=10)
    scene_detector  = SceneChangeDetector(threshold=35.0)
    track_history   = {}

    detector.model.to("cpu")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)

    if total == 0:
        return None, "❌ Video has 0 readable frames. Try re-uploading."

    tmp_out  = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx   = 0
    scene_cuts  = 0
    last_tracks = []
    max_det     = int(max_det)

    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        progress(frame_idx / total,
                 desc=f"Processing frame {frame_idx}/{total}...")

        # Scene change check
        if scene_detector.is_scene_change(frame):
            tracker         = Tracker(max_age=60, n_init=5)
            track_history   = {}
            last_tracks     = []
            team_classifier.reset()
            scene_cuts += 1
            out.write(frame)
            continue

        # Skip every 2nd frame
        if frame_idx % PROCESS_EVERY_N != 0:
            annotated = draw(frame, last_tracks, track_history,
                             team_classifier, frame_idx, total, scene_cuts)
            out.write(annotated)
            continue

        # Downscale for inference
        small_frame, scale = resize_frame(frame)

        # Detect
        detections, crops, ball = detector.detect(small_frame)
        if len(detections) > max_det:
            detections = detections[:max_det]
            crops      = crops[:max_det]

        # Track
        tracks_small = tracker.update(detections, small_frame)
        tracks       = scale_tracks_up(tracks_small, scale)

        # Team classify
        team_classifier.classify_by_crops(
            tracks_small, detections, crops, frame_idx)

        last_tracks = tracks

        # Clean history
        active_ids    = set(t[4] for t in tracks)
        track_history = {tid: pts for tid, pts in track_history.items()
                         if tid in active_ids}

        annotated = draw(frame, tracks, track_history,
                         team_classifier, frame_idx, total, scene_cuts)
        out.write(annotated)

    cap.release()
    out.release()

    stats = (f"✅ Done! {frame_idx} frames processed | "
             f"Scene cuts: {scene_cuts} | "
             f"Capped at {MAX_FRAMES} frames (~15 sec).")
    return out_path, stats


# ── Demo runner ───────────────────────────────────────────────────────────
def run_demo(conf, maxdet, progress=gr.Progress()):
    if DEMO_CLIP_PATH is None:
        return None, "❌ demo_clip.mp4 not found on server. Please use Upload tab."
    return run_tracking(DEMO_CLIP_PATH, conf, maxdet, progress)


# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Sports Player Tracker") as demo:

    gr.Markdown("""
    # ⚽ Multi-Object Player Tracking
    **YOLOv8n + DeepSORT | Team Classification | Trajectory Visualization**

    Detects players, assigns persistent IDs, classifies teams by jersey color,
    and draws movement trails.

    > ⏱️ CPU processing — ~3-5 FPS. Best with clips under **15 seconds**.
    """)

    with gr.Tabs():

        # ── Tab 1: Upload ──────────────────────────────────────────────
        with gr.TabItem("📤 Upload Your Video"):
            gr.Markdown("""
            Upload any broadcast sports clip.
            Wide-angle shots with multiple players work best.
            Keep clips under **15 seconds** for reasonable processing time.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    video_upload = gr.Video(
                        label="Upload Video Clip",
                        sources=["upload"]
                    )
                    with gr.Accordion("⚙️ Settings", open=False):
                        conf_upload = gr.Slider(
                            minimum=0.3, maximum=0.8,
                            value=0.5, step=0.05,
                            label="Detection Confidence"
                        )
                        maxdet_upload = gr.Slider(
                            minimum=5, maximum=30,
                            value=20, step=1,
                            label="Max Detections per Frame"
                        )
                    run_upload_btn = gr.Button(
                        "▶ Run Tracking", variant="primary"
                    )
                with gr.Column(scale=1):
                    out_upload  = gr.Video(label="Annotated Output")
                    stat_upload = gr.Textbox(label="Status", interactive=False)

            run_upload_btn.click(
                fn=run_tracking,
                inputs=[video_upload, conf_upload, maxdet_upload],
                outputs=[out_upload, stat_upload]
            )

        # ── Tab 2: Demo ────────────────────────────────────────────────
        with gr.TabItem("🎬 Try Demo Clip"):
            gr.Markdown("""
            Pre-loaded FA Cup 2024 clip —
            Manchester City vs Manchester United, Wembley Stadium.
            Click **Run Demo** — no upload needed.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        f"**Demo clip path:** `{DEMO_CLIP_PATH or 'Not found'}`"
                    )
                    with gr.Accordion("⚙️ Settings", open=False):
                        conf_demo   = gr.Slider(
                            minimum=0.3, maximum=0.8,
                            value=0.5, step=0.05,
                            label="Detection Confidence"
                        )
                        maxdet_demo = gr.Slider(
                            minimum=5, maximum=30,
                            value=20, step=1,
                            label="Max Detections per Frame"
                        )
                    run_demo_btn = gr.Button(
                        "▶ Run Demo", variant="primary"
                    )
                with gr.Column(scale=1):
                    out_demo  = gr.Video(label="Annotated Output")
                    stat_demo = gr.Textbox(label="Status", interactive=False)

            run_demo_btn.click(
                fn=run_demo,
                inputs=[conf_demo, maxdet_demo],
                outputs=[out_demo, stat_demo]
            )

    gr.Markdown("""
    ---
    | Step | Module | Detail |
    |---|---|---|
    | Detection | YOLOv8n | Person detection, confidence filtered |
    | Tracking | DeepSORT | Kalman Filter + appearance Re-ID |
    | Team Classification | HSV K-Means | Jersey color clustering |
    | Scene Detection | Frame diff | Tracker reset on camera cuts |

    **Source code:** [GitHub](https://github.com/your-username/multi-object-tracking) |
    **Video source:** [FA Cup 2024](https://youtu.be/X0we8220k74)
    """)

demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())