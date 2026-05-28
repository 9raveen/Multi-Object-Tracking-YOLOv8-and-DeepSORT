import numpy as np
import cv2
import csv
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
from collections import defaultdict


class AnalyticsTracker:
    """
    Tracks per-player analytics across all frames:
    - Position history (for heatmap)
    - Speed estimation (pixels → km/h)
    - Distance covered
    - Time on screen
    - Ball possession
    """

    # Standard football pitch dimensions
    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M  = 68.0

    def __init__(self, fps=25, frame_width=1920, frame_height=1080):
        self.fps          = fps
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # pixels per metre estimation
        # assume pitch covers ~80% of frame width and ~70% of frame height
        self.px_per_m = (frame_width * 0.80) / self.PITCH_LENGTH_M

        # per-player data
        self.positions   = defaultdict(list)   # track_id → [(cx,cy,frame_idx)]
        self.teams       = {}                  # track_id → team name
        self.speeds      = defaultdict(list)   # track_id → [speed_kmh]
        self.last_pos    = {}                  # track_id → (cx, cy)

        # ball data
        self.ball_positions = []              # [(cx, cy, frame_idx)]
        self.possession     = defaultdict(int) # team → frame count

        # current frame speeds for overlay
        self.current_speeds = {}  # track_id → speed_kmh

    def update(self, tracks, team_assignments, ball_pos, frame_idx):
        """
        Call every frame with current tracks and ball position.
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            team = team_assignments.get(track_id, "Other")
            self.teams[track_id] = team

            # Record position
            self.positions[track_id].append((cx, cy, frame_idx))

            # Speed estimation
            if track_id in self.last_pos:
                px, py = self.last_pos[track_id]
                pixel_dist = ((cx - px)**2 + (cy - py)**2) ** 0.5
                metres     = pixel_dist / self.px_per_m
                # distance per frame → km/h
                speed_kmh  = metres * self.fps * 3.6
                # cap at realistic max (40 km/h)
                speed_kmh  = min(speed_kmh, 40.0)
                self.speeds[track_id].append(speed_kmh)
                self.current_speeds[track_id] = speed_kmh
            else:
                self.current_speeds[track_id] = 0.0

            self.last_pos[track_id] = (cx, cy)

        # Ball tracking
        if ball_pos is not None:
            self.ball_positions.append((*ball_pos, frame_idx))
            # Possession — which team is closest to ball
            closest_team = self._closest_team_to_ball(
                ball_pos, tracks, team_assignments)
            if closest_team:
                self.possession[closest_team] += 1

    def _closest_team_to_ball(self, ball_pos, tracks, team_assignments):
        if not tracks or ball_pos is None:
            return None
        bx, by    = ball_pos
        best_dist = float("inf")
        best_team = None
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cx   = (x1 + x2) / 2
            cy   = (y1 + y2) / 2
            dist = ((cx - bx)**2 + (cy - by)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_team = team_assignments.get(track_id, "Other")
        return best_team

    def get_speed(self, track_id):
        """Get current speed for overlay display."""
        return self.current_speeds.get(track_id, 0.0)

    def reset_on_cut(self):
        """Clear last positions on scene cut — prevents jump speeds."""
        self.last_pos.clear()
        self.current_speeds.clear()

    # ── Report Generation ─────────────────────────────────────────────────

    def save_csv_report(self, output_path="outputs/player_analytics.csv"):
        """Generate per-player analytics CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rows = []
        for track_id, positions in self.positions.items():
            team         = self.teams.get(track_id, "Other")
            frames_seen  = len(positions)
            time_on_screen = frames_seen / self.fps

            speeds       = self.speeds.get(track_id, [0])
            avg_speed    = np.mean(speeds) if speeds else 0
            max_speed    = np.max(speeds)  if speeds else 0

            # total distance
            total_dist_px = 0
            for i in range(1, len(positions)):
                x1, y1, _ = positions[i-1]
                x2, y2, _ = positions[i]
                total_dist_px += ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            total_dist_m = total_dist_px / self.px_per_m

            rows.append({
                "track_id":        track_id,
                "team":            team,
                "frames_seen":     frames_seen,
                "time_on_screen_s": round(time_on_screen, 2),
                "total_distance_m": round(total_dist_m, 2),
                "avg_speed_kmh":   round(avg_speed, 2),
                "max_speed_kmh":   round(max_speed, 2),
            })

        rows.sort(key=lambda x: x["total_distance_m"], reverse=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV report saved: {output_path}")
        return output_path

    def save_heatmaps(self, output_dir="outputs"):
        """Generate per-team heatmaps overlaid on pitch diagram."""
        os.makedirs(output_dir, exist_ok=True)

        teams = ["Man Utd", "Man City"]
        colors = {
            "Man Utd":  "Reds",
            "Man City": "Blues"
        }

        for team in teams:
            heatmap = np.zeros((self.frame_height, self.frame_width),
                               dtype=np.float32)

            for track_id, positions in self.positions.items():
                if self.teams.get(track_id) != team:
                    continue
                for cx, cy, _ in positions:
                    if 0 <= cy < self.frame_height and \
                       0 <= cx < self.frame_width:
                        heatmap[cy, cx] += 1

            if heatmap.max() == 0:
                print(f"No data for {team} heatmap")
                continue

            # Gaussian blur for smooth heatmap
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            heatmap = heatmap / heatmap.max()  # normalise 0-1

            fig, ax = plt.subplots(figsize=(12, 7))

            # Draw pitch background
            self._draw_pitch(ax)

            # Overlay heatmap
            ax.imshow(heatmap,
                      cmap=colors[team],
                      alpha=0.6,
                      extent=[0, self.frame_width,
                               self.frame_height, 0])

            ax.set_title(f"{team} — Player Position Heatmap",
                         fontsize=14, fontweight="bold")
            ax.axis("off")

            fname = f"{output_dir}/heatmap_{team.replace(' ','_')}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Heatmap saved: {fname}")

    def save_possession_chart(self, output_dir="outputs"):
        """Bar chart showing ball possession % per team."""
        os.makedirs(output_dir, exist_ok=True)

        if not self.possession:
            print("No possession data recorded.")
            return

        total  = sum(self.possession.values())
        labels = list(self.possession.keys())
        values = [round(v/total*100, 1) for v in self.possession.values()]
        colors_map = {
            "Man Utd": "#e63946",
            "Man City": "#4cc9f0",
            "Other":   "#888888"
        }
        bar_colors = [colors_map.get(l, "#888888") for l in labels]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white")
        ax.set_ylabel("Possession %", fontsize=12)
        ax.set_title("Ball Possession by Team", fontsize=13,
                     fontweight="bold")
        ax.set_ylim(0, 100)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"{val}%", ha="center", fontsize=11,
                    fontweight="bold")

        plt.tight_layout()
        fname = f"{output_dir}/possession_chart.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Possession chart saved: {fname}")

    def _draw_pitch(self, ax):
        """Draw a basic football pitch outline."""
        w, h = self.frame_width, self.frame_height
        ax.set_facecolor("#2d6a2d")
        # Pitch outline
        rect = plt.Rectangle((w*0.05, h*0.05),
                               w*0.90, h*0.90,
                               fill=False, color="white", lw=2)
        ax.add_patch(rect)
        # Centre line
        ax.axvline(x=w*0.5, color="white", lw=2,
                   ymin=0.05, ymax=0.95)
        # Centre circle
        circle = plt.Circle((w*0.5, h*0.5), w*0.07,
                              fill=False, color="white", lw=2)
        ax.add_patch(circle)