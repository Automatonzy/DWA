#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk


@dataclass
class MapMeta:
    image: str
    resolution: float
    origin: tuple[float, float, float]


@dataclass
class PickedPoint:
    index: int
    row: int
    col: int
    x: float
    y: float
    raw_free: bool
    clearance_free: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "grid": {"row": self.row, "col": self.col},
            "world": {"x": self.x, "y": self.y},
            "raw_free": self.raw_free,
            "clearance_free": self.clearance_free,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click on a nav-map preview image and read back world coordinates.")
    parser.add_argument("--map", required=True, help="Path to nav-map metadata (.json/.yaml).")
    parser.add_argument(
        "--preview",
        default=None,
        help="Preview image to display. Defaults to preview.ppm next to the map metadata.",
    )
    parser.add_argument("--scale", type=int, default=6, help="Integer zoom factor for the display.")
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.25,
        help="Safety clearance in meters used to judge whether a point is comfortably free.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to save all picked points on exit.",
    )
    return parser.parse_args()


def load_meta(path: Path) -> MapMeta:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = {}
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value.strip("'\"")
    else:
        raise ValueError(f"Unsupported map metadata extension: {path.suffix}")

    origin = data.get("origin", [0.0, 0.0, 0.0])
    return MapMeta(
        image=str(data["image"]),
        resolution=float(data["resolution"]),
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
    )


def load_pgm(path: Path) -> list[list[bool]]:
    with path.open("rb") as fh:
        magic = fh.readline().strip()
        if magic != b"P5":
            raise ValueError("Only binary PGM (P5) occupancy maps are supported.")

        def next_token_line():
            line = fh.readline()
            while line.startswith(b"#"):
                line = fh.readline()
            return line.strip()

        dims = next_token_line()
        while len(dims.split()) < 2:
            dims += b" " + next_token_line()
        width, height = map(int, dims.split())
        _max_value = int(next_token_line())
        raw = fh.read(width * height)
        if len(raw) != width * height:
            raise ValueError("PGM file ended unexpectedly.")
        pixels = list(raw)

    grid: list[list[bool]] = []
    for row in range(height):
        row_values = []
        for col in range(width):
            value = pixels[row * width + col]
            row_values.append(value == 0)
        grid.append(row_values)
    return grid


class GridMap:
    def __init__(self, occupancy: list[list[bool]], resolution: float, origin: tuple[float, float, float]):
        self.occupancy = occupancy
        self.resolution = resolution
        self.origin = origin
        self.height = len(occupancy)
        self.width = len(occupancy[0]) if occupancy else 0

    @classmethod
    def from_meta_file(cls, meta_path: Path) -> "GridMap":
        meta = load_meta(meta_path)
        image_path = Path(meta.image)
        if not image_path.is_absolute():
            image_path = (meta_path.parent / image_path).resolve()
        occupancy = load_pgm(image_path)
        return cls(occupancy=occupancy, resolution=meta.resolution, origin=meta.origin)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_occupied(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return self.occupancy[row][col]

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.origin[0] + (col + 0.5) * self.resolution
        row_from_bottom = self.height - 1 - row
        y = self.origin[1] + (row_from_bottom + 0.5) * self.resolution
        return x, y

    def inflate(self, radius_m: float) -> "GridMap":
        radius_cells = int((radius_m / self.resolution) + 0.999999)
        if radius_cells <= 0:
            return self
        inflated = [[self.occupancy[r][c] for c in range(self.width)] for r in range(self.height)]
        offsets = []
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc <= radius_cells * radius_cells:
                    offsets.append((dr, dc))
        for row in range(self.height):
            for col in range(self.width):
                if not self.occupancy[row][col]:
                    continue
                for dr, dc in offsets:
                    rr = row + dr
                    cc = col + dc
                    if 0 <= rr < self.height and 0 <= cc < self.width:
                        inflated[rr][cc] = True
        return GridMap(inflated, self.resolution, self.origin)


def default_preview_path(map_path: Path) -> Path:
    preview_path = map_path.parent / "preview.ppm"
    if not preview_path.exists():
        raise FileNotFoundError("preview.ppm not found next to the map metadata; pass --preview explicitly.")
    return preview_path


def main():
    args = parse_args()
    map_path = Path(args.map).expanduser().resolve()
    preview_path = (
        Path(args.preview).expanduser().resolve()
        if args.preview is not None
        else default_preview_path(map_path)
    )

    grid_map = GridMap.from_meta_file(map_path)
    clearance_map = grid_map.inflate(args.clearance)

    root = tk.Tk()
    root.title(f"Nav Point Picker - {preview_path.name}")
    root.resizable(False, False)

    image = tk.PhotoImage(file=str(preview_path))
    display_image = image.zoom(args.scale, args.scale) if args.scale != 1 else image

    canvas = tk.Canvas(root, width=display_image.width(), height=display_image.height(), highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=display_image)

    status_var = tk.StringVar()
    status_var.set("Left click: add point | Right click: undo | C: clear | S: save | Q/Esc: quit")
    status = tk.Label(root, textvariable=status_var, anchor="w", justify="left")
    status.pack(fill=tk.X)

    picked_points: list[PickedPoint] = []
    marker_ids: list[tuple[int, int]] = []

    def canvas_to_grid(event_x: int, event_y: int):
        col = event_x // args.scale
        row = event_y // args.scale
        if not grid_map.in_bounds(row, col):
            return None
        return row, col

    def build_point(row: int, col: int) -> PickedPoint:
        x, y = grid_map.grid_to_world(row, col)
        return PickedPoint(
            index=len(picked_points) + 1,
            row=row,
            col=col,
            x=x,
            y=y,
            raw_free=not grid_map.is_occupied(row, col),
            clearance_free=not clearance_map.is_occupied(row, col),
        )

    def redraw_markers():
        for oval_id, text_id in marker_ids:
            canvas.delete(oval_id)
            canvas.delete(text_id)
        marker_ids.clear()
        radius = max(3, args.scale)
        for point in picked_points:
            cx = point.col * args.scale + args.scale / 2
            cy = point.row * args.scale + args.scale / 2
            color = "#2e8b57" if point.clearance_free else "#cc5500"
            oval_id = canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=color, width=2)
            text_id = canvas.create_text(cx + radius + 6, cy, text=str(point.index), fill=color, anchor=tk.W)
            marker_ids.append((oval_id, text_id))

    def save_points():
        if args.output is None:
            return
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "map": str(map_path),
            "preview": str(preview_path),
            "clearance": args.clearance,
            "points": [point.to_dict() for point in picked_points],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"[INFO] Saved {len(picked_points)} points to {output_path}")

    def print_plan_hint():
        if len(picked_points) < 2:
            return
        start = picked_points[-2]
        goal = picked_points[-1]
        print()
        print("[HINT] Example A* visualisation command with the last two points:")
        print(
            "python scripts/navigation/plan_astar.py "
            f"--map {map_path} "
            f"--start {start.x:.3f} {start.y:.3f} "
            f"--goal {goal.x:.3f} {goal.y:.3f} "
            "--inflate-radius 0.30"
        )
        print()
        print("[HINT] Example play_nav_cs.py command (fill in <USD_MAP_PATH> and adjust yaw values):")
        lines = [
            "python scripts/reinforcement_learning/rsl_rl/play_nav_cs.py \\",
            "  --task RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \\",
            "  --checkpoint /home/y/DWA/flat/model_8500.pt \\",
            "  --num_envs 1 \\",
            "  --map <USD_MAP_PATH> \\",
            f"  --nav-map {map_path} \\",
            f"  --goal {goal.x:.3f} {goal.y:.3f} \\",
            "  --goal-yaw 0.0 \\",
            "  --goal-yaw-tolerance 0.10 \\",
            f"  --spawn {start.x:.3f} {start.y:.3f} 0.0 \\",
            "  --inflate-radius 0.3 \\",
            "  --local-clearance-radius 0.25 \\",
            "  --settle-steps 120 \\",
            "  --debug-print-every 20 \\",
            "  --max-steps 3000 \\",
            "  --head-camera \\",
            "  --goal-tolerance 0.35 \\",
            "  --dataset-dir /home/y/DWA/episodes",
        ]
        print("\n".join(lines))

    def on_motion(event):
        grid = canvas_to_grid(event.x, event.y)
        if grid is None:
            status_var.set("Cursor outside map")
            return
        row, col = grid
        x, y = grid_map.grid_to_world(row, col)
        raw_state = "free" if not grid_map.is_occupied(row, col) else "occupied"
        clearance_state = "free" if not clearance_map.is_occupied(row, col) else "occupied"
        status_var.set(
            f"grid=({row}, {col}) world=({x:.3f}, {y:.3f}) raw={raw_state} clearance={clearance_state}"
        )

    def on_left_click(event):
        grid = canvas_to_grid(event.x, event.y)
        if grid is None:
            return
        point = build_point(*grid)
        picked_points.append(point)
        redraw_markers()
        print(
            f"[POINT] {point.index:02d}: grid=({point.row}, {point.col}) "
            f"world=({point.x:.3f}, {point.y:.3f}) raw_free={point.raw_free} clearance_free={point.clearance_free}"
        )
        print_plan_hint()

    def on_right_click(_event):
        if not picked_points:
            return
        removed = picked_points.pop()
        redraw_markers()
        print(
            f"[UNDO] removed point {removed.index}: grid=({removed.row}, {removed.col}) "
            f"world=({removed.x:.3f}, {removed.y:.3f})"
        )

    def clear_points(_event=None):
        if not picked_points:
            return
        picked_points.clear()
        redraw_markers()
        print("[INFO] Cleared all picked points.")

    def save_shortcut(_event=None):
        save_points()

    def close_window(_event=None):
        save_points()
        root.destroy()

    canvas.bind("<Motion>", on_motion)
    canvas.bind("<Button-1>", on_left_click)
    canvas.bind("<Button-3>", on_right_click)
    root.bind("<KeyPress-c>", clear_points)
    root.bind("<KeyPress-C>", clear_points)
    root.bind("<KeyPress-s>", save_shortcut)
    root.bind("<KeyPress-S>", save_shortcut)
    root.bind("<KeyPress-q>", close_window)
    root.bind("<KeyPress-Q>", close_window)
    root.bind("<Escape>", close_window)
    root.protocol("WM_DELETE_WINDOW", close_window)

    print("[INFO] Nav point picker started.")
    print("[INFO] Left click to add a point, right click to undo, C to clear, S to save, Q/Esc to quit.")
    print(f"[INFO] map={map_path}")
    print(f"[INFO] preview={preview_path}")

    root.mainloop()


if __name__ == "__main__":
    main()
