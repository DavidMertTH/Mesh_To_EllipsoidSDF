"""
train_neural_triangle.py -- interactive TriangleDeepestNet trainer.

TriangleDeepestNet(Ellipsoid, Triangle) -> bary(u, v, w)
q = u*p1 + v*p2 + w*p3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from neural_gradient import vec_to_surface_exact_torch
from neural_sdf import sdf_exact_torch
from neural_triangle import (
    EarlyStopping,
    generate_triangle_samples_torch,
    triangle_deepest_exact_torch,
    triangle_deepest_refined_torch,
)


_DEMO_CASES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "Tilted 3D": (
        np.array([0.58, 0.36, 0.22], dtype=np.float32),
        np.array([
            [-0.62, -0.28, 0.08],
            [0.58, -0.18, -0.04],
            [0.18, 0.58, 0.05],
        ], dtype=np.float32),
    ),
    "Edge slice": (
        np.array([0.52, 0.46, 0.20], dtype=np.float32),
        np.array([
            [-0.54, 0.12, 0.0],
            [0.66, 0.18, 0.0],
            [0.12, 0.72, 0.0],
        ], dtype=np.float32),
    ),
    "Interior": (
        np.array([0.64, 0.38, 0.16], dtype=np.float32),
        np.array([
            [-0.68, -0.36, 0.04],
            [0.72, -0.12, -0.02],
            [0.18, 0.66, 0.07],
        ], dtype=np.float32),
    ),
    "Outside": (
        np.array([0.48, 0.34, 0.20], dtype=np.float32),
        np.array([
            [0.62, 0.30, 0.04],
            [1.08, 0.18, -0.02],
            [0.82, 0.74, 0.06],
        ], dtype=np.float32),
    ),
}
_DEFAULT_DEMO_CASE = "Tilted 3D"


def _demo_case_arrays(name: str | None) -> tuple[np.ndarray, np.ndarray]:
    return _DEMO_CASES.get(str(name), _DEMO_CASES[_DEFAULT_DEMO_CASE])

RUNS_DIR = Path(__file__).resolve().parent / "saved_runs" / "neural_triangle"
RUNS_INDEX = RUNS_DIR / "runs.json"


def _fmt_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not np.isfinite(seconds):
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fmt_loss(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.5f}" if value >= 1e-4 else f"{value:.2e}"


def _fmt_bary(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] < 3:
        return "-"
    return f"{arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f}"


def _load_run_records() -> list[dict]:
    if not RUNS_INDEX.exists():
        return []
    try:
        data = json.loads(RUNS_INDEX.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = data if isinstance(data, list) else data.get("runs", [])
    if not isinstance(records, list):
        return []
    existing = []
    for record in records:
        if isinstance(record, dict) and record.get("checkpoint"):
            if Path(record["checkpoint"]).exists():
                existing.append(record)
    return existing


def _save_run_records(records: list[dict]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_INDEX.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _append_run_record(record: dict) -> None:
    records = _load_run_records()
    records.append(record)
    records.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
    _save_run_records(records)


def _run_record_label(record: dict) -> str:
    ts = str(record.get("timestamp", "-")).replace("T", " ")[:19]
    phase = record.get("phase", "-")
    final_loss = _fmt_loss(record.get("final_loss"))
    best_loss = _fmt_loss(record.get("best_loss"))
    return f"{ts} | {phase} | final {final_loss} | best {best_loss}"


def _float_from_line(edit: QtWidgets.QLineEdit, fallback: float) -> float:
    try:
        return float(edit.text().strip())
    except ValueError:
        return fallback


def _parse_layers(text: str, fallback: list[int]) -> list[int]:
    values: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            return fallback
        if v <= 0:
            return fallback
        values.append(v)
    return values or fallback


def _make_mlp(input_dim: int, layers: list[int]) -> tuple[nn.Sequential, int]:
    modules: list[nn.Module] = []
    last = input_dim
    for width in layers:
        modules += [nn.Linear(last, width), nn.SiLU()]
        last = width
    return nn.Sequential(*modules), last


def _normalize_gradient(points: torch.Tensor, radii: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
    vec_to_surface = vec_to_surface_exact_torch(points, radii)
    norm = vec_to_surface.norm(dim=-1, keepdim=True)

    fallback = points / radii.clamp(min=1e-8).pow(2)
    fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    outside_grad = -vec_to_surface / norm.clamp(min=1e-8)
    inside_grad = vec_to_surface / norm.clamp(min=1e-8)
    grad = torch.where((sdf < 0.0).unsqueeze(-1), inside_grad, outside_grad)
    grad = torch.where(norm > 1e-7, grad, fallback)
    return grad.float()


def triangle_input(
    radii: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r_max = radii.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    inp = torch.cat([radii / r_max, v1 / r_max, v2 / r_max, v3 / r_max], dim=-1)
    return inp.float(), r_max.squeeze(-1)


def bary_to_point(
    bary: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
) -> torch.Tensor:
    return bary[:, 0:1] * v1 + bary[:, 1:2] * v2 + bary[:, 2:3] * v3


def triangle_gt(
    radii: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    grid_G: int,
    refine_steps: int,
    refine_grid: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if refine_steps > 0:
        bary = triangle_deepest_refined_torch(
            radii, v1, v2, v3, G=grid_G,
            refine_steps=refine_steps, refine_grid=refine_grid,
        )
    else:
        bary = triangle_deepest_exact_torch(radii, v1, v2, v3, G=grid_G)
    point = bary_to_point(bary, v1, v2, v3)
    with torch.no_grad():
        sdf = sdf_exact_torch(point, radii)
        inside = (sdf < 0.0).float()
        grad = _normalize_gradient(point, radii, sdf)
    return bary.float(), point.float(), inside, grad


class TriangleDeepestNet(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        self.layers = list(layers)
        self.body, last = _make_mlp(12, layers)
        self.head = nn.Linear(last, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(self.body(x)), dim=-1)


class CollapsibleSection(QtWidgets.QWidget):
    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("foldout")
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("foldoutHeader")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setAutoRaise(True)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.clicked.connect(self._sync)

        self.body = QtWidgets.QWidget()
        self.body.setObjectName("foldoutBody")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self._sync(expanded)

    def set_content_layout(self, layout: QtWidgets.QLayout) -> None:
        self.body.setLayout(layout)

    def _sync(self, checked: bool) -> None:
        self.toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.body.setVisible(checked)


class WheelGuard(QtCore.QObject):
    """Prevent mouse-wheel edits on focused form fields inside scroll areas."""

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.Wheel:
            scroll_area = watched.parent()
            while scroll_area is not None and not isinstance(scroll_area, QtWidgets.QScrollArea):
                scroll_area = scroll_area.parent()
            if isinstance(scroll_area, QtWidgets.QScrollArea):
                QtWidgets.QApplication.sendEvent(scroll_area.viewport(), event)
            return True
        return super().eventFilter(watched, event)


@dataclass
class RunArchive:
    name: str
    steps: list[int]
    losses: list[float]
    curve: pg.PlotDataItem
    action: QtGui.QAction


class TriangleTrainingWorker(QtCore.QThread):
    progress = QtCore.Signal(object)
    validation = QtCore.Signal(object)
    demo = QtCore.Signal(object)
    finished_info = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, params: dict, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.params = dict(params)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._run_training()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_training(self) -> None:
        p = self.params
        device = p["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

        tri_net = TriangleDeepestNet(p["triangle_layers"]).to(device)
        self._load_optional(tri_net, p["triangle_load"], device, "triangle")
        opt = torch.optim.Adam(tri_net.parameters(), lr=float(p["lr"]))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(p["steps"]), eta_min=float(p["eta_min"])
        )
        early = EarlyStopping(
            patience=int(p["early_patience"]),
            min_delta=float(p["early_min_delta"]),
            warmup=int(p["early_warmup"]),
            smooth=int(p["early_smooth"]),
            restore_best=False,
        )

        cache = self._precompute_triangle_cache(device, p) if p["precompute_samples"] > 0 else None
        val_cache = self._build_validation_cache(device, p)
        total_steps = int(p["steps"])
        log_every = max(1, int(p["log_every"]))
        viz_every = max(1, int(p["viz_every"]))
        t0 = time.monotonic()
        best_loss = float("inf")
        best_step = 0
        final_loss: float | None = None
        stop_reason = "completed"

        tri_net.train()
        for step in range(1, total_steps + 1):
            if self._stop_requested:
                stop_reason = "stopped"
                break

            loss, parts = self._training_step(tri_net, cache, device, p)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            loss_value = float(loss.detach().cpu())
            final_loss = loss_value
            if np.isfinite(loss_value) and loss_value < best_loss:
                best_loss = loss_value
                best_step = step

            stop_now = early.update(step, loss_value)
            elapsed = time.monotonic() - t0
            speed = step / elapsed if elapsed > 1e-9 else 0.0
            eta = (total_steps - step) / speed if speed > 1e-9 else None

            if step == 1 or step % log_every == 0 or stop_now:
                self.progress.emit({
                    "step": step,
                    "total": total_steps,
                    "loss": loss_value,
                    "best_loss": best_loss,
                    "best_step": best_step,
                    "lr": sched.get_last_lr()[0],
                    "elapsed": elapsed,
                    "speed": speed,
                    "eta": eta,
                    "device": device,
                    "state": "running",
                    "cache": f"{cache['inp'].shape[0]:,}" if cache is not None else "-",
                    "loss_parts": parts,
                    "phase": "triangle",
                })

            if step == 1 or step % viz_every == 0 or stop_now:
                self._emit_validation(tri_net, device, step, val_cache, p)
                self._emit_demo(tri_net, device, step, p)

            if stop_now:
                stop_reason = "early_stop"
                break

        tri_net.eval()
        run_record = self._persist_run_checkpoint(
            tri_net, p, stop_reason,
            best_loss if np.isfinite(best_loss) else None,
            final_loss,
            best_step,
            time.monotonic() - t0,
        )
        if p["save_path"]:
            self._save_checkpoint(str(p["save_path"]), tri_net, p)

        self.finished_info.emit({
            "reason": stop_reason,
            "best_loss": best_loss if np.isfinite(best_loss) else None,
            "final_loss": final_loss,
            "best_step": best_step,
            "elapsed": time.monotonic() - t0,
            "save_path": str(p["save_path"]) if p["save_path"] else run_record.get("checkpoint", ""),
            "run_record": run_record,
        })

    def _precompute_triangle_cache(self, device, p: dict) -> dict[str, torch.Tensor] | None:
        total = int(p["precompute_samples"])
        chunk = max(1, int(p["precompute_chunk"]))
        inp_parts: list[torch.Tensor] = []
        radii_parts: list[torch.Tensor] = []
        v1_parts: list[torch.Tensor] = []
        v2_parts: list[torch.Tensor] = []
        v3_parts: list[torch.Tensor] = []
        bary_parts: list[torch.Tensor] = []
        point_parts: list[torch.Tensor] = []
        done = 0
        t0 = time.monotonic()
        while done < total and not self._stop_requested:
            n = min(chunk, total - done)
            radii, v1, v2, v3 = generate_triangle_samples_torch(n, p["r_min"], p["r_max"], device)
            inp, _ = triangle_input(radii, v1, v2, v3)
            bary, point, _, _ = triangle_gt(
                radii, v1, v2, v3,
                int(p["grid_G"]), int(p["gt_refine_steps"]), int(p["gt_refine_grid"]),
            )
            inp_parts.append(inp.detach())
            radii_parts.append(radii.detach())
            v1_parts.append(v1.detach())
            v2_parts.append(v2.detach())
            v3_parts.append(v3.detach())
            bary_parts.append(bary.detach())
            point_parts.append(point.detach())
            done += n

            elapsed = time.monotonic() - t0
            speed = done / elapsed if elapsed > 1e-9 else 0.0
            eta = (total - done) / speed if speed > 1e-9 else None
            self.progress.emit({
                "step": done,
                "total": total,
                "loss": None,
                "best_loss": None,
                "best_step": 0,
                "lr": p["lr"],
                "elapsed": elapsed,
                "speed": speed,
                "eta": eta,
                "device": device,
                "state": "precomputing GT",
                "cache": f"{done:,} / {total:,}",
                "loss_parts": {},
                "phase": "triangle",
            })

        if not inp_parts:
            return None
        return {
            "inp": torch.cat(inp_parts, dim=0).contiguous(),
            "radii": torch.cat(radii_parts, dim=0).contiguous(),
            "v1": torch.cat(v1_parts, dim=0).contiguous(),
            "v2": torch.cat(v2_parts, dim=0).contiguous(),
            "v3": torch.cat(v3_parts, dim=0).contiguous(),
            "bary": torch.cat(bary_parts, dim=0).contiguous(),
            "point": torch.cat(point_parts, dim=0).contiguous(),
        }

    def _build_validation_cache(self, device, p: dict) -> dict[str, torch.Tensor] | None:
        total = int(p.get("validation_samples", 512))
        if total <= 0:
            return None

        seed = int(p.get("validation_seed", 1729))
        torch_device = torch.device(device)
        rng_devices = [torch_device.index or torch.cuda.current_device()] if torch_device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            if torch_device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            radii, v1, v2, v3 = generate_triangle_samples_torch(total, p["r_min"], p["r_max"], device)
            inp, _ = triangle_input(radii, v1, v2, v3)
            bary, point, _, _ = triangle_gt(
                radii, v1, v2, v3,
                int(p["grid_G"]), int(p["gt_refine_steps"]), int(p["gt_refine_grid"]),
            )
            return {
                "inp": inp.detach().contiguous(),
                "radii": radii.detach().contiguous(),
                "v1": v1.detach().contiguous(),
                "v2": v2.detach().contiguous(),
                "v3": v3.detach().contiguous(),
                "bary": bary.detach().contiguous(),
                "point": point.detach().contiguous(),
            }

    def _training_step(
        self,
        tri_net: TriangleDeepestNet,
        cache: dict[str, torch.Tensor] | None,
        device,
        p: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if cache is not None:
            idx = torch.randint(0, cache["inp"].shape[0], (p["batch_size"],), device=device)
            tri_inp = cache["inp"][idx]
            bary_gt = cache["bary"][idx]
            q_gt = cache["point"][idx]
            radii = cache["radii"][idx]
            v1 = cache["v1"][idx]
            v2 = cache["v2"][idx]
            v3 = cache["v3"][idx]
        else:
            radii, v1, v2, v3 = generate_triangle_samples_torch(
                p["batch_size"], p["r_min"], p["r_max"], device
            )
            tri_inp, _ = triangle_input(radii, v1, v2, v3)
            bary_gt, q_gt, _, _ = triangle_gt(
                radii, v1, v2, v3,
                int(p["grid_G"]), int(p["gt_refine_steps"]), int(p["gt_refine_grid"]),
            )

        bary_pred = tri_net(tri_inp)
        total, parts, _ = self._triangle_loss(
            bary_pred, bary_gt, radii, v1, v2, v3, q_gt, p
        )
        parts["total"] = float(total.detach().cpu())
        return total, parts

    @staticmethod
    def _triangle_loss(
        bary_pred: torch.Tensor,
        bary_gt: torch.Tensor,
        radii: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor,
        q_gt: torch.Tensor,
        p: dict,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        q_pred = bary_to_point(bary_pred, v1, v2, v3)
        r_scale = radii.max(dim=-1).values.clamp(min=1e-8)
        bary_loss = nn.functional.l1_loss(bary_pred, bary_gt)
        point_loss = ((q_pred - q_gt).norm(dim=-1) / r_scale).mean()

        sdf_weight = float(p.get("loss_sdf_weight", 0.0))
        if sdf_weight:
            sdf_pred = sdf_exact_torch(q_pred, radii)
            sdf_gt = sdf_exact_torch(q_gt, radii)
            sdf_loss = ((sdf_pred - sdf_gt).abs() / r_scale).mean()
        else:
            with torch.no_grad():
                sdf_pred = sdf_exact_torch(q_pred, radii)
                sdf_gt = sdf_exact_torch(q_gt, radii)
                sdf_loss = ((sdf_pred - sdf_gt).abs() / r_scale).mean()

        total = (
            float(p.get("loss_bary_weight", 1.0)) * bary_loss
            + float(p.get("loss_point_weight", 0.25)) * point_loss
            + sdf_weight * sdf_loss
        )
        return total, {
            "bary": float(bary_loss.detach().cpu()),
            "point": float(point_loss.detach().cpu()),
            "sdf": float(sdf_loss.detach().cpu()),
            "triangle": float(total.detach().cpu()),
        }, q_pred

    def _emit_validation(
        self,
        tri_net: TriangleDeepestNet,
        device,
        step: int,
        cache: dict[str, torch.Tensor] | None,
        p: dict,
    ) -> None:
        if cache is None:
            return
        tri_net.eval()
        with torch.no_grad():
            bary_pred = tri_net(cache["inp"])
            total, parts, _ = self._triangle_loss(
                bary_pred, cache["bary"], cache["radii"],
                cache["v1"], cache["v2"], cache["v3"], cache["point"], p,
            )
            parts["total"] = float(total.detach().cpu())
            self.validation.emit({
                "step": step,
                "loss": float(total.detach().cpu()),
                "parts": parts,
                "phase": "triangle",
                "samples": int(cache["inp"].shape[0]),
                "device": device,
            })
        tri_net.train()

    def _emit_demo(
        self,
        tri_net: TriangleDeepestNet,
        device,
        step: int,
        p: dict,
    ) -> None:
        tri_net.eval()
        demo_radii, demo_tri = _demo_case_arrays(p.get("demo_case"))
        r = torch.as_tensor(demo_radii[None], dtype=torch.float32, device=device)
        v = torch.as_tensor(demo_tri[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            gt_bary, gt_point, _, gt_grad = triangle_gt(
                r, v[:, 0], v[:, 1], v[:, 2],
                grid_G=60, refine_steps=6, refine_grid=9,
            )
            tri_inp, _ = triangle_input(r, v[:, 0], v[:, 1], v[:, 2])
            bary = tri_net(tri_inp)
            triangle_demo_loss, tri_parts, tri_q = self._triangle_loss(
                bary, gt_bary, r, v[:, 0], v[:, 1], v[:, 2], gt_point, p,
            )
            q = tri_q
            sdf = sdf_exact_torch(q, r)
            exact_grad = _normalize_gradient(q, r, sdf)
            gt_sdf = sdf_exact_torch(gt_point, r)
            demo_loss = triangle_demo_loss
            point_error = (q - gt_point).norm(dim=-1)
            sdf_error = (sdf - gt_sdf).abs()
            grad_align = nn.functional.cosine_similarity(exact_grad, gt_grad, dim=-1)
            display_grad = exact_grad
            target_grad_point = gt_point
            target_grad = gt_grad
        tri_net.train()
        self.demo.emit({
            "step": step,
            "phase": "triangle",
            "bary": bary[0].detach().cpu().numpy(),
            "gt_bary": gt_bary[0].detach().cpu().numpy(),
            "point": q[0].detach().cpu().numpy(),
            "gt_point": gt_point[0].detach().cpu().numpy(),
            "target_grad_point": target_grad_point[0].detach().cpu().numpy(),
            "sdf": float(sdf[0].detach().cpu()),
            "gt_sdf": float(gt_sdf[0].detach().cpu()),
            "demo_loss": float(demo_loss.detach().cpu()),
            "demo_bary_l1": float(tri_parts["bary"]),
            "demo_point_error": float(point_error[0].detach().cpu()),
            "demo_sdf_error": float(sdf_error[0].detach().cpu()),
            "demo_grad_align": float(grad_align[0].detach().cpu()),
            "grad": display_grad[0].detach().cpu().numpy(),
            "target_grad": target_grad[0].detach().cpu().numpy(),
        })

    @staticmethod
    def _load_optional(module: nn.Module, path: str, device, key: str) -> None:
        if not path:
            return
        ckpt = torch.load(path, map_location=device, weights_only=True)
        if key in ckpt and isinstance(ckpt[key], dict):
            module.load_state_dict(ckpt[key])
        elif "state_dict" in ckpt:
            module.load_state_dict(ckpt["state_dict"])

    @staticmethod
    def _save_checkpoint(path: str, tri_net: TriangleDeepestNet, p: dict) -> None:
        torch.save({
            "kind": "triangle_deepest",
            "phase": "triangle",
            "triangle": tri_net.state_dict(),
            "triangle_layers": p["triangle_layers"],
        }, path)

    @staticmethod
    def _persist_run_checkpoint(
        tri_net: TriangleDeepestNet,
        p: dict,
        reason: str,
        best_loss: float | None,
        final_loss: float | None,
        best_step: int,
        elapsed: float,
    ) -> dict:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        loss_label = "nan" if final_loss is None else f"{final_loss:.6g}"
        filename = f"{timestamp.replace(':', '-')}_triangle_loss-{loss_label}.pt"
        checkpoint = RUNS_DIR / filename
        record = {
            "timestamp": timestamp,
            "phase": "triangle",
            "reason": reason,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "best_step": int(best_step),
            "elapsed": float(elapsed),
            "checkpoint": str(checkpoint),
            "triangle_layers": p["triangle_layers"],
        }
        torch.save({
            "kind": "triangle_deepest",
            "record": record,
            "phase": "triangle",
            "triangle": tri_net.state_dict(),
            "triangle_layers": p["triangle_layers"],
        }, checkpoint)
        _append_run_record(record)
        return record


class TriangleTrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.setWindowTitle("Triangle Training")
        self.resize(1420, 840)

        self.worker: TriangleTrainingWorker | None = None
        self.run_index = 0
        self.history: list[RunArchive] = []
        self.current_steps: list[int] = []
        self.current_losses: list[float] = []
        self.validation_steps: list[int] = []
        self.validation_losses: list[float] = []
        self.demo_steps: list[int] = []
        self.demo_losses: list[float] = []
        self.current_run_name = ""
        self._wheel_guard = WheelGuard(self)
        self.demo_case_name = args.demo_case if args.demo_case in _DEMO_CASES else _DEFAULT_DEMO_CASE
        self._gt_bary, self._gt_point, self._gt_sdf, self._gt_grad = self._compute_demo_gt()

        self._build_ui(args)
        self._apply_style()
        self._reset_dashboard()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(2000)
        super().closeEvent(event)

    def _build_ui(self, args: argparse.Namespace) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        left = QtWidgets.QWidget()
        left.setMinimumWidth(292)
        left.setMaximumWidth(340)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        title = QtWidgets.QLabel("Triangle Trainer")
        title.setObjectName("title")
        left_layout.addWidget(title)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 3, 0)
        content_layout.setSpacing(5)
        content_layout.addWidget(self._build_settings(args))
        content_layout.addWidget(self._build_run_panel(args))
        content_layout.addStretch(1)
        scroll.setWidget(content)
        left_layout.addWidget(scroll, stretch=1)

        main = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)
        main_layout.addWidget(self._build_dashboard())
        main_layout.addWidget(self._build_training_view(), stretch=1)

        layout.addWidget(left)
        layout.addWidget(main, stretch=1)

    def _build_settings(self, args: argparse.Namespace) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(5)

        self.triangle_layers_edit = QtWidgets.QLineEdit(args.triangle_layers)
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(args.device or "auto")
        for widget in (
            self.triangle_layers_edit, self.device_combo,
        ):
            self._disable_wheel_edit(widget)

        model_form = self._form_layout()
        model_form.addRow("Triangle layers", self.triangle_layers_edit)
        model_form.addRow("Device", self.device_combo)
        panel_layout.addWidget(self._section("Architecture", model_form, True))

        self.steps_spin = self._spin(args.steps, 1, 10_000_000, 1000)
        self.batch_spin = self._spin(args.batch, 1, 262_144, 128)
        self.lr_edit = self._float_edit(args.lr)
        self.eta_min_edit = self._float_edit(args.eta_min)
        self.log_spin = self._spin(args.log_every, 1, 100_000, 1)
        self.viz_spin = self._spin(args.viz_every, 1, 100_000, 1)
        train_form = self._form_layout()
        train_form.addRow("Steps", self.steps_spin)
        train_form.addRow("Batch", self.batch_spin)
        train_form.addRow("LR", self.lr_edit)
        train_form.addRow("Eta min", self.eta_min_edit)
        train_form.addRow("Log every", self.log_spin)
        train_form.addRow("Viz every", self.viz_spin)
        panel_layout.addWidget(self._section("Optimizer and Logging", train_form, True))

        self.grid_spin = self._spin(args.grid_g, 1, 256, 1)
        self.gt_refine_steps_spin = self._spin(args.gt_refine_steps, 0, 12, 1)
        self.gt_refine_grid_spin = self._spin(args.gt_refine_grid, 3, 31, 2)
        self.precompute_samples_spin = self._spin(args.precompute_samples, 0, 10_000_000, 10_000)
        self.precompute_chunk_spin = self._spin(args.precompute_chunk, 1, 262_144, 512)
        self.validation_samples_spin = self._spin(args.validation_samples, 0, 100_000, 128)
        self.validation_seed_spin = self._spin(args.validation_seed, 0, 2_147_483_647, 1)
        self.demo_case_combo = QtWidgets.QComboBox()
        self.demo_case_combo.addItems(list(_DEMO_CASES.keys()))
        self.demo_case_combo.setCurrentText(self.demo_case_name)
        self._disable_wheel_edit(self.demo_case_combo)
        self.demo_case_combo.currentTextChanged.connect(self._select_demo_case)
        self.r_min_edit = self._float_edit(args.r_min)
        self.r_max_edit = self._float_edit(args.r_max)
        gt_form = self._form_layout()
        gt_form.addRow("Grid G", self.grid_spin)
        gt_form.addRow("GT refine steps", self.gt_refine_steps_spin)
        gt_form.addRow("GT refine grid", self.gt_refine_grid_spin)
        gt_form.addRow("Precompute samples", self.precompute_samples_spin)
        gt_form.addRow("Precompute chunk", self.precompute_chunk_spin)
        gt_form.addRow("Validation samples", self.validation_samples_spin)
        gt_form.addRow("Validation seed", self.validation_seed_spin)
        gt_form.addRow("Demo case", self.demo_case_combo)
        gt_form.addRow("R min", self.r_min_edit)
        gt_form.addRow("R max", self.r_max_edit)
        panel_layout.addWidget(self._section("Ground Truth and Cache", gt_form, True))

        self.loss_bary_edit = self._float_edit(args.loss_bary_weight)
        self.loss_point_edit = self._float_edit(args.loss_point_weight)
        self.loss_sdf_edit = self._float_edit(args.loss_sdf_weight)
        loss_form = self._form_layout()
        loss_form.addRow("Bary", self.loss_bary_edit)
        loss_form.addRow("Spatial point", self.loss_point_edit)
        loss_form.addRow("SDF", self.loss_sdf_edit)
        panel_layout.addWidget(self._section("Loss Weights", loss_form, False))

        self.early_patience_spin = self._spin(args.early_stopping_patience, 0, 10_000_000, 100)
        self.early_warmup_spin = self._spin(args.early_stopping_warmup, 0, 10_000_000, 100)
        self.early_smooth_spin = self._spin(args.early_stopping_smooth, 1, 1_000_000, 10)
        self.early_delta_edit = self._float_edit(args.early_stopping_min_delta)
        early_form = self._form_layout()
        early_form.addRow("Patience", self.early_patience_spin)
        early_form.addRow("Warmup", self.early_warmup_spin)
        early_form.addRow("Smooth", self.early_smooth_spin)
        early_form.addRow("Min delta", self.early_delta_edit)
        panel_layout.addWidget(self._section("Early Stopping", early_form, False))

        self.triangle_load_edit = QtWidgets.QLineEdit(args.triangle_load)
        self.run_checkpoint_combo = QtWidgets.QComboBox()
        for widget in (self.triangle_load_edit, self.run_checkpoint_combo):
            self._disable_wheel_edit(widget)
        self.refresh_runs_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_runs_btn.clicked.connect(self._refresh_run_checkpoints)
        self.use_triangle_btn = QtWidgets.QPushButton("Use")
        self.use_triangle_btn.clicked.connect(lambda: self._apply_selected_checkpoint("triangle"))

        load_layout = QtWidgets.QVBoxLayout()
        load_layout.setContentsMargins(4, 3, 4, 5)
        load_layout.setSpacing(4)
        load_layout.addWidget(self.run_checkpoint_combo)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.refresh_runs_btn)
        row.addWidget(self.use_triangle_btn)
        load_layout.addLayout(row)
        load_form = self._form_layout()
        load_form.addRow("Triangle ckpt", self.triangle_load_edit)
        load_layout.addLayout(load_form)
        panel_layout.addWidget(self._section("Checkpoints", load_layout, False))
        self._refresh_run_checkpoints()
        return panel

    def _build_run_panel(self, args: argparse.Namespace) -> QtWidgets.QWidget:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 3, 4, 5)
        layout.setSpacing(4)
        self.save_edit = QtWidgets.QLineEdit(args.save)
        self._disable_wheel_edit(self.save_edit)
        layout.addWidget(QtWidgets.QLabel("Save path"))
        layout.addWidget(self.save_edit)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_run)
        self.stop_btn.clicked.connect(self._stop_run)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        layout.addLayout(row)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return self._section("Run Control", layout, True)

    def _build_dashboard(self) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        self.cards: dict[str, QtWidgets.QLabel] = {}
        specs = [
            ("state", "State"), ("phase", "Phase"), ("step", "Step"), ("loss", "Loss"),
            ("val", "Val"), ("best", "Best"), ("lr", "LR"), ("speed", "Speed"),
            ("elapsed", "Elapsed"), ("eta", "ETA"), ("device", "Device"), ("cache", "GT cache"),
            ("saved", "Saved"),
        ]
        for i, (key, label) in enumerate(specs):
            grid.addWidget(self._card(key, label), i // 4, i % 4)
        return wrap

    def _build_training_view(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_plot_area())
        splitter.addWidget(self._build_demo_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 520])
        return splitter

    def _build_plot_area(self) -> QtWidgets.QWidget:
        plot_wrap = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_wrap)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)

        history_bar = QtWidgets.QHBoxLayout()
        history_bar.setContentsMargins(0, 0, 0, 0)
        history_bar.setSpacing(6)
        self.history_button = QtWidgets.QToolButton()
        self.history_button.setText("Past runs")
        self.history_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.history_button.setObjectName("compactTool")
        self.history_menu = QtWidgets.QMenu(self.history_button)
        self.history_button.setMenu(self.history_menu)
        self.history_count_label = QtWidgets.QLabel("0")
        self.history_count_label.setObjectName("subtleLabel")
        clear_btn = QtWidgets.QToolButton()
        clear_btn.setText("Clear")
        clear_btn.setObjectName("compactTool")
        clear_btn.clicked.connect(self._clear_history)
        history_bar.addWidget(self.history_button)
        history_bar.addWidget(self.history_count_label)
        history_bar.addStretch(1)
        history_bar.addWidget(clear_btn)
        plot_layout.addLayout(history_bar)

        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setLabel("left", "Loss")
        self.loss_plot.setLabel("bottom", "Step")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.25)
        self.loss_plot.setLogMode(False, True)
        self.loss_plot.addLegend(offset=(8, 8))
        self.current_curve = self.loss_plot.plot(
            [], [], pen=pg.mkPen("#f59e0b", width=2), name="train loss",
        )
        self.validation_curve = self.loss_plot.plot(
            [], [], pen=pg.mkPen("#16a34a", width=2), name="validation mean",
        )
        self.demo_curve = self.loss_plot.plot(
            [], [], pen=pg.mkPen("#2563eb", width=2, style=QtCore.Qt.DashLine),
            name="selected demo",
        )
        plot_layout.addWidget(self.loss_plot)
        return plot_wrap

    def _build_demo_panel(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Demo triangle")
        box.setMinimumWidth(420)
        layout = QtWidgets.QVBoxLayout(box)
        self.demo_plot = pg.PlotWidget()
        self.demo_plot.setAspectLocked(True)
        self.demo_plot.showGrid(x=True, y=True, alpha=0.2)
        self.demo_plot.setXRange(-0.9, 0.9)
        self.demo_plot.setYRange(-0.9, 0.9)
        self._init_demo_plot()
        layout.addWidget(self.demo_plot, stretch=1)

        metrics = QtWidgets.QGridLayout()
        metrics.setContentsMargins(0, 2, 0, 0)
        metrics.setSpacing(6)
        self.demo_metrics: dict[str, QtWidgets.QLabel] = {}
        metric_specs = [
            ("demo_loss", "Val demo"), ("point_error", "Point err"),
            ("sdf_error", "SDF err"), ("grad_align", "Grad align"),
            ("pred_bary", "Pred bary"), ("gt_bary", "GT bary"),
        ]
        for i, (key, caption) in enumerate(metric_specs):
            metrics.addWidget(self._demo_metric(key, caption), i // 3, i % 3)
        layout.addLayout(metrics)
        return box

    def _init_demo_plot(self) -> None:
        radii, tri3 = _demo_case_arrays(self.demo_case_name)
        theta = np.linspace(0.0, 2.0 * np.pi, 220)
        self.demo_ellipse_item = self.demo_plot.plot(
            radii[0] * np.cos(theta),
            radii[1] * np.sin(theta),
            pen=pg.mkPen("#22c55e", width=2, style=QtCore.Qt.DashLine),
        )
        tri = np.vstack([tri3[:, :2], tri3[0, :2]])
        self.demo_triangle_item = self.demo_plot.plot(tri[:, 0], tri[:, 1], pen=pg.mkPen("#94a3b8", width=2))
        self.gt_point_item = self.demo_plot.plot(
            [self._gt_point[0]], [self._gt_point[1]],
            pen=None, symbol="star", symbolBrush="#22c55e", symbolPen="#dcfce7", symbolSize=18,
        )
        self.pred_point_item = self.demo_plot.plot(
            [], [], pen=None, symbol="o", symbolBrush="#f97316", symbolPen="#ffedd5", symbolSize=13,
        )
        self.demo_error_line = self.demo_plot.plot([], [], pen=pg.mkPen("#ef4444", width=1.8))
        self.gt_grad_line = self.demo_plot.plot([], [], pen=pg.mkPen("#22c55e", width=2))
        self.pred_grad_line = self.demo_plot.plot([], [], pen=pg.mkPen("#f97316", width=2))

    def _compute_demo_gt(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        radii, tri = _demo_case_arrays(self.demo_case_name)
        r = torch.from_numpy(radii[None]).float()
        v = torch.from_numpy(tri[None]).float()
        bary = triangle_deepest_refined_torch(
            r, v[:, 0], v[:, 1], v[:, 2],
            G=60, refine_steps=6, refine_grid=9,
        )
        point = bary_to_point(bary, v[:, 0], v[:, 1], v[:, 2])
        with torch.no_grad():
            sdf = sdf_exact_torch(point, r)
            grad = _normalize_gradient(point, r, sdf)
        return bary[0].numpy(), point[0].numpy(), float(sdf[0]), grad[0].numpy()

    def _select_demo_case(self, name: str) -> None:
        if name not in _DEMO_CASES:
            return
        self.demo_case_name = name
        self._gt_bary, self._gt_point, self._gt_sdf, self._gt_grad = self._compute_demo_gt()
        radii, tri3 = _demo_case_arrays(name)
        theta = np.linspace(0.0, 2.0 * np.pi, 220)
        self.demo_ellipse_item.setData(radii[0] * np.cos(theta), radii[1] * np.sin(theta))
        tri = np.vstack([tri3[:, :2], tri3[0, :2]])
        self.demo_triangle_item.setData(tri[:, 0], tri[:, 1])
        self.gt_point_item.setData([self._gt_point[0]], [self._gt_point[1]])
        self.pred_point_item.setData([], [])
        self.demo_error_line.setData([], [])
        self.gt_grad_line.setData([], [])
        self.pred_grad_line.setData([], [])
        for label in getattr(self, "demo_metrics", {}).values():
            label.setText("-")

    def _params(self) -> dict:
        device = self.device_combo.currentText()
        return {
            "triangle_layers": _parse_layers(self.triangle_layers_edit.text(), [512, 512, 256, 256]),
            "steps": self.steps_spin.value(),
            "batch_size": self.batch_spin.value(),
            "grid_G": self.grid_spin.value(),
            "gt_refine_steps": self.gt_refine_steps_spin.value(),
            "gt_refine_grid": self.gt_refine_grid_spin.value(),
            "precompute_samples": self.precompute_samples_spin.value(),
            "precompute_chunk": self.precompute_chunk_spin.value(),
            "validation_samples": self.validation_samples_spin.value(),
            "validation_seed": self.validation_seed_spin.value(),
            "demo_case": self.demo_case_combo.currentText(),
            "lr": _float_from_line(self.lr_edit, 1e-3),
            "eta_min": _float_from_line(self.eta_min_edit, 1e-5),
            "r_min": _float_from_line(self.r_min_edit, 0.05),
            "r_max": _float_from_line(self.r_max_edit, 1.0),
            "log_every": self.log_spin.value(),
            "viz_every": self.viz_spin.value(),
            "device": None if device == "auto" else device,
            "save_path": Path(self.save_edit.text()).expanduser() if self.save_edit.text().strip() else None,
            "early_patience": self.early_patience_spin.value(),
            "early_min_delta": _float_from_line(self.early_delta_edit, 0.0),
            "early_warmup": self.early_warmup_spin.value(),
            "early_smooth": self.early_smooth_spin.value(),
            "loss_bary_weight": _float_from_line(self.loss_bary_edit, 1.0),
            "loss_point_weight": _float_from_line(self.loss_point_edit, 0.25),
            "loss_sdf_weight": _float_from_line(self.loss_sdf_edit, 0.0),
            "triangle_load": self.triangle_load_edit.text().strip(),
        }

    def _refresh_run_checkpoints(self) -> None:
        self.run_checkpoint_combo.clear()
        records = _load_run_records()
        for record in records:
            self.run_checkpoint_combo.addItem(_run_record_label(record), record)
        if not records:
            self.run_checkpoint_combo.addItem("No saved runs yet", None)

    def _apply_selected_checkpoint(self, target: str) -> None:
        record = self.run_checkpoint_combo.currentData()
        if not record:
            return
        checkpoint = str(record.get("checkpoint", ""))
        if not checkpoint:
            return
        self.triangle_load_edit.setText(checkpoint)

    def _start_run(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        self._archive_current_run()
        self.run_index += 1
        self.current_run_name = f"Run {self.run_index}"
        self.current_steps = []
        self.current_losses = []
        self.validation_steps = []
        self.validation_losses = []
        self.demo_steps = []
        self.demo_losses = []
        self.current_curve.setData([], [])
        self.validation_curve.setData([], [])
        self.demo_curve.setData([], [])
        self.demo_error_line.setData([], [])
        self._reset_dashboard()
        self.cards["state"].setText("Starting")
        self.status_label.setText("Starting training thread...")

        self.worker = TriangleTrainingWorker(self._params(), self)
        self.worker.progress.connect(self._on_progress)
        self.worker.validation.connect(self._on_validation)
        self.worker.demo.connect(self._on_demo)
        self.worker.finished_info.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_parameter_enabled(False)

    def _stop_run(self) -> None:
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Stopping after current batch...")
            self.cards["state"].setText("Stopping")
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)

    def _on_progress(self, data: dict) -> None:
        step = int(data["step"])
        loss = data["loss"]
        if loss is not None:
            self.current_steps.append(step)
            self.current_losses.append(float(loss))
            self.current_curve.setData(self.current_steps, self.current_losses)

        total = int(data["total"])
        self.cards["state"].setText(str(data["state"]).title())
        self.cards["phase"].setText(str(data.get("phase", "-")))
        self.cards["step"].setText(f"{step} / {total} ({step / max(1, total) * 100:.1f}%)")
        self.cards["loss"].setText(_fmt_loss(loss))
        self.cards["best"].setText(_fmt_loss(data["best_loss"]))
        self.cards["lr"].setText(f"{float(data['lr']):.2e}")
        self.cards["speed"].setText(f"{float(data['speed']):.2f}/s" if data["speed"] else "-")
        self.cards["elapsed"].setText(_fmt_time(data["elapsed"]))
        self.cards["eta"].setText(_fmt_time(data["eta"]))
        self.cards["device"].setText(str(data["device"]))
        self.cards["cache"].setText(str(data.get("cache", "-")))
        parts = data.get("loss_parts", {}) or {}
        suffix = ""
        if parts:
            suffix = "  [" + ", ".join(
                f"{k}={_fmt_loss(v)}" for k, v in parts.items() if k != "total"
            ) + "]"
        self.status_label.setText(
            f"{self.current_run_name}: step {step:,}/{total:,}, loss {_fmt_loss(loss)}{suffix}"
        )

    def _on_validation(self, data: dict) -> None:
        loss = data.get("loss")
        if loss is None:
            return
        step = int(data.get("step", 0))
        val = float(loss)
        self.validation_steps.append(step)
        self.validation_losses.append(max(val, 1e-12))
        self.validation_curve.setData(self.validation_steps, self.validation_losses)
        self.cards["val"].setText(_fmt_loss(val))

    def _on_demo(self, data: dict) -> None:
        step = int(data.get("step", 0))
        point = np.asarray(data["point"], dtype=np.float32)
        self.pred_point_item.setData([point[0]], [point[1]])

        gt_point = np.asarray(data.get("gt_point", self._gt_point), dtype=np.float32)
        self.demo_error_line.setData([gt_point[0], point[0]], [gt_point[1], point[1]])

        grad = np.asarray(data["grad"], dtype=np.float32)
        target_grad = np.asarray(data.get("target_grad", self._gt_grad), dtype=np.float32)
        target_grad_point = np.asarray(data.get("target_grad_point", self._gt_point), dtype=np.float32)
        s = 0.18
        self.pred_grad_line.setData([point[0], point[0] + s * grad[0]],
                                    [point[1], point[1] + s * grad[1]])
        self.gt_grad_line.setData(
            [target_grad_point[0], target_grad_point[0] + s * target_grad[0]],
            [target_grad_point[1], target_grad_point[1] + s * target_grad[1]],
        )
        demo_loss = float(data.get("demo_loss", np.nan))
        if np.isfinite(demo_loss):
            self.demo_steps.append(step)
            self.demo_losses.append(max(demo_loss, 1e-12))
            self.demo_curve.setData(self.demo_steps, self.demo_losses)

        self.demo_metrics["demo_loss"].setText(_fmt_loss(data.get("demo_loss")))
        self.demo_metrics["point_error"].setText(_fmt_loss(data.get("demo_point_error")))
        self.demo_metrics["sdf_error"].setText(_fmt_loss(data.get("demo_sdf_error")))
        grad_align = data.get("demo_grad_align")
        self.demo_metrics["grad_align"].setText("-" if grad_align is None else f"{float(grad_align):.4f}")
        self.demo_metrics["pred_bary"].setText(_fmt_bary(data.get("bary", [])))
        self.demo_metrics["gt_bary"].setText(_fmt_bary(data.get("gt_bary", self._gt_bary)))

    def _on_finished(self, data: dict) -> None:
        reason = str(data["reason"])
        labels = {"completed": "Completed", "stopped": "Stopped", "early_stop": "Early stopped"}
        self.cards["state"].setText(labels.get(reason, reason))
        self.cards["best"].setText(_fmt_loss(data["best_loss"]))
        self.cards["elapsed"].setText(_fmt_time(data["elapsed"]))
        self.cards["saved"].setText(data["save_path"] or "-")
        self._refresh_run_checkpoints()
        self.status_label.setText(
            f"{labels.get(reason, reason)}. Best loss {_fmt_loss(data['best_loss'])} "
            f"at step {data['best_step']}."
        )
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_parameter_enabled(True)

    def _on_failed(self, message: str) -> None:
        self.cards["state"].setText("Failed")
        self.status_label.setText(message)
        QtWidgets.QMessageBox.critical(self, "Training failed", message)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_parameter_enabled(True)

    def _archive_current_run(self) -> None:
        if not self.current_steps:
            return
        color = pg.intColor(len(self.history), hues=12, values=1.0, maxValue=220)
        curve = self.loss_plot.plot(
            self.current_steps, self.current_losses, pen=pg.mkPen(color, width=1.4)
        )
        action = QtGui.QAction(self.current_run_name, self.history_menu)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(lambda checked, c=curve: c.setVisible(checked))
        self.history_menu.addAction(action)
        self.history.append(RunArchive(
            self.current_run_name, list(self.current_steps), list(self.current_losses), curve, action
        ))
        self.history_count_label.setText(str(len(self.history)))

    def _clear_history(self) -> None:
        for run in self.history:
            self.loss_plot.removeItem(run.curve)
            self.history_menu.removeAction(run.action)
        self.history.clear()
        self.history_count_label.setText("0")

    def _reset_dashboard(self) -> None:
        for label in self.cards.values():
            label.setText("-")
        self.cards["state"].setText("Ready")
        for label in getattr(self, "demo_metrics", {}).values():
            label.setText("-")

    def _set_parameter_enabled(self, enabled: bool) -> None:
        for widget in (
            self.triangle_layers_edit, self.device_combo,
            self.steps_spin, self.batch_spin, self.lr_edit, self.eta_min_edit,
            self.log_spin, self.viz_spin, self.grid_spin, self.gt_refine_steps_spin,
            self.gt_refine_grid_spin, self.precompute_samples_spin, self.precompute_chunk_spin,
            self.validation_samples_spin, self.validation_seed_spin, self.demo_case_combo,
            self.r_min_edit, self.r_max_edit, self.loss_bary_edit, self.loss_point_edit,
            self.loss_sdf_edit, self.early_patience_spin, self.early_warmup_spin,
            self.early_smooth_spin, self.early_delta_edit, self.triangle_load_edit,
            self.save_edit, self.run_checkpoint_combo, self.refresh_runs_btn, self.use_triangle_btn,
        ):
            widget.setEnabled(enabled)

    def _card(self, key: str, caption: str) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(10, 7, 10, 7)
        cap = QtWidgets.QLabel(caption)
        cap.setObjectName("caption")
        value = QtWidgets.QLabel("-")
        value.setObjectName("value")
        value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(cap)
        layout.addWidget(value)
        self.cards[key] = value
        return frame

    def _demo_metric(self, key: str, caption: str) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("demoMetric")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(1)
        cap = QtWidgets.QLabel(caption)
        cap.setObjectName("caption")
        value = QtWidgets.QLabel("-")
        value.setObjectName("metricValue")
        value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        value.setMinimumWidth(92)
        layout.addWidget(cap)
        layout.addWidget(value)
        self.demo_metrics[key] = value
        return frame

    def _spin(self, value: int, minimum: int, maximum: int, step: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(int(value))
        self._disable_wheel_edit(spin)
        return spin

    def _float_edit(self, value: float) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(f"{float(value):g}")
        edit.setValidator(QtGui.QDoubleValidator(bottom=-1e30, top=1e30, decimals=12))
        self._disable_wheel_edit(edit)
        return edit

    def _disable_wheel_edit(self, widget: QtWidgets.QWidget) -> None:
        widget.installEventFilter(self._wheel_guard)

    def _form_layout(self) -> QtWidgets.QFormLayout:
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignTop)
        form.setContentsMargins(4, 3, 4, 5)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(3)
        return form

    def _section(self, title: str, layout: QtWidgets.QLayout, expanded: bool) -> CollapsibleSection:
        section = CollapsibleSection(title, expanded=expanded)
        section.set_content_layout(layout)
        return section

    def _apply_style(self) -> None:
        pg.setConfigOptions(antialias=True)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #f6f8fb;
                color: #182230;
                font-size: 11px;
            }
            QLabel#title {
                font-size: 16px;
                font-weight: 700;
                padding: 0 0 4px 0;
            }
            QScrollArea#settingsScroll {
                background: transparent;
                border: 0;
            }
            QWidget#foldout {
                background: transparent;
                border-top: 1px solid #dbe3ec;
                margin-top: 2px;
            }
            QWidget#foldoutBody {
                background: transparent;
                border-left: 1px solid #e5ebf2;
                margin-left: 7px;
            }
            QToolButton#foldoutHeader {
                background: transparent;
                border: 0;
                border-radius: 3px;
                padding: 3px 2px;
                color: #334155;
                font-weight: 700;
                text-align: left;
            }
            QToolButton#foldoutHeader:hover {
                background: #edf3fa;
                color: #182230;
            }
            QToolButton#foldoutHeader:checked {
                color: #111827;
            }
            QToolButton#compactTool {
                background: #ffffff;
                border: 1px solid #cfd8e3;
                border-radius: 4px;
                padding: 2px 6px;
                color: #334155;
                font-weight: 600;
            }
            QToolButton#compactTool:hover {
                background: #edf3fa;
            }
            QLabel#subtleLabel {
                color: #64748b;
                padding: 0 2px;
            }
            QGroupBox {
                border: 1px solid #cfd8e3;
                border-radius: 5px;
                margin-top: 6px;
                padding: 5px;
                font-weight: 600;
            }
            QLineEdit, QSpinBox, QComboBox, QListWidget {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 2px 4px;
                color: #182230;
                min-height: 19px;
            }
            QMenu {
                background: #ffffff;
                color: #182230;
                border: 1px solid #cfd8e3;
            }
            QMenu::item {
                padding: 5px 18px;
            }
            QMenu::item:selected {
                background: #edf3fa;
            }
            QPushButton {
                background: #2563eb;
                border: 0;
                border-radius: 4px;
                padding: 4px 7px;
                color: white;
                font-weight: 600;
                min-height: 20px;
            }
            QPushButton:disabled {
                background: #d8dee8;
                color: #7b8798;
            }
            QFrame#card {
                background: #ffffff;
                border: 1px solid #d8dee8;
                border-radius: 6px;
            }
            QFrame#demoMetric {
                background: #ffffff;
                border: 1px solid #d8dee8;
                border-radius: 5px;
            }
            QLabel#caption {
                color: #64748b;
                font-size: 10px;
            }
            QLabel#value {
                color: #111827;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#metricValue {
                color: #111827;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#status {
                color: #334155;
                font-family: Consolas, monospace;
            }
        """)
        for plot in (self.loss_plot, self.demo_plot):
            plot.setBackground("#ffffff")
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setPen("#64748b")
                axis.setTextPen("#334155")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TriangleDeepestNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=16_384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--triangle-layers", type=str, default="512,512,256,256")
    parser.add_argument("--grid-g", type=int, default=24)
    parser.add_argument("--gt-refine-steps", type=int, default=3)
    parser.add_argument("--gt-refine-grid", type=int, default=7)
    parser.add_argument("--precompute-samples", type=int, default=100_000)
    parser.add_argument("--precompute-chunk", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--validation-seed", type=int, default=1729)
    parser.add_argument("--demo-case", choices=tuple(_DEMO_CASES.keys()), default=_DEFAULT_DEMO_CASE)
    parser.add_argument("--r-min", type=float, default=0.05)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save", type=str, default="triangle_deepest.pt")
    parser.add_argument("--triangle-load", type=str, default="")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--viz-every", type=int, default=100)
    parser.add_argument("--loss-bary-weight", type=float, default=1.0)
    parser.add_argument("--loss-point-weight", type=float, default=0.25)
    parser.add_argument("--loss-sdf-weight", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2000)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument("--early-stopping-warmup", type=int, default=5000)
    parser.add_argument("--early-stopping-smooth", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = TriangleTrainingWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
