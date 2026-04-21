"""
train_neural_triangle.py -- Live-Training-UI fuer EllipsoidTriangleNet

Aufruf:
    python train_neural_triangle.py
    python train_neural_triangle.py --steps 100000 --batch 2048 --save tri.pt

Tasten im Fenster:
    Q / Fenster schliessen  -> Training abbrechen und speichern
"""

from __future__ import annotations

import argparse
import queue
import threading
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import torch
import torch.nn as nn

from neural_sdf import sdf_exact_torch
from neural_triangle import (EllipsoidTriangleNet, EllipsoidTriangleTrainer,
                              triangle_deepest_exact_torch)


# ── Demo-Konfiguration ────────────────────────────────────────────────────────

_VIZ_RADII  = np.array([0.5, 0.4, 0.15], dtype=np.float32)

# Dreieck in z=0-Ebene: schneidet die Ellipse, Schwerpunkt liegt innen
_VIZ_TRI = np.array([
    [-0.30, -0.55,  0.0],   # p1 — ausserhalb
    [ 0.65,  0.05,  0.0],   # p2 — ausserhalb
    [-0.05,  0.65,  0.0],   # p3 — ausserhalb
], dtype=np.float32)

_VIZ_EXTENT = 0.85
_VIZ_N      = 96     # SDF-Heatmap-Aufloesung


def _build_sdf_grid(n: int = _VIZ_N, extent: float = _VIZ_EXTENT):
    lin = np.linspace(-extent, extent, n, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    pts = np.stack([xx.ravel(), yy.ravel(),
                    np.zeros(n * n, dtype=np.float32)], axis=-1)
    return pts, (n, n), lin


# ── Haupt-App ─────────────────────────────────────────────────────────────────

class TrainingApp:
    """
    Startet das Training in einem Background-Thread und zeigt Loss-Kurve +
    Dreieck-Visualisierung (SDF-Heatmap, GT-Punkt, Vorhersage-Punkt) live an.
    """

    def __init__(
        self,
        n_steps:    int,
        batch_size: int,
        lr:         float,
        hidden:     int,
        depth:      int,
        device:     Optional[str],
        save_path:  str,
        log_every:  int,
        viz_every:  int,
        grid_G:     int,
    ):
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.lr         = lr
        self.save_path  = save_path
        self.log_every  = log_every
        self.viz_every  = viz_every
        self.grid_G     = grid_G
        self._stop      = threading.Event()

        self.trainer = EllipsoidTriangleTrainer(
            hidden=hidden, depth=depth, device=device)
        self.device = self.trainer.device

        self._q: queue.Queue = queue.Queue()

        # SDF-Gitter vorberechnen (statisch)
        self._pts, self._shape, self._lin = _build_sdf_grid()
        self._sdf_gt = self._compute_gt_sdf()   # Hintergrund-Heatmap

        # GT-Tiefpunkt vorberechnen (statisch, hohe Aufloesung)
        self._gt_bary, self._gt_point = self._compute_gt_deepest(G=50)

        self._setup_figure()

    # ── Vorberechnungen ───────────────────────────────────────────────────

    def _compute_gt_sdf(self) -> np.ndarray:
        """Exakter SDF des Demo-Ellipsoids auf dem z=0-Gitter."""
        r = torch.from_numpy(_VIZ_RADII).double()
        p = torch.from_numpy(self._pts).double()
        r2 = r ** 2
        inside = ((p / r) ** 2).sum(dim=-1) < 1.0
        eps = 1e-10 * r.min()
        p_abs = p.abs().clamp(min=eps)
        T_max = r.max() * p_abs.norm(dim=-1) + r2.sum()
        T_min = -r2.min() * (1.0 - 1e-15)
        t_lo = torch.where(inside, T_min.expand_as(inside), torch.zeros_like(T_min.expand_as(inside)))
        t_hi = torch.where(inside, torch.zeros_like(T_max), T_max)
        for _ in range(80):
            t_m = 0.5 * (t_lo + t_hi)
            F = ((r * p_abs / (r2 + t_m.unsqueeze(-1))) ** 2).sum(dim=-1) - 1.0
            t_lo = torch.where(F > 0, t_m, t_lo)
            t_hi = torch.where(F <= 0, t_m, t_hi)
        t = 0.5 * (t_lo + t_hi)
        q = r2 * p_abs / (r2 + t.unsqueeze(-1))
        d = (p_abs - q).norm(dim=-1)
        sdf = torch.where(inside, -d, d).float().numpy()
        return sdf.reshape(self._shape)

    def _compute_gt_deepest(self, G: int = 50):
        """GT-Tiefpunkt des Demo-Dreiecks im Demo-Ellipsoid."""
        r_t = torch.from_numpy(_VIZ_RADII[None]).float()
        v   = torch.from_numpy(_VIZ_TRI[None]).float()     # (1, 3, 3)
        bary = triangle_deepest_exact_torch(
            r_t, v[:, 0], v[:, 1], v[:, 2], G=G
        ).squeeze(0).numpy()                               # (3,)
        point = (bary[0] * _VIZ_TRI[0]
               + bary[1] * _VIZ_TRI[1]
               + bary[2] * _VIZ_TRI[2])
        return bary, point

    # ── Matplotlib-Fenster aufbauen ───────────────────────────────────────

    def _setup_figure(self):
        BG     = "#111827"
        PANEL  = "#1f2937"
        ACCENT = "#f97316"
        TEXT   = "#e5e7eb"
        SUBTLE = "#6b7280"

        plt.ion()
        self.fig = plt.figure(figsize=(14, 7.5), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "Ellipsoid Triangle -- Training")

        gs = gridspec.GridSpec(
            2, 2,
            figure=self.fig,
            width_ratios=[1, 2.2],
            height_ratios=[1, 5],
            hspace=0.10,
            wspace=0.30,
        )

        # ── Info-Zeile ────────────────────────────────────────────────────
        self.ax_info = self.fig.add_subplot(gs[0, :])
        self.ax_info.set_facecolor(BG)
        self.ax_info.axis("off")
        self._info_text = self.ax_info.text(
            0.0, 0.5,
            self._info_str(0, float("nan")),
            transform=self.ax_info.transAxes,
            va="center", ha="left",
            color=TEXT, fontsize=11, fontfamily="monospace",
        )
        self._big_loss = self.ax_info.text(
            1.0, 0.5, "L1 = ---",
            transform=self.ax_info.transAxes,
            va="center", ha="right",
            color=ACCENT, fontsize=22, fontweight="bold",
            fontfamily="monospace",
        )

        # ── Loss-Kurve ────────────────────────────────────────────────────
        self.ax_loss = self.fig.add_subplot(gs[1, 0])
        self.ax_loss.set_facecolor(PANEL)
        self.ax_loss.set_xlabel("Schritt",    color=SUBTLE, fontsize=9)
        self.ax_loss.set_ylabel("L1-Verlust", color=SUBTLE, fontsize=9)
        self.ax_loss.tick_params(colors=SUBTLE, labelsize=8)
        for sp in self.ax_loss.spines.values():
            sp.set_edgecolor("#374151")
        self.ax_loss.grid(color="#374151", lw=0.5, alpha=0.6)

        self._loss_line,   = self.ax_loss.plot(
            [], [], color=ACCENT, lw=1.5, alpha=0.9)
        self._loss_smooth, = self.ax_loss.plot(
            [], [], color="#fcd34d", lw=1.0, alpha=0.6, label="geglaettet")
        self.ax_loss.legend(fontsize=8, labelcolor=TEXT,
                            facecolor=PANEL, edgecolor="none")
        self._steps_hist: list[int]   = []
        self._loss_hist:  list[float] = []

        # ── Szenen-Panel ──────────────────────────────────────────────────
        self.ax_v = self.fig.add_subplot(gs[1, 1])
        self.ax_v.set_facecolor(PANEL)
        self.ax_v.set_aspect("equal")
        self.ax_v.set_title("Tiefster Punkt  (z=0-Schnitt)",
                             color=TEXT, fontsize=9, pad=4)
        self.ax_v.tick_params(colors=SUBTLE, labelsize=7)
        for sp in self.ax_v.spines.values():
            sp.set_edgecolor("#374151")

        # SDF-Heatmap (statisch)
        vmax = float(np.abs(self._sdf_gt).max())
        self.ax_v.imshow(
            self._sdf_gt, origin="lower",
            extent=[-_VIZ_EXTENT, _VIZ_EXTENT,
                    -_VIZ_EXTENT, _VIZ_EXTENT],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            interpolation="bilinear", zorder=1,
        )

        # Ellipsen-Kontur
        self.ax_v.contour(
            self._lin, self._lin, self._sdf_gt,
            levels=[0.0], colors=["#4ade80"],
            linewidths=1.2, linestyles="--", zorder=2,
        )

        # Demo-Dreieck
        tri_xy = _VIZ_TRI[:, :2]
        poly = MplPolygon(
            tri_xy, closed=True,
            facecolor="#ffffff18", edgecolor="#94a3b8",
            linewidth=1.5, zorder=3,
        )
        self.ax_v.add_patch(poly)
        # Eckpunkt-Beschriftungen
        for i, (label, pt) in enumerate(
                zip(["p1", "p2", "p3"], _VIZ_TRI)):
            self.ax_v.text(
                pt[0], pt[1] + 0.04, label,
                color="#94a3b8", fontsize=8, ha="center", zorder=4,
            )

        # GT-Tiefpunkt (gruen, Stern)
        self._gt_scatter = self.ax_v.scatter(
            self._gt_point[0], self._gt_point[1],
            s=120, c="#4ade80", marker="*",
            zorder=6, label="GT",
        )

        # Vorhersage-Punkt (orange, Kreis) — wird aktualisiert
        self._pred_scatter = self.ax_v.scatter(
            [], [],
            s=80, c="#f97316", marker="o",
            zorder=7, label="Vorhersage",
        )

        # Legende
        self.ax_v.legend(
            fontsize=8, labelcolor=TEXT,
            facecolor=PANEL, edgecolor="none",
            loc="lower right",
        )

        # Baryzentrische-Koordinaten-Text
        gt_u, gt_v, gt_w = self._gt_bary
        self._bary_text = self.ax_v.text(
            0.02, 0.98,
            self._bary_str("GT", gt_u, gt_v, gt_w,
                           self._sdf_at(self._gt_point)),
            transform=self.ax_v.transAxes,
            va="top", ha="left",
            color=TEXT, fontsize=7.5, fontfamily="monospace",
            zorder=8,
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event",     self._on_close)
        self.fig.subplots_adjust(left=0.07, right=0.97,
                                  top=0.95, bottom=0.10,
                                  hspace=0.10, wspace=0.35)
        self.fig.canvas.draw()
        plt.pause(0.05)

    # ── Hilfsfunktionen ───────────────────────────────────────────────────

    def _info_str(self, step: int, loss: float) -> str:
        return (
            f"Geraet: {str(self.device).upper()}   "
            f"Batch: {self.batch_size:,}   "
            f"Schritte: {step:,} / {self.n_steps:,}"
        )

    def _bary_str(self, label: str, u: float, v: float, w: float,
                  sdf_val: float) -> str:
        return (
            f"{label}\n"
            f"  u={u:.3f}  v={v:.3f}  w={w:.3f}\n"
            f"  SDF = {sdf_val:.4f}"
        )

    def _sdf_at(self, point: np.ndarray) -> float:
        """Exakter SDF des Demo-Ellipsoids an einem Punkt."""
        p_t = torch.from_numpy(point[None].astype(np.float32))
        r_t = torch.from_numpy(_VIZ_RADII[None])
        with torch.no_grad():
            return float(sdf_exact_torch(p_t, r_t)[0])

    @staticmethod
    def _smooth(values: list[float], w: int = 20) -> list[float]:
        if len(values) < w:
            return values
        return np.convolve(values, np.ones(w) / w, mode="valid").tolist()

    # ── Tastatur / Fenster ────────────────────────────────────────────────

    def _on_key(self, event):
        if event.key in ("q", "Q", "escape"):
            print("\nAbbruch -- speichere und beende...")
            self._stop.set()

    def _on_close(self, event):
        self._stop.set()

    # ── Training (Background-Thread) ──────────────────────────────────────

    def _train_loop(self):
        net   = self.trainer.net
        opt   = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_steps, eta_min=1e-5)

        net.train()
        for step in range(1, self.n_steps + 1):
            if self._stop.is_set():
                break

            inp, target = self.trainer._generate_batch(
                self.batch_size, 0.05, 1.0, self.grid_G)
            pred = net(inp)
            loss = nn.functional.l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            if step % self.log_every == 0 or step == 1:
                self._q.put(("loss", step, loss.item()))

            if step % self.viz_every == 0:
                # Demo-Dreieck auswerten
                net.eval()
                bary = net.predict_np(
                    _VIZ_RADII,
                    _VIZ_TRI[0:1], _VIZ_TRI[1:2], _VIZ_TRI[2:3],
                    device=self.device,
                )[0]
                net.train()
                point = (bary[0] * _VIZ_TRI[0]
                       + bary[1] * _VIZ_TRI[1]
                       + bary[2] * _VIZ_TRI[2])
                self._q.put(("viz", step, (bary, point)))

        net.eval()
        self._q.put(("done", None, None))

    # ── UI-Update (Main-Thread) ───────────────────────────────────────────

    def _drain_queue(self) -> bool:
        changed_loss = changed_viz = done = False

        while True:
            try:
                kind, step, data = self._q.get_nowait()
            except queue.Empty:
                break

            if kind == "loss":
                self._steps_hist.append(step)
                self._loss_hist.append(data)
                changed_loss = True
            elif kind == "viz":
                self._last_bary, self._last_point = data
                self._last_viz_step = step
                changed_viz = True
            elif kind == "done":
                done = True

        if changed_loss:
            xs, ys = self._steps_hist, self._loss_hist
            self._loss_line.set_data(xs, ys)
            sm    = self._smooth(ys)
            xs_sm = xs[len(xs) - len(sm):]
            self._loss_smooth.set_data(xs_sm, sm)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            self._info_text.set_text(self._info_str(xs[-1], ys[-1]))
            self._big_loss.set_text(f"L1 = {ys[-1]:.5f}")

        if changed_viz:
            u, v, w = self._last_bary
            pt      = self._last_point
            sdf_val = self._sdf_at(pt)

            self._pred_scatter.set_offsets([[pt[0], pt[1]]])

            gt_u, gt_v, gt_w = self._gt_bary
            self._bary_text.set_text(
                self._bary_str("GT", gt_u, gt_v, gt_w,
                               self._sdf_at(self._gt_point))
                + "\n\n"
                + self._bary_str(
                    f"Vorhersage  (Schritt {self._last_viz_step:,})",
                    u, v, w, sdf_val)
            )

            self.ax_v.set_title(
                f"Tiefster Punkt  (z=0,  Schritt {self._last_viz_step:,})",
                color="#e5e7eb", fontsize=9, pad=4,
            )

        if changed_loss or changed_viz:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        return done

    # ── Hauptschleife ─────────────────────────────────────────────────────

    def run(self):
        print("=" * 58)
        print("  Ellipsoid Triangle -- Neural Network Training")
        print("=" * 58)
        print(f"  Schritte : {self.n_steps:,}")
        print(f"  Batch    : {self.batch_size:,}")
        print(f"  Grid G   : {self.grid_G}  "
              f"({(self.grid_G+1)*(self.grid_G+2)//2} Punkte/Dreieck)")
        print(f"  Geraet   : {self.device}")
        print(f"  Speichern: {self.save_path or '(nicht gespeichert)'}")
        gt_u, gt_v, gt_w = self._gt_bary
        print(f"  GT-Bary  : u={gt_u:.4f}  v={gt_v:.4f}  w={gt_w:.4f}")
        print(f"  GT-SDF   : {self._sdf_at(self._gt_point):.5f}")
        print("  Q / Fenster schliessen = Abbrechen")
        print("=" * 58)

        self._last_bary     = np.array([1/3, 1/3, 1/3])
        self._last_point    = _VIZ_TRI.mean(axis=0)
        self._last_viz_step = 0

        thread = threading.Thread(target=self._train_loop, daemon=True)
        thread.start()

        while thread.is_alive():
            done = self._drain_queue()
            plt.pause(0.05)
            if done or self._stop.is_set():
                break

        thread.join(timeout=2.0)
        self._drain_queue()
        plt.pause(0.1)

        if self.save_path:
            self.trainer.save(self.save_path)
            print(f"\nModell gespeichert: {self.save_path}")

        print("\nTraining beendet -- Fenster schliessen zum Beenden.")
        plt.ioff()
        plt.show(block=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trainiere EllipsoidTriangleNet mit Live-Visualisierung",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps",     type=int,   default=50_000)
    parser.add_argument("--batch",     type=int,   default=4_096)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--depth",     type=int,   default=4)
    parser.add_argument("--grid-g",    type=int,   default=24)
    parser.add_argument("--device",    type=str,   default=None)
    parser.add_argument("--save",      type=str,   default="ellipsoid_triangle.pt")
    parser.add_argument("--log-every", type=int,   default=10)
    parser.add_argument("--viz-every", type=int,   default=100)
    args = parser.parse_args()

    app = TrainingApp(
        n_steps    = args.steps,
        batch_size = args.batch,
        lr         = args.lr,
        hidden     = args.hidden,
        depth      = args.depth,
        device     = args.device,
        save_path  = args.save,
        log_every  = args.log_every,
        viz_every  = args.viz_every,
        grid_G     = args.grid_g,
    )
    app.run()


if __name__ == "__main__":
    main()
