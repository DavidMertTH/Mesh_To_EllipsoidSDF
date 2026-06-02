"""
train_neural_gradient.py -- Live-Training-UI fuer EllipsoidGradientNet

Aufruf:
    python train_neural_gradient.py
    python train_neural_gradient.py --steps 100000 --batch 32768 --save grad.pt

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
import numpy as np
import torch
import torch.nn as nn

from neural_gradient import (EllipsoidGradientNet, EllipsoidGradientTrainer,
                              vec_to_surface_exact_torch)


# ── Visualisierungs-Einstellungen ─────────────────────────────────────────────

_VIZ_RADII   = np.array([0.5, 0.4, 0.6], dtype=np.float32)  # Test-Ellipsoid (flach => GT-Pfeile innen in XY-Ebene)
_VIZ_EXTENT  = 0.85                                            # Bildbereich
_VIZ_N       = 96                                              # Magnitude-Aufl.
_QUIVER_N    = 22                                              # Pfeil-Aufl.


def _build_grid(n: int, extent: float = _VIZ_EXTENT):
    """XY-Gitter bei z=0."""
    lin = np.linspace(-extent, extent, n, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    pts = np.stack([xx.ravel(), yy.ravel(),
                    np.zeros(n * n, dtype=np.float32)], axis=-1)
    return pts, (n, n), lin, xx, yy


# ── Haupt-App ─────────────────────────────────────────────────────────────────

class TrainingApp:
    """
    Startet das Training in einem Background-Thread und zeigt Loss-Kurve +
    Gradientenfeld (Magnitude-Heatmap + Quiver-Pfeile) live an.
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
    ):
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.lr         = lr
        self.save_path  = save_path
        self.log_every  = log_every
        self.viz_every  = viz_every
        self._stop      = threading.Event()

        self.trainer = EllipsoidGradientTrainer(hidden=hidden, depth=depth,
                                                  device=device)
        self.device = self.trainer.device

        # Thread-sichere Queue: Training -> UI
        self._q: queue.Queue = queue.Queue()

        # Gitter vorberechnen
        self._pts_mag,  self._shape_mag, self._lin_mag, _, _      = \
            _build_grid(_VIZ_N)
        self._pts_quiv, self._shape_quiv, _, self._xx_q, self._yy_q = \
            _build_grid(_QUIVER_N)

        self._setup_figure()

    # ── Matplotlib-Fenster aufbauen ───────────────────────────────────────

    def _setup_figure(self):
        BG     = "#111827"
        PANEL  = "#1f2937"
        ACCENT = "#f97316"   # Orange
        TEXT   = "#e5e7eb"
        SUBTLE = "#6b7280"

        plt.ion()
        self.fig = plt.figure(figsize=(14, 7.5), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "Ellipsoid Gradient -- Training")

        gs = gridspec.GridSpec(
            2, 2,
            figure=self.fig,
            width_ratios=[1, 2.2],
            height_ratios=[1, 5],
            hspace=0.10,
            wspace=0.30,
        )

        # ── Titelzeile (Info-Text) ────────────────────────────────────────
        self.ax_info = self.fig.add_subplot(gs[0, :])
        self.ax_info.set_facecolor(BG)
        self.ax_info.axis("off")
        self._info_text = self.ax_info.text(
            0.0, 0.5,
            self._info_str(step=0, loss=float("nan")),
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
        self.ax_loss.set_xlabel("Step",        color=SUBTLE, fontsize=9)
        self.ax_loss.set_ylabel("L1 loss",     color=SUBTLE, fontsize=9)
        self.ax_loss.tick_params(colors=SUBTLE, labelsize=8)
        for sp in self.ax_loss.spines.values():
            sp.set_edgecolor("#374151")
        self.ax_loss.grid(color="#374151", lw=0.5, alpha=0.6)

        self._loss_line,   = self.ax_loss.plot(
            [], [], color=ACCENT, lw=1.5, alpha=0.9
        )
        self._loss_smooth, = self.ax_loss.plot(
            [], [], color="#fcd34d", lw=1.0, alpha=0.6, label="smoothed"
        )
        self.ax_loss.legend(fontsize=8, labelcolor=TEXT,
                            facecolor=PANEL, edgecolor="none")

        self._steps_hist: list[int]   = []
        self._loss_hist:  list[float] = []

        # ── Gradientenfeld ───────────────────────────────────────────────
        self.ax_g = self.fig.add_subplot(gs[1, 1])
        self.ax_g.set_facecolor(PANEL)
        self.ax_g.set_aspect("equal")
        self.ax_g.set_title("Gradient field  (z = 0)",
                             color=TEXT, fontsize=9, pad=4)
        self.ax_g.tick_params(colors=SUBTLE, labelsize=7)
        for sp in self.ax_g.spines.values():
            sp.set_edgecolor("#374151")

        # Magnitude-Heatmap (Distanz zur Oberflaeche)
        dummy = np.zeros(self._shape_mag, dtype=np.float32)
        self._im = self.ax_g.imshow(
            dummy, origin="lower",
            extent=[-_VIZ_EXTENT, _VIZ_EXTENT,
                    -_VIZ_EXTENT, _VIZ_EXTENT],
            cmap="magma", vmin=0.0, vmax=_VIZ_EXTENT,
            interpolation="bilinear",
        )
        cb = self.fig.colorbar(self._im, ax=self.ax_g,
                               fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=SUBTLE, labelsize=7)
        cb.outline.set_edgecolor("#374151")
        cb.set_label("|q − p|", color=SUBTLE, fontsize=8)

        # Ground-Truth-Kontur (Ellipse) als gestrichelte gruene Linie
        gt_inside = self._compute_gt_inside_mask()
        self.ax_g.contour(
            self._lin_mag, self._lin_mag, gt_inside.astype(np.float32),
            levels=[0.5], colors=["#4ade80"],
            linewidths=1.0, linestyles="--",
        )
        self.ax_g.text(
            0.02, 0.03, "-- Surface",
            transform=self.ax_g.transAxes,
            color="#4ade80", fontsize=7,
        )

        # Ground-Truth-Pfeile (unterlegt, gedaempfte Farbe) — zeigen wie die
        # Vorhersage aussehen SOLLTE.
        gt_vec = self._compute_gt_quiver()                     # (Nq, 3)
        U_gt = gt_vec[:, 0].reshape(self._shape_quiv)
        V_gt = gt_vec[:, 1].reshape(self._shape_quiv)
        self.ax_g.quiver(
            self._xx_q, self._yy_q, U_gt, V_gt,
            color="#4ade80",
            angles="xy", scale_units="xy", scale=1.0,
            width=0.003, alpha=0.55, zorder=2,
        )

        # Initial-Quiver fuer die Vorhersage (wird in _drain_queue ueberschrieben).
        # angles+scale_units='xy' und scale=1 => Pfeillaenge == |vec| in Daten-
        # koordinaten, d.h. die Pfeilspitze landet exakt auf der Oberflaeche.
        self._quiver = self.ax_g.quiver(
            self._xx_q, self._yy_q,
            np.zeros_like(self._xx_q), np.zeros_like(self._yy_q),
            color="#f97316",
            angles="xy", scale_units="xy", scale=1.0,
            width=0.004, alpha=0.9, zorder=3,
        )

        # Legende
        self.ax_g.text(
            0.02, 0.09, "-> Ground Truth",
            transform=self.ax_g.transAxes,
            color="#4ade80", fontsize=7,
        )
        self.ax_g.text(
            0.02, 0.06, "-> Prediction",
            transform=self.ax_g.transAxes,
            color="#f97316", fontsize=7,
        )

        # Q zum Abbrechen
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event",     self._on_close)

        self.fig.subplots_adjust(left=0.07, right=0.97,
                                  top=0.95, bottom=0.10,
                                  hspace=0.10, wspace=0.35)
        self.fig.canvas.draw()
        plt.pause(0.05)

    # ── Hilfsfunktionen ───────────────────────────────────────────────────

    def _info_str(self, step: int, loss: float) -> str:
        device_str = str(self.device).upper()
        return (
            f"Device: {device_str}   "
            f"Batch: {self.batch_size:,}   "
            f"Steps: {step:,} / {self.n_steps:,}"
        )

    def _compute_gt_inside_mask(self) -> np.ndarray:
        """Boolesche Maske: True innerhalb des Test-Ellipsoids (z=0-Schnitt)."""
        r = _VIZ_RADII.astype(np.float64)
        p = self._pts_mag.astype(np.float64)
        return (((p / r) ** 2).sum(axis=1) < 1.0).reshape(self._shape_mag)

    def _compute_gt_quiver(self) -> np.ndarray:
        """Exakter Verschiebungsvektor q − p auf dem Quiver-Gitter (CPU)."""
        pts   = torch.from_numpy(self._pts_quiv)
        radii = torch.from_numpy(
            np.broadcast_to(_VIZ_RADII, (pts.shape[0], 3)).copy()
        )
        with torch.no_grad():
            vec = vec_to_surface_exact_torch(pts, radii)
        return vec.cpu().numpy()

    @staticmethod
    def _smooth(values: list[float], w: int = 20) -> list[float]:
        if len(values) < w:
            return values
        kernel = np.ones(w) / w
        return np.convolve(values, kernel, mode="valid").tolist()

    # ── Tastatur / Fenster-Events ─────────────────────────────────────────

    def _on_key(self, event):
        if event.key in ("q", "Q", "escape"):
            print("\nInterrupted by user -- saving and exiting...")
            self._stop.set()

    def _on_close(self, event):
        self._stop.set()

    # ── Training (Background-Thread) ──────────────────────────────────────

    def _train_loop(self):
        net   = self.trainer.net
        opt   = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_steps, eta_min=1e-5
        )

        net.train()
        for step in range(1, self.n_steps + 1):
            if self._stop.is_set():
                break

            inp, target = self.trainer._generate_batch(
                self.batch_size, 0.05, 1.0
            )
            pred = net(inp)
            loss = nn.functional.l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            loss_val = loss.item()

            if step % self.log_every == 0 or step == 1:
                self._q.put(("loss", step, loss_val))

            if step % self.viz_every == 0:
                net.eval()
                with torch.no_grad():
                    vec_mag  = net.predict_np(
                        _VIZ_RADII, self._pts_mag,  device=self.device)
                    vec_quiv = net.predict_np(
                        _VIZ_RADII, self._pts_quiv, device=self.device)
                net.train()
                mag_grid = np.linalg.norm(vec_mag, axis=1) \
                              .reshape(self._shape_mag)
                self._q.put(("viz", step, (mag_grid, vec_quiv)))

        net.eval()
        self._q.put(("done", None, None))

    # ── UI-Update (Main-Thread) ───────────────────────────────────────────

    def _drain_queue(self):
        """Verarbeite alle Nachrichten in der Queue; True wenn Training fertig."""
        changed_loss = False
        changed_viz  = False
        done         = False

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
                self._last_viz      = data
                self._last_viz_step = step
                changed_viz = True

            elif kind == "done":
                done = True

        if changed_loss:
            xs = self._steps_hist
            ys = self._loss_hist
            self._loss_line.set_data(xs, ys)

            sm    = self._smooth(ys)
            xs_sm = xs[len(xs) - len(sm):]
            self._loss_smooth.set_data(xs_sm, sm)

            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

            self._info_text.set_text(self._info_str(xs[-1], ys[-1]))
            self._big_loss.set_text(f"L1 = {ys[-1]:.5f}")

        if changed_viz:
            mag_grid, vec_quiv = self._last_viz

            # Magnitude-Heatmap
            vmax = max(float(mag_grid.max()), 0.01)
            self._im.set_data(mag_grid)
            self._im.set_clim(0.0, vmax)

            # Quiver: U,V aus den XY-Komponenten der Vorhersage
            U = vec_quiv[:, 0].reshape(self._shape_quiv)
            V = vec_quiv[:, 1].reshape(self._shape_quiv)
            self._quiver.set_UVC(U, V)

            self.ax_g.set_title(
                f"Gradient field  (z=0,  step {self._last_viz_step:,})",
                color="#e5e7eb", fontsize=9, pad=4,
            )

        if changed_loss or changed_viz:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        return done

    # ── Hauptschleife ─────────────────────────────────────────────────────

    def run(self):
        print("=" * 58)
        print("  Ellipsoid Gradient -- Neural Network Training")
        print("=" * 58)
        print(f"  Steps    : {self.n_steps:,}")
        print(f"  Batch    : {self.batch_size:,}")
        print(f"  LR       : {self.lr}")
        print(f"  Device   : {self.device}")
        print(f"  Saving   : {self.save_path or '(not saved)'}")
        print("  Q / close window = cancel")
        print("=" * 58)

        self._last_viz      = (np.zeros(self._shape_mag,  dtype=np.float32),
                                np.zeros((np.prod(self._shape_quiv), 3),
                                         dtype=np.float32))
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
            print(f"\nModel saved: {self.save_path}")

        print("\nTraining finished -- close the window to exit.")
        plt.ioff()
        plt.show(block=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train EllipsoidGradientNet with live visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps",     type=int,   default=50_000)
    parser.add_argument("--batch",     type=int,   default=32_768)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--hidden",    type=int,   default=128)
    parser.add_argument("--depth",     type=int,   default=3)
    parser.add_argument("--device",    type=str,   default=None,
                        help="cpu or cuda (default: auto)")
    parser.add_argument("--save",      type=str,   default="ellipsoid_grad.pt")
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
    )
    app.run()


if __name__ == "__main__":
    main()
