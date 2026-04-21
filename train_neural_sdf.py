"""
train_neural_sdf.py -- Live-Training-UI fuer EllipsoidSDFNet

Aufruf:
    python train_neural_sdf.py
    python train_neural_sdf.py --steps 100000 --batch 32768 --save model.pt

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

from neural_sdf import EllipsoidSDFNet, EllipsoidSDFTrainer


# ── Visualisierungs-Einstellungen ─────────────────────────────────────────────

_VIZ_RADII  = np.array([0.5, 0.3, 0.2], dtype=np.float32)   # Test-Ellipsoid
_VIZ_EXTENT = 0.85                                            # Bildbereich
_VIZ_N      = 128                                             # Schnitt-Aufloesung


def _build_slice_grid(n: int = _VIZ_N, extent: float = _VIZ_EXTENT):
    """XY-Gitter bei z=0 fuer das Querschnittsbild."""
    lin = np.linspace(-extent, extent, n, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    pts = np.stack([xx.ravel(), yy.ravel(),
                    np.zeros(n * n, dtype=np.float32)], axis=-1)
    return pts, (n, n), lin


# ── Haupt-App ─────────────────────────────────────────────────────────────────

class TrainingApp:
    """
    Startet das Training in einem Background-Thread und
    zeigt Loss-Kurve + SDF-Querschnitt live im Matplotlib-Fenster.
    """

    def __init__(
        self,
        n_steps:   int,
        batch_size: int,
        lr:        float,
        hidden:    int,
        depth:     int,
        device:    Optional[str],
        save_path: str,
        log_every: int,
        viz_every: int,
    ):
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.lr         = lr
        self.save_path  = save_path
        self.log_every  = log_every
        self.viz_every  = viz_every
        self._stop      = threading.Event()

        self.trainer = EllipsoidSDFTrainer(hidden=hidden, depth=depth,
                                           device=device)
        self.device = self.trainer.device

        # Thread-sichere Queue: Training -> UI
        self._q: queue.Queue = queue.Queue()

        # Slice-Gitter vorberechnen
        self._pts, self._shape, self._lin = _build_slice_grid()

        self._setup_figure()

    # ── Matplotlib-Fenster aufbauen ───────────────────────────────────────

    def _setup_figure(self):
        BG      = "#111827"
        PANEL   = "#1f2937"
        ACCENT  = "#f97316"   # Orange
        TEXT    = "#e5e7eb"
        SUBTLE  = "#6b7280"

        plt.ion()
        self.fig = plt.figure(figsize=(13, 5.5), facecolor=BG)
        self.fig.canvas.manager.set_window_title("Ellipsoid SDF -- Training")

        gs = gridspec.GridSpec(
            2, 2,
            figure=self.fig,
            width_ratios=[2.8, 1],
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
        self.ax_loss.set_xlabel("Schritt", color=SUBTLE, fontsize=9)
        self.ax_loss.set_ylabel("L1-Verlust", color=SUBTLE, fontsize=9)
        self.ax_loss.tick_params(colors=SUBTLE, labelsize=8)
        for sp in self.ax_loss.spines.values():
            sp.set_edgecolor("#374151")
        self.ax_loss.grid(color="#374151", lw=0.5, alpha=0.6)

        self._loss_line, = self.ax_loss.plot(
            [], [], color=ACCENT, lw=1.5, alpha=0.9
        )
        self._loss_smooth, = self.ax_loss.plot(
            [], [], color="#fcd34d", lw=1.0, alpha=0.6, label="geglaettet"
        )
        self.ax_loss.legend(fontsize=8, labelcolor=TEXT,
                            facecolor=PANEL, edgecolor="none")

        self._steps_hist: list[int]   = []
        self._loss_hist:  list[float] = []

        # ── SDF-Querschnitt ───────────────────────────────────────────────
        self.ax_sdf = self.fig.add_subplot(gs[1, 1])
        self.ax_sdf.set_facecolor(PANEL)
        self.ax_sdf.set_aspect("equal")
        self.ax_sdf.set_title("SDF Schnitt  (z = 0)", color=TEXT,
                               fontsize=9, pad=4)
        self.ax_sdf.tick_params(colors=SUBTLE, labelsize=7)
        for sp in self.ax_sdf.spines.values():
            sp.set_edgecolor("#374151")

        dummy = np.zeros(self._shape, dtype=np.float32)
        self._im = self.ax_sdf.imshow(
            dummy, origin="lower",
            extent=[-_VIZ_EXTENT, _VIZ_EXTENT,
                    -_VIZ_EXTENT, _VIZ_EXTENT],
            cmap="RdBu_r", vmin=-_VIZ_EXTENT, vmax=_VIZ_EXTENT,
            interpolation="bilinear",
        )
        cb = self.fig.colorbar(self._im, ax=self.ax_sdf,
                               fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=SUBTLE, labelsize=7)
        cb.outline.set_edgecolor("#374151")

        # Zeige Ground-Truth-Kontur als gestrichelte Linie
        gt_sdf = self._compute_gt_slice()
        self.ax_sdf.contour(
            self._lin, self._lin, gt_sdf,
            levels=[0.0], colors=["#4ade80"],
            linewidths=1.0, linestyles="--",
        )
        self.ax_sdf.text(
            0.02, 0.03, "-- Ground Truth",
            transform=self.ax_sdf.transAxes,
            color="#4ade80", fontsize=7,
        )

        self._nn_contour = None

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
            f"Geraet: {device_str}   "
            f"Batch: {self.batch_size:,}   "
            f"Schritte: {step:,} / {self.n_steps:,}"
        )

    def _compute_gt_slice(self) -> np.ndarray:
        """Exakter SDF fuer das Test-Ellipsoid (numpy Bisektion)."""
        pts = self._pts.astype(np.float64)
        r   = _VIZ_RADII.astype(np.float64)
        r2  = r ** 2
        inside = np.sum((pts / r) ** 2, axis=1) < 1.0
        eps = 1e-10 * r.min()
        p   = np.maximum(np.abs(pts), eps)
        T_max = r.max() * np.linalg.norm(p, axis=1) + r2.sum()
        T_min = -r2.min() * (1.0 - 1e-15)
        t_lo = np.where(inside, T_min, 0.0)
        t_hi = np.where(inside, 0.0, T_max)
        for _ in range(80):
            t_m = 0.5 * (t_lo + t_hi)
            F   = np.sum((r * p / (r2 + t_m[:, None])) ** 2, axis=1) - 1.0
            t_lo = np.where(F > 0, t_m, t_lo)
            t_hi = np.where(F <= 0, t_m, t_hi)
        t = 0.5 * (t_lo + t_hi)
        q = r2 * p / (r2 + t[:, None])
        d = np.linalg.norm(p - q, axis=1)
        return np.where(inside, -d, d).astype(np.float32).reshape(self._shape)

    @staticmethod
    def _smooth(values: list[float], w: int = 20) -> list[float]:
        if len(values) < w:
            return values
        kernel = np.ones(w) / w
        return np.convolve(values, kernel, mode="valid").tolist()

    # ── Tastatur / Fenster-Events ─────────────────────────────────────────

    def _on_key(self, event):
        if event.key in ("q", "Q", "escape"):
            print("\nAbbruch durch Nutzer -- speichere und beende...")
            self._stop.set()

    def _on_close(self, event):
        self._stop.set()

    # ── Training (Background-Thread) ──────────────────────────────────────

    def _train_loop(self):
        net  = self.trainer.net
        opt  = torch.optim.Adam(net.parameters(), lr=self.lr)
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
                    sdf = net.predict_np(
                        _VIZ_RADII, self._pts, device=self.device
                    )
                net.train()
                self._q.put(("sdf", step, sdf.reshape(self._shape)))

        net.eval()
        self._q.put(("done", None, None))

    # ── UI-Update (Main-Thread) ───────────────────────────────────────────

    def _drain_queue(self):
        """Verarbeite alle Nachrichten in der Queue; gibt True zurueck wenn fertig."""
        changed_loss = False
        changed_sdf  = False
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

            elif kind == "sdf":
                self._last_sdf      = data
                self._last_sdf_step = step
                changed_sdf = True

            elif kind == "done":
                done = True

        if changed_loss:
            xs = self._steps_hist
            ys = self._loss_hist
            self._loss_line.set_data(xs, ys)

            sm = self._smooth(ys)
            xs_sm = xs[len(xs) - len(sm):]
            self._loss_smooth.set_data(xs_sm, sm)

            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

            self._info_text.set_text(self._info_str(xs[-1], ys[-1]))
            self._big_loss.set_text(f"L1 = {ys[-1]:.5f}")

        if changed_sdf:
            sdf_img = self._last_sdf
            vmax = max(float(np.abs(sdf_img).max()), 0.01)
            self._im.set_data(sdf_img)
            self._im.set_clim(-vmax, vmax)

            # Neuronale Null-Kontur aktualisieren
            if self._nn_contour is not None:
                try:
                    self._nn_contour.remove()
                except Exception:
                    try:
                        for c in self._nn_contour.collections:
                            c.remove()
                    except Exception:
                        pass

            self._nn_contour = self.ax_sdf.contour(
                self._lin, self._lin, sdf_img,
                levels=[0.0], colors=["#f97316"],
                linewidths=1.5,
            )
            self.ax_sdf.set_title(
                f"SDF Schnitt  (z=0,  Schritt {self._last_sdf_step:,})",
                color="#e5e7eb", fontsize=9, pad=4,
            )

        if changed_loss or changed_sdf:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        return done

    # ── Hauptschleife ─────────────────────────────────────────────────────

    def run(self):
        print("=" * 58)
        print("  Ellipsoid SDF -- Neural Network Training")
        print("=" * 58)
        print(f"  Schritte : {self.n_steps:,}")
        print(f"  Batch    : {self.batch_size:,}")
        print(f"  LR       : {self.lr}")
        print(f"  Geraet   : {self.device}")
        print(f"  Speichern: {self.save_path or '(nicht gespeichert)'}")
        print("  Q / Fenster schliessen = Abbrechen")
        print("=" * 58)

        self._last_sdf      = np.zeros(self._shape, dtype=np.float32)
        self._last_sdf_step = 0

        thread = threading.Thread(target=self._train_loop, daemon=True)
        thread.start()

        while thread.is_alive():
            done = self._drain_queue()
            plt.pause(0.05)
            if done or self._stop.is_set():
                break

        # Noch ausstehende Updates verarbeiten
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
        description="Trainiere EllipsoidSDFNet mit Live-Visualisierung",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps",     type=int,   default=50_000,
                        help="Anzahl Trainingsschritte")
    parser.add_argument("--batch",     type=int,   default=32_768,
                        help="Batch-Groesse pro Schritt")
    parser.add_argument("--lr",        type=float, default=1e-3,
                        help="Anfangs-Lernrate")
    parser.add_argument("--hidden",    type=int,   default=128,
                        help="Neuronen pro versteckter Schicht")
    parser.add_argument("--depth",     type=int,   default=3,
                        help="Anzahl versteckter Schichten")
    parser.add_argument("--device",    type=str,   default=None,
                        help="Trainingsgeraet: cpu oder cuda (Standard: auto)")
    parser.add_argument("--save",      type=str,   default="ellipsoid_sdf.pt",
                        help="Speicherpfad fuer das trainierte Modell")
    parser.add_argument("--log-every", type=int,   default=10,
                        help="Loss alle N Schritte in die Queue schicken")
    parser.add_argument("--viz-every", type=int,   default=100,
                        help="SDF-Bild alle N Schritte aktualisieren")
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
