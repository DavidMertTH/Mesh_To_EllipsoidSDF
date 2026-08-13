"""
neural_triangle.py — Neuronales Netz fuer den tiefsten Dreieckspunkt im Ellipsoid-SDF.

Gegeben ein achsen-ausgerichtetes Ellipsoid am Ursprung und ein Dreieck (drei Eckpunkte
im lokalen Ellipsoid-Rahmen), liefert das Netz die baryzentrischen Koordinaten (u, v, w)
des Punktes auf dem Dreieck mit dem kleinsten SDF-Wert (tiefster Punkt im Ellipsoid).

Eingabe (normiert): [rx/rmax, ry/rmax, rz/rmax,
                     p1x/rmax, p1y/rmax, p1z/rmax,
                     p2x/rmax, p2y/rmax, p2z/rmax,
                     p3x/rmax, p3y/rmax, p3z/rmax]  -- 12 Werte

Ausgabe: (u, v, w) mit u+v+w=1, u,v,w >= 0
         Tiefster Punkt: q = u*p1 + v*p2 + w*p3
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from neural_sdf import sdf_exact_torch


class EarlyStopping:
    """
    Optionaler Early-Stopping-Helfer fuer noisy Trainingsverluste.

    patience zaehlt Trainingsschritte ohne Verbesserung des geglaetteten Loss.
    Bei aktivem restore_best wird der beste state_dict-Snapshot auf CPU gehalten.
    """

    def __init__(
        self,
        patience: int = 0,
        min_delta: float = 0.0,
        warmup: int = 0,
        smooth: int = 1,
        restore_best: bool = True,
    ):
        self.patience = max(0, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.warmup = max(0, int(warmup))
        self.smooth = max(1, int(smooth))
        self.restore_best_enabled = restore_best

        self.enabled = self.patience > 0
        self.best_metric = float("inf")
        self.best_step = 0
        self.bad_steps = 0
        self._loss_window: list[float] = []
        self._best_state: Optional[dict[str, torch.Tensor]] = None

    def update(self, step: int, loss_value: float, net: Optional[nn.Module] = None) -> bool:
        """Returns True, wenn das Training gestoppt werden soll."""
        if not self.enabled:
            return False

        self._loss_window.append(float(loss_value))
        if len(self._loss_window) > self.smooth:
            self._loss_window.pop(0)

        if step < self.warmup or len(self._loss_window) < self.smooth:
            return False

        metric = sum(self._loss_window) / len(self._loss_window)
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.best_step = step
            self.bad_steps = 0
            if self.restore_best_enabled and net is not None:
                self._best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in net.state_dict().items()
                }
        else:
            self.bad_steps += 1

        return self.bad_steps >= self.patience

    def restore_best(self, net: nn.Module) -> bool:
        if self._best_state is None:
            return False
        net.load_state_dict(self._best_state)
        return True


# ── Ground-Truth-Berechnung (Gittersuche im Simplex) ─────────────────────────

def _simplex_grid(G: int, device) -> torch.Tensor:
    """
    Erzeugt (M, 3) baryzentrische Gitterpunkte gleichmaessig im Einheits-Simplex.
    M = (G+1)(G+2)/2 Punkte.
    """
    uvw = []
    for i in range(G + 1):
        for j in range(G + 1 - i):
            uvw.append([i / G, j / G, 1.0 - i / G - j / G])
    return torch.tensor(uvw, dtype=torch.float32, device=device)  # (M, 3)


def _project_to_simplex(x: torch.Tensor) -> torch.Tensor:
    """Projiziert (..., 3) auf den Einheits-Simplex."""
    y, _ = torch.sort(x, dim=-1, descending=True)
    cssv = y.cumsum(dim=-1) - 1.0
    ind = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    cond = y - cssv / ind > 0
    rho = cond.sum(dim=-1, keepdim=True).clamp(min=1)
    theta = cssv.gather(-1, rho - 1) / rho.to(x.dtype)
    return torch.clamp(x - theta, min=0.0)


def generate_triangle_samples_torch(
    batch_size: int,
    r_min: float,
    r_max_val: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Erzeugt zufaellige Ellipsoid-/Dreieck-Kombinationen auf ``device``."""
    N = int(batch_size)
    radii = torch.empty(N, 3, device=device).uniform_(r_min, r_max_val)
    r_max_per = radii.max(dim=-1, keepdim=True).values

    half = N // 2

    def rand_verts(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = 2.5 * r_max_per[:n]
        a = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * scale
        b = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * scale
        c = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * scale
        return a, b, c

    v1a, v2a, v3a = rand_verts(half)

    n2 = N - half
    rp2 = r_max_per[half:]
    theta = torch.acos(torch.empty(n2, device=device).uniform_(-1.0, 1.0))
    phi = torch.empty(n2, device=device).uniform_(0.0, 2.0 * np.pi)
    unit = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ], dim=-1)
    surf_pt = radii[half:] * unit
    sigma = 0.3 * rp2
    v1b = surf_pt + torch.randn(n2, 3, device=device) * sigma
    v2b = (torch.rand(n2, 3, device=device) * 2.0 - 1.0) * 2.5 * rp2
    v3b = (torch.rand(n2, 3, device=device) * 2.0 - 1.0) * 2.5 * rp2

    v1 = torch.cat([v1a, v1b], dim=0)
    v2 = torch.cat([v2a, v2b], dim=0)
    v3 = torch.cat([v3a, v3b], dim=0)
    return radii, v1, v2, v3


def triangle_deepest_exact_torch(
    radii: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    G: int = 24,
    sdf_chunk: int = 524_288,
) -> torch.Tensor:
    """
    Findet die baryzentrischen Koordinaten des tiefsten Punktes (min SDF) auf dem Dreieck.

    Strategie: Gittersuche ueber (G+1)(G+2)/2 Abtastpunkte im Simplex, danach wird das
    globale Minimum zurueckgegeben. Bei gleichem Minimum (Dreieck komplett ausserhalb)
    wird der Punkt mit dem kleinsten SDF zurueckgegeben.

    Args:
        radii:      (N, 3) Halbachsen im lokalen Rahmen
        v1, v2, v3: (N, 3) Dreieck-Eckpunkte im lokalen Ellipsoid-Rahmen
        G:          Gitter-Aufloesung; ergibt (G+1)(G+2)/2 Abtastpunkte
        sdf_chunk:  Max. Punkte pro SDF-Aufruf (Speicherkontrolle)

    Returns:
        (N, 3) baryzentrische Koordinaten (u, v, w) des Minimums
    """
    device = radii.device
    N = radii.shape[0]

    bary = _simplex_grid(G, device)    # (M, 3)
    M = bary.shape[0]

    # Dreiecks-Punkte: q[n, m] = u*v1[n] + v*v2[n] + w*v3[n]
    v123 = torch.stack([v1, v2, v3], dim=1)           # (N, 3, 3)
    points = torch.einsum("mk,nkd->nmd", bary, v123)  # (N, M, 3)

    # SDF in Chunks auswerten (Speichersicherheit bei grossem N*M)
    pts_flat    = points.reshape(N * M, 3)
    radii_flat  = radii.unsqueeze(1).expand(N, M, 3).reshape(N * M, 3)

    sdf_parts = []
    for start in range(0, N * M, sdf_chunk):
        end = min(start + sdf_chunk, N * M)
        with torch.no_grad():
            sdf_parts.append(
                sdf_exact_torch(pts_flat[start:end], radii_flat[start:end])
            )
    sdf_flat = torch.cat(sdf_parts, dim=0)            # (N*M,)

    sdf_grid = sdf_flat.reshape(N, M)
    min_idx  = sdf_grid.argmin(dim=1)                 # (N,)
    return bary[min_idx]                               # (N, 3)


def triangle_deepest_refined_torch(
    radii: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    G: int = 32,
    refine_steps: int = 3,
    refine_grid: int = 7,
    sdf_chunk: int = 524_288,
) -> torch.Tensor:
    """
    Praezisere GT-Suche auf der GPU: erst Simplex-Gitter, danach lokale
    hierarchische Verfeinerung um das beste baryzentrische Sample.

    ``refine_grid=7`` erzeugt pro Verfeinerungsrunde 49 Kandidaten je Dreieck.
    ``refine_steps=3..5`` ist meist deutlich genauer als nur ein hohes ``G``,
    ohne die Kandidatenzahl explodieren zu lassen.
    """
    best = triangle_deepest_exact_torch(
        radii, v1, v2, v3, G=G, sdf_chunk=sdf_chunk
    )
    refine_steps = max(0, int(refine_steps))
    refine_grid = max(3, int(refine_grid))
    if refine_steps <= 0:
        return best

    device = radii.device
    N = radii.shape[0]
    v123 = torch.stack([v1, v2, v3], dim=1)  # (N, 3, 3)

    # Offsets in (u, v); w wird ueber 1-u-v rekonstruiert und danach auf den
    # Simplex projiziert. So bleiben auch Rand-/Eckloesungen erreichbar.
    lin = torch.linspace(-1.0, 1.0, refine_grid, device=device)
    du, dv = torch.meshgrid(lin, lin, indexing="ij")
    offsets_uv = torch.stack([du.reshape(-1), dv.reshape(-1)], dim=-1)
    C = offsets_uv.shape[0]

    radius = 2.0 / max(float(G), 1.0)
    best_sdf = None
    for _ in range(refine_steps):
        uv = best[:, :2].unsqueeze(1) + offsets_uv.unsqueeze(0) * radius
        w = 1.0 - uv[..., 0:1] - uv[..., 1:2]
        cand = _project_to_simplex(torch.cat([uv, w], dim=-1))  # (N, C, 3)

        points = torch.einsum("nck,nkd->ncd", cand, v123)
        pts_flat = points.reshape(N * C, 3)
        radii_flat = radii.unsqueeze(1).expand(N, C, 3).reshape(N * C, 3)

        sdf_parts = []
        for start in range(0, N * C, sdf_chunk):
            end = min(start + sdf_chunk, N * C)
            with torch.no_grad():
                sdf_parts.append(
                    sdf_exact_torch(pts_flat[start:end], radii_flat[start:end])
                )
        sdf_grid = torch.cat(sdf_parts, dim=0).reshape(N, C)
        min_idx = sdf_grid.argmin(dim=1)
        step_best_sdf = sdf_grid[torch.arange(N, device=device), min_idx]
        step_best = cand[torch.arange(N, device=device), min_idx]

        if best_sdf is None:
            best_sdf = step_best_sdf
            best = step_best
        else:
            improved = step_best_sdf < best_sdf
            best_sdf = torch.where(improved, step_best_sdf, best_sdf)
            best = torch.where(improved.unsqueeze(-1), step_best, best)

        radius *= 0.35

    return best


# ── Netzwerk ──────────────────────────────────────────────────────────────────

class EllipsoidTriangleNet(nn.Module):
    """
    MLP: 12 normierte Eingaben → 3 Logits → Softmax → baryzentrische Koordinaten.

    Eingabe: [rx/rmax, ry/rmax, rz/rmax,
              p1/rmax (3 Werte), p2/rmax (3 Werte), p3/rmax (3 Werte)]
    Ausgabe: (u, v, w) mit u+v+w=1 und u,v,w >= 0
             Tiefster Punkt: q = u*p1 + v*p2 + w*p3

    Punkte muessen im lokalen Ellipsoid-Rahmen vorliegen (zentriert, de-rotiert).
    Keine Symmetrie-Ausnutzung — Roh-Training ueber alle Konfigurationen.
    """

    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(12, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.depth  = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 12) → (N, 3) baryzentrische Koordinaten via Softmax."""
        return torch.softmax(self.net(x), dim=-1)

    @staticmethod
    def normalize(
        radii: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        v3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Erstelle normierte Netzwerk-Eingabe.

        Returns:
            inp:   (N, 12) Netzwerk-Eingabe
            r_max: (N,)    Skalierungsfaktor (zum Ruecknormieren von Punkten)
        """
        r_max = radii.max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # (N, 1)
        inp = torch.cat([
            radii / r_max,
            v1    / r_max,
            v2    / r_max,
            v3    / r_max,
        ], dim=-1)                                                        # (N, 12)
        return inp, r_max.squeeze(-1)

    def predict_np(
        self,
        radii: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Vorhersage der baryzentrischen Koordinaten fuer mehrere (Ellipsoid, Dreieck)-Paare.

        Args:
            radii:      (3,) oder (N, 3) Halbachsen
            v1, v2, v3: (N, 3) Dreieck-Eckpunkte im lokalen Ellipsoid-Rahmen

        Returns:
            (N, 3) baryzentrische Koordinaten (u, v, w)
        """
        if device is None:
            device = next(self.parameters()).device

        v1 = np.asarray(v1, np.float32)
        v2 = np.asarray(v2, np.float32)
        v3 = np.asarray(v3, np.float32)
        N  = len(v1)
        radii_np = np.broadcast_to(
            np.asarray(radii, np.float32), (N, 3)
        ).copy()

        def _t(arr: np.ndarray) -> torch.Tensor:
            if str(device).startswith("cuda"):
                return torch.as_tensor(arr.copy(), dtype=torch.float32,
                                       device=device)
            return torch.from_numpy(arr.copy())

        r_t  = _t(radii_np)
        v1_t = _t(v1)
        v2_t = _t(v2)
        v3_t = _t(v3)

        self.eval()
        with torch.no_grad():
            inp, _ = self.normalize(r_t, v1_t, v2_t, v3_t)
            bary   = self(inp)
        return bary.cpu().numpy()


# ── Trainer ───────────────────────────────────────────────────────────────────

class EllipsoidTriangleTrainer:
    """
    Trainiert ein :class:`EllipsoidTriangleNet`.

    Datengenerierung:
      - Zufaellige Ellipsoide (r_min, r_max)
      - Dreieck-Eckpunkte gleichverteilt in [-2.5·rmax, 2.5·rmax]³:
          50% vollstaendig zufaellig
          50% mit mindestens einer Ecke in der Naehe der Oberflaeche,
              damit alle Konfigurationen (innen / schneidend / aussen) vertreten sind

    Verlust: L1 auf den drei baryzentrischen Koordinaten.

    Hinweis: batch_size deutlich kleiner als beim SDF-Trainer (~4 096),
    da die Ground-Truth-Berechnung pro Sample (G+1)(G+2)/2 SDF-Auswertungen benoetigt.
    """

    def __init__(
        self,
        hidden: int = 256,
        depth:  int = 4,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = EllipsoidTriangleNet(hidden=hidden,
                                         depth=depth).to(self.device)
        self._cache_inp: Optional[torch.Tensor] = None
        self._cache_target: Optional[torch.Tensor] = None

    # ── Datengenerierung ──────────────────────────────────────────────────

    def _generate_batch(
        self,
        batch_size: int,
        r_min:      float,
        r_max_val:  float,
        grid_G:     int,
        refine_steps: int = 0,
        refine_grid: int = 7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Erzeuge (Eingabe, Ziel)-Paare fuer einen Trainingsschritt.

        Returns:
            inp:    (N, 12) normierte Netzwerk-Eingabe
            target: (N, 3)  baryzentrische Koordinaten des tiefsten Punktes
        """
        radii, v1, v2, v3 = generate_triangle_samples_torch(
            batch_size, r_min, r_max_val, self.device
        )
        if refine_steps > 0:
            target = triangle_deepest_refined_torch(
                radii, v1, v2, v3, G=grid_G,
                refine_steps=refine_steps, refine_grid=refine_grid,
            )
        else:
            target = triangle_deepest_exact_torch(radii, v1, v2, v3, G=grid_G)
        inp, _ = EllipsoidTriangleNet.normalize(radii, v1, v2, v3)
        return inp, target

        #            die anderen zufaellig → hoher Anteil schneidender Dreiecke
    # ── Trainingsschleife ─────────────────────────────────────────────────

    def precompute_gt_dataset(
        self,
        n_samples: int,
        chunk_size: int,
        r_min: float,
        r_max_val: float,
        grid_G: int,
        refine_steps: int = 3,
        refine_grid: int = 7,
        progress_cb=None,
    ) -> None:
        """
        Berechnet viele GT-Paare vorab auf ``self.device`` und haelt sie dort.

        Gespeichert werden nur die normierten Netzeingaben (N, 12) und Targets
        (N, 3). Fuer 100k Samples sind das ca. 6 MB in float32.
        """
        n_samples = max(0, int(n_samples))
        chunk_size = max(1, int(chunk_size))
        if n_samples <= 0:
            self._cache_inp = None
            self._cache_target = None
            return

        inp_parts = []
        target_parts = []
        done = 0
        while done < n_samples:
            n = min(chunk_size, n_samples - done)
            radii, v1, v2, v3 = generate_triangle_samples_torch(
                n, r_min, r_max_val, self.device
            )
            target = triangle_deepest_refined_torch(
                radii, v1, v2, v3, G=grid_G,
                refine_steps=refine_steps, refine_grid=refine_grid,
            )
            inp, _ = EllipsoidTriangleNet.normalize(radii, v1, v2, v3)
            inp_parts.append(inp.detach())
            target_parts.append(target.detach())
            done += n
            if progress_cb is not None:
                progress_cb(done, n_samples)

        self._cache_inp = torch.cat(inp_parts, dim=0).contiguous()
        self._cache_target = torch.cat(target_parts, dim=0).contiguous()

    def sample_precomputed_batch(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache_inp is None or self._cache_target is None:
            raise RuntimeError("GT cache is empty; call precompute_gt_dataset first.")
        n = self._cache_inp.shape[0]
        idx = torch.randint(0, n, (int(batch_size),), device=self._cache_inp.device)
        return self._cache_inp[idx], self._cache_target[idx]

    def train(
        self,
        n_steps:    int   = 50_000,
        batch_size: int   = 4_096,
        r_min:      float = 0.05,
        r_max:      float = 1.0,
        lr:         float = 1e-3,
        log_every:  int   = 500,
        grid_G:     int   = 24,
        save_path:  Optional[str] = None,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        early_stopping_warmup: int = 0,
        early_stopping_smooth: int = 1,
        gt_refine_steps: int = 0,
        gt_refine_grid: int = 7,
        precompute_samples: int = 0,
        precompute_chunk: int = 2_048,
    ) -> "EllipsoidTriangleNet":
        """
        Args:
            batch_size: Klein halten (~4 096) da GT-Berechnung teuer ist.
            grid_G:     Simplex-Gitteraufloesung fuer den GT (Standard 24 → 325 Punkte).
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=1e-5
        )
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            warmup=early_stopping_warmup,
            smooth=early_stopping_smooth,
            restore_best=True,
        )
        if precompute_samples > 0:
            print(
                f"  Precomputing GT: {precompute_samples:,} samples "
                f"(chunk={precompute_chunk:,}, refine={gt_refine_steps}x{gt_refine_grid})"
            )
            self.precompute_gt_dataset(
                n_samples=precompute_samples,
                chunk_size=precompute_chunk,
                r_min=r_min,
                r_max_val=r_max,
                grid_G=grid_G,
                refine_steps=gt_refine_steps,
                refine_grid=gt_refine_grid,
            )

        self.net.train()
        for step in range(1, n_steps + 1):
            if precompute_samples > 0:
                inp, target = self.sample_precomputed_batch(batch_size)
            else:
                inp, target = self._generate_batch(
                    batch_size, r_min, r_max, grid_G,
                    refine_steps=gt_refine_steps,
                    refine_grid=gt_refine_grid,
                )
            pred = self.net(inp)
            loss = nn.functional.l1_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if early_stopping.enabled:
                with torch.no_grad():
                    loss_value = nn.functional.l1_loss(self.net(inp), target).item()
            else:
                loss_value = loss.item()
            stop_now = early_stopping.update(step, loss_value, self.net)

            if step % log_every == 0:
                print(f"  Step {step:6d}/{n_steps}  "
                      f"loss={loss_value:.6f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

            if stop_now:
                early_stopping.restore_best(self.net)
                print(
                    f"  Early stopping at step {step:,}; "
                    f"best smoothed loss={early_stopping.best_metric:.6f} "
                    f"at step {early_stopping.best_step:,}"
                )
                break

        self.net.eval()
        if save_path:
            self.save(save_path)
        return self.net

    # ── Persistenz ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "hidden":     self.net.hidden,
            "depth":      self.net.depth,
            "state_dict": self.net.state_dict(),
        }, path)
        print(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str,
             device: Optional[str] = None) -> "EllipsoidTriangleTrainer":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt   = torch.load(path, map_location=device, weights_only=True)
        trainer = cls(hidden=ckpt["hidden"], depth=ckpt["depth"], device=device)
        trainer.net.load_state_dict(ckpt["state_dict"])
        trainer.net.eval()
        return trainer


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train EllipsoidTriangleNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps",     type=int,   default=50_000)
    parser.add_argument("--batch",     type=int,   default=16_384)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--depth",     type=int,   default=4)
    parser.add_argument("--grid-g",    type=int,   default=24,
                        help="Simplex grid resolution for GT")
    parser.add_argument("--device",    type=str,   default=None,
                        help="cpu or cuda (default: auto)")
    parser.add_argument("--save",      type=str,   default="ellipsoid_triangle.pt")
    parser.add_argument("--log-every", type=int,   default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=2000,
                        help="0 disables early stopping; otherwise steps without improvement")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5,
                        help="Minimum smoothed-loss improvement required")
    parser.add_argument("--early-stopping-warmup", type=int, default=5000,
                        help="Steps before early stopping starts watching")
    parser.add_argument("--early-stopping-smooth", type=int, default=100,
                        help="Moving-average window for the watched loss")
    parser.add_argument("--gt-refine-steps", type=int, default=3,
                        help="Local simplex refinement rounds after coarse GT grid")
    parser.add_argument("--gt-refine-grid", type=int, default=7,
                        help="Candidates per axis for each GT refinement round")
    parser.add_argument("--precompute-samples", type=int, default=100_000,
                        help="Precompute this many GT samples on the training device")
    parser.add_argument("--precompute-chunk", type=int, default=2048,
                        help="GT precompute chunk size")
    args = parser.parse_args()

    print("=" * 58)
    print("  Ellipsoid Triangle -- Neural Network Training")
    print("=" * 58)
    print(f"  Steps    : {args.steps:,}")
    print(f"  Batch    : {args.batch:,}")
    print(f"  Grid G   : {args.grid_g}  ({(args.grid_g+1)*(args.grid_g+2)//2} points/triangle)")
    print(f"  LR       : {args.lr}")
    print(f"  Saving   : {args.save}")
    if args.early_stopping_patience > 0:
        print(f"  EarlyStop: patience={args.early_stopping_patience:,}  "
              f"min_delta={args.early_stopping_min_delta:g}  "
              f"warmup={args.early_stopping_warmup:,}  "
              f"smooth={args.early_stopping_smooth:,}")
    if args.gt_refine_steps > 0 or args.precompute_samples > 0:
        print(f"  GT refine: steps={args.gt_refine_steps:,}  "
              f"grid={args.gt_refine_grid:,}  "
              f"precompute={args.precompute_samples:,}")
    print("=" * 58)

    trainer = EllipsoidTriangleTrainer(
        hidden=args.hidden, depth=args.depth, device=args.device
    )
    trainer.train(
        n_steps    = args.steps,
        batch_size = args.batch,
        lr         = args.lr,
        log_every  = args.log_every,
        grid_G     = args.grid_g,
        save_path  = args.save,
        early_stopping_patience = args.early_stopping_patience,
        early_stopping_min_delta = args.early_stopping_min_delta,
        early_stopping_warmup = args.early_stopping_warmup,
        early_stopping_smooth = args.early_stopping_smooth,
        gt_refine_steps = args.gt_refine_steps,
        gt_refine_grid = args.gt_refine_grid,
        precompute_samples = args.precompute_samples,
        precompute_chunk = args.precompute_chunk,
    )


if __name__ == "__main__":
    main()
