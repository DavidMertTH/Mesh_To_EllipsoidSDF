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

    # ── Datengenerierung ──────────────────────────────────────────────────

    def _generate_batch(
        self,
        batch_size: int,
        r_min:      float,
        r_max_val:  float,
        grid_G:     int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Erzeuge (Eingabe, Ziel)-Paare fuer einen Trainingsschritt.

        Returns:
            inp:    (N, 12) normierte Netzwerk-Eingabe
            target: (N, 3)  baryzentrische Koordinaten des tiefsten Punktes
        """
        dev = self.device
        N   = batch_size

        radii    = torch.empty(N, 3, device=dev).uniform_(r_min, r_max_val)
        r_max_per = radii.max(dim=-1, keepdim=True).values          # (N, 1)

        half = N // 2

        # Haelfte 1: vollstaendig zufaellige Eckpunkte im Wuerfel
        def rand_verts(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            scale = 2.5 * r_max_per[:n]
            a = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * scale
            b = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * scale
            c = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * scale
            return a, b, c

        v1a, v2a, v3a = rand_verts(half)

        # Haelfte 2: eine Ecke nah an der Ellipsoid-Oberflaeche,
        #            die anderen zufaellig → hoher Anteil schneidender Dreiecke
        n2   = N - half
        rp2  = r_max_per[half:]
        # Ecke 1 auf Ellipsoid-Oberflaeche (nahe)
        theta = torch.acos(torch.empty(n2, device=dev).uniform_(-1.0, 1.0))
        phi   = torch.empty(n2, device=dev).uniform_(0.0, 2.0 * np.pi)
        unit  = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1)
        surf_pt = radii[half:] * unit                               # (n2, 3)
        sigma   = 0.3 * rp2
        v1b = surf_pt + torch.randn(n2, 3, device=dev) * sigma
        v2b = (torch.rand(n2, 3, device=dev) * 2.0 - 1.0) * 2.5 * rp2
        v3b = (torch.rand(n2, 3, device=dev) * 2.0 - 1.0) * 2.5 * rp2

        v1 = torch.cat([v1a, v1b], dim=0)
        v2 = torch.cat([v2a, v2b], dim=0)
        v3 = torch.cat([v3a, v3b], dim=0)

        # Ground Truth via Gittersuche
        target = triangle_deepest_exact_torch(radii, v1, v2, v3, G=grid_G)

        inp, _ = EllipsoidTriangleNet.normalize(radii, v1, v2, v3)
        return inp, target

    # ── Trainingsschleife ─────────────────────────────────────────────────

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

        self.net.train()
        for step in range(1, n_steps + 1):
            inp, target = self._generate_batch(batch_size, r_min, r_max, grid_G)
            pred = self.net(inp)
            loss = nn.functional.l1_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % log_every == 0:
                print(f"  Schritt {step:6d}/{n_steps}  "
                      f"Verlust={loss.item():.6f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

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
        print(f"Modell gespeichert: {path}")

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
        description="Trainiere EllipsoidTriangleNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps",     type=int,   default=50_000)
    parser.add_argument("--batch",     type=int,   default=4_096)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--depth",     type=int,   default=4)
    parser.add_argument("--grid-g",    type=int,   default=24,
                        help="Simplex-Gitteraufloesung fuer GT")
    parser.add_argument("--device",    type=str,   default=None,
                        help="cpu oder cuda (Standard: auto)")
    parser.add_argument("--save",      type=str,   default="ellipsoid_triangle.pt")
    parser.add_argument("--log-every", type=int,   default=500)
    args = parser.parse_args()

    print("=" * 58)
    print("  Ellipsoid Triangle -- Neural Network Training")
    print("=" * 58)
    print(f"  Schritte : {args.steps:,}")
    print(f"  Batch    : {args.batch:,}")
    print(f"  Grid G   : {args.grid_g}  ({(args.grid_g+1)*(args.grid_g+2)//2} Punkte/Dreieck)")
    print(f"  LR       : {args.lr}")
    print(f"  Speichern: {args.save}")
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
    )


if __name__ == "__main__":
    main()
