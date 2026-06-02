"""
neural_sdf.py — Neuronale Netz-Approximation des Ellipsoid-SDF.

Stellt bereit:
  - EllipsoidSDFNet   : kleines MLP (6 Eingaben → 1 Ausgabe)
  - EllipsoidSDFTrainer : Datengenerierung, Training, Speichern/Laden

Das Netz arbeitet im normierten, achsen-symmetrischen Raum:
    Eingabe  = [rx/rmax, ry/rmax, rz/rmax, |x|/rmax, |y|/rmax, |z|/rmax]
    Ausgabe  = sdf / rmax

Damit ist das Netz skalen-invariant und nutzt die Oktant-Symmetrie des SDF.
Punkte müssen im lokalen Rahmen des Ellipsoids übergeben werden
(d.h. bereits zentriert und de-rotiert).

Schnellstart::

    trainer = EllipsoidSDFTrainer(device="cuda")
    net = trainer.train(n_steps=50_000, log_every=1000)
    trainer.save("ellipsoid_sdf.pt")

    # Laden und benutzen:
    net = EllipsoidSDFTrainer.load("ellipsoid_sdf.pt").net
    sdf = net.predict_np(radii=np.array([0.5, 0.3, 0.2]),
                         points=some_points)  # (N, 3) lokale Punkte
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Hilfsfunktionen Quaternion-Rotation ──────────────────────────────────────

def quat_rotate_inv_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotiere Vektor(en) v mit der INVERSEN der Quaternion q (xyzw-Konvention).

    Args:
        q_xyzw: (4,)  quaternion [qx, qy, qz, qw]
        v:      (N, 3) oder (3,) Vektoren

    Returns:
        Rotierte Vektoren, gleiche Form wie v.
    """
    qx, qy, qz, qw = q_xyzw.astype(np.float64)
    # Konjugierte Quaternion: xyz negieren → inverse Rotation
    tv = np.array([-qx, -qy, -qz], dtype=np.float64)
    batched = v.ndim == 2
    pts = v.reshape(-1, 3).astype(np.float64)
    cross1 = np.cross(tv[np.newaxis, :], pts)           # (N, 3)
    cross2 = np.cross(tv[np.newaxis, :], cross1)        # (N, 3)
    result = pts + 2.0 * qw * cross1 + 2.0 * cross2
    return result if batched else result[0]


def quat_rotate_inv_torch(q_xyzw: np.ndarray,
                           v: torch.Tensor) -> torch.Tensor:
    """
    Wie quat_rotate_inv_np, aber für einen torch.Tensor v (N, 3).

    Args:
        q_xyzw: (4,) numpy-Array [qx, qy, qz, qw]
        v:      (N, 3) torch.Tensor

    Returns:
        (N, 3) torch.Tensor — rotierte Punkte
    """
    qx, qy, qz, qw = float(q_xyzw[0]), float(q_xyzw[1]), \
                      float(q_xyzw[2]), float(q_xyzw[3])
    tv = torch.tensor([-qx, -qy, -qz], dtype=v.dtype, device=v.device)
    tv_exp = tv.unsqueeze(0).expand_as(v)
    cross1 = torch.linalg.cross(tv_exp, v)
    cross2 = torch.linalg.cross(tv_exp, cross1)
    return v + 2.0 * qw * cross1 + 2.0 * cross2


def quat_rotate_inv_batch_torch(
        q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Gebatchte inverse Quaternion-Rotation.

    Rodrigues-Formel auf (E, T, 3)-Tensoren:
        v_rot = v + 2·qw·(tv × v) + 2·(tv × (tv × v))
    mit tv = −q_xyz (Konjugat).

    Args:
        q_xyzw: (E, 4) float32 — Quaternionen in xyzw-Konvention
        v:      (E, T, 3) float32 — zu rotierende Vektoren

    Returns:
        (E, T, 3) float32 — rotierte Vektoren
    """
    tv = -q_xyzw[:, :3]                            # (E, 3)  Konjugat
    qw = q_xyzw[:, 3].unsqueeze(-1).unsqueeze(-1)  # (E, 1, 1)
    tv_exp = tv.unsqueeze(1).expand_as(v)           # (E, T, 3)
    cross1 = torch.linalg.cross(tv_exp, v)          # (E, T, 3)
    cross2 = torch.linalg.cross(tv_exp, cross1)     # (E, T, 3)
    return v + 2.0 * qw * cross1 + 2.0 * cross2


def eval_union_cuda(
    net,
    pts_world: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    rotations_xyzw: torch.Tensor,
    point_chunk: int = 131_072,
) -> torch.Tensor:
    """
    Vollständig gebatchter Union-SDF für E Ellipsoide an T Punkten.

    Anstelle einer Python-Schleife über Ellipsoide wird pro Chunk ein
    einziger Netzwerk-Vorwärtspass mit E·C Eingaben durchgeführt:

        pts_world (T,3)  →  pro Chunk (C,3)
        → lokale Rahmen (E,C,3) via gebatchter Rotation
        → Normierung  →  reshape (E·C, 6)
        → net(...)    →  (E·C,)
        → reshape (E, C)  →  min(dim=0)  →  (C,)
        → akkumuliertes Minimum  →  (T,)

    Speichersicher: ``point_chunk`` begrenzt die gleichzeitig verarbeiteten
    Punkte (Standard 128 K ≈ 25 MB GPU-Speicher für E=5).

    Args:
        net:             EllipsoidSDFNet — muss auf demselben Gerät liegen
        pts_world:       (T, 3) float32 — Anfragepunkte in Weltkoordinaten
        centers:         (E, 3) float32
        radii:           (E, 3) float32 — Halbachsen
        rotations_xyzw:  (E, 4) float32 — Quaternionen (xyzw-Konvention)
        point_chunk:     Punkte pro Chunk (Speicherkontrolle)

    Returns:
        (T,) float32 — Union-SDF auf demselben Gerät wie ``pts_world``
    """
    E   = centers.shape[0]
    T   = pts_world.shape[0]
    dev = pts_world.device

    # (E, 1) — Skalierungsfaktor pro Ellipsoid
    r_max   = radii.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    min_sdf = torch.full((T,), 1.0e6, dtype=torch.float32, device=dev)

    net.eval()
    with torch.inference_mode():
        for p0 in range(0, T, point_chunk):
            p1    = min(p0 + point_chunk, T)
            chunk = pts_world[p0:p1]                               # (C, 3)
            C     = p1 - p0

            # Weltpunkte → lokale Rahmen aller Ellipsoide
            local_raw = chunk.unsqueeze(0) - centers.unsqueeze(1)  # (E, C, 3)
            local_pts = quat_rotate_inv_batch_torch(
                rotations_xyzw, local_raw)                         # (E, C, 3)

            # Normierung und Eingabe zusammenbauen
            r_max_bc = r_max.unsqueeze(1)                          # (E, 1, 1)
            inp = torch.cat([
                radii.unsqueeze(1).expand(E, C, 3) / r_max_bc,
                local_pts.abs() / r_max_bc,
            ], dim=-1).reshape(E * C, 6)                           # (E·C, 6)

            # Einzelner Vorwärtspass
            sdf_norm = net(inp)                                    # (E·C,)

            # Denormierung → (E, C), Minimum über Ellipsoide → (C,)
            r_flat = r_max.expand(E, C).reshape(E * C)
            sdf    = (sdf_norm * r_flat).reshape(E, C)
            torch.minimum(min_sdf[p0:p1], sdf.min(dim=0).values,
                          out=min_sdf[p0:p1])

    return min_sdf


# ── Exakter SDF via Bisektionsverfahren (PyTorch, batchfähig) ─────────────────

def sdf_exact_torch(points: torch.Tensor, radii: torch.Tensor,
                    n_bisect: int = 64) -> torch.Tensor:
    """
    Exakter SDF eines achsen-ausgerichteten Ellipsoids am Ursprung.

    Berechnet den Parameter t mittels Bisektion, sodass der Fußpunkt
        q_i = r_i² · |p_i| / (r_i² + t)
    auf der Oberfläche liegt (Σ (q_i/r_i)² = 1).

    Args:
        points:   (N, 3) Anfragepunkte im lokalen Ellipsoidrahmen
        radii:    (N, 3) oder (1, 3) Halbachsen
        n_bisect: Bisektionsiterationen (64 reicht für float32)

    Returns:
        (N,) signed distances (negativ innen, positiv außen)
    """
    r = radii.double()
    p_orig = points.double()
    r2 = r ** 2

    inside = ((p_orig / r) ** 2).sum(dim=-1) < 1.0

    eps = 1e-10 * r.min(dim=-1, keepdim=True).values
    p = p_orig.abs().clamp(min=eps)

    T_max = r.max(dim=-1).values * p.norm(dim=-1) + r2.sum(dim=-1)
    T_min = -r2.min(dim=-1).values * (1.0 - 1e-15)

    t_lo = torch.where(inside, T_min, torch.zeros_like(T_min))
    t_hi = torch.where(inside, torch.zeros_like(T_max), T_max)

    for _ in range(n_bisect):
        t_mid = 0.5 * (t_lo + t_hi)
        denom = r2 + t_mid.unsqueeze(-1)                  # (N, 3)
        F = ((r * p / denom) ** 2).sum(dim=-1) - 1.0     # (N,)
        t_lo = torch.where(F > 0, t_mid, t_lo)
        t_hi = torch.where(F <= 0, t_mid, t_hi)

    t = 0.5 * (t_lo + t_hi)
    q = r2 * p / (r2 + t.unsqueeze(-1))
    dist = (p - q).norm(dim=-1)
    return torch.where(inside, -dist, dist).float()


# ── Netzwerk ──────────────────────────────────────────────────────────────────

class EllipsoidSDFNet(nn.Module):
    """
    Kleines MLP das den SDF eines einzelnen achsen-ausgerichteten
    Ellipsoids am Ursprung approximiert.

    Eingabe (normiert):  [rx/rmax, ry/rmax, rz/rmax, |x|/rmax, |y|/rmax, |z|/rmax]
    Ausgabe (normiert):  sdf / rmax

    Das Netz ist damit skalen-invariant. Punkte müssen **vor** dem Aufruf
    in den lokalen Ellipsoidrahmen transformiert werden
    (Translation und Rotation herausrechnen).
    """

    def __init__(self, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(6, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 6) normierte Eingabe → (N,) normierter SDF"""
        return self.net(x).squeeze(-1)

    @staticmethod
    def normalize(radii: torch.Tensor,
                  points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Erstelle die normierten Eingabefeatures.

        Args:
            radii:  (N, 3) Halbachsen
            points: (N, 3) Punkte (lokaler Rahmen)

        Returns:
            inp:   (N, 6) Netzwerk-Eingabe
            r_max: (N,)   Skalierungsfaktor zum Rückrechnen des SDF
        """
        r_max = radii.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        inp = torch.cat([radii / r_max, points.abs() / r_max], dim=-1)
        return inp, r_max.squeeze(-1)

    def predict_np(self,
                   radii: np.ndarray,
                   points: np.ndarray,
                   device: Optional[str] = None) -> np.ndarray:
        """
        SDF-Auswertung für mehrere (Halbachsen, Punkt)-Paare.

        Punkte müssen bereits im lokalen Ellipsoidrahmen vorliegen
        (zentriert und de-rotiert).

        Args:
            radii:  (3,) oder (N, 3) float32 — Halbachsen
            points: (N, 3) float32 — Anfragepunkte
            device: torch-Gerät (Standard: Gerät des Netzes)

        Returns:
            (N,) float32 — signed distances
        """
        if device is None:
            device = next(self.parameters()).device
        points = np.asarray(points, np.float32)
        N = len(points)
        radii_np = np.broadcast_to(np.asarray(radii, np.float32), (N, 3))

        # Pinned-Memory-Transfer fuer CUDA: vermeidet doppelte Kopie
        if str(device).startswith("cuda"):
            r_t = torch.as_tensor(radii_np.copy(), dtype=torch.float32,
                                  device=device)
            p_t = torch.as_tensor(points,          dtype=torch.float32,
                                  device=device)
        else:
            r_t = torch.from_numpy(radii_np.copy())
            p_t = torch.from_numpy(points)

        self.eval()
        with torch.no_grad():
            inp, r_max = self.normalize(r_t, p_t)
            sdf_norm = self(inp)
        return (sdf_norm * r_max).cpu().numpy()


# ── Trainer ───────────────────────────────────────────────────────────────────

class EllipsoidSDFTrainer:
    """
    Trainiert ein :class:`EllipsoidSDFNet` zur Approximation des exakten
    Ellipsoid-SDF.

    Datengenerierung:
      - 20 % gleichmäßig in einer Box [−3·rmax, 3·rmax]³ (Außenraum)
      - 50 % nahe der Oberfläche: Ellipsoidpunkt + Gaußrauschen
      - 30 % tief im Inneren: gleichmäßig in der Ellipsoid-Volumen

    Training:
      - Adam-Optimizer mit kosinusförmiger Lernratenabnahme
      - L1-Verlustfunktion (robust gegenüber Ausreißern)

    Schnellstart::

        trainer = EllipsoidSDFTrainer(device="cuda")
        net = trainer.train(n_steps=50_000, log_every=1000)
        trainer.save("ellipsoid_sdf.pt")

        net2 = EllipsoidSDFTrainer.load("ellipsoid_sdf.pt").net
    """

    def __init__(
        self,
        hidden: int = 128,
        depth: int = 3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = EllipsoidSDFNet(hidden=hidden, depth=depth).to(self.device)

    # ── Datengenerierung ──────────────────────────────────────────────────

    def _generate_batch(
        self,
        batch_size: int,
        r_min: float,
        r_max_val: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Erzeuge (Eingabe, Ziel)-Paare für einen Trainingsschritt.

        Returns:
            inp:    (N, 6) normierte Netzwerk-Eingabe
            target: (N,)  normierte SDF-Zielwerte
        """
        dev = self.device
        N = batch_size

        # Zufällige Halbachsen für jeden Datenpunkt
        radii = torch.empty(N, 3, device=dev).uniform_(r_min, r_max_val)
        r_max_per = radii.max(dim=-1, keepdim=True).values  # (N, 1)

        n_ext = int(0.2 * N)
        n_surf = int(0.5 * N)
        n_int = N - n_ext - n_surf

        # 1. Außenraum: gleichmäßig in [-3·rmax, 3·rmax]³
        p_ext = (torch.rand(n_ext, 3, device=dev) * 2.0 - 1.0) \
                * 3.0 * r_max_per[:n_ext]

        # 2. Oberflächennah: Ellipsoidpunkt + Gaußrauschen
        theta = torch.acos(torch.empty(n_surf, device=dev).uniform_(-1.0, 1.0))
        phi = torch.empty(n_surf, device=dev).uniform_(0.0, 2.0 * np.pi)
        unit_surf = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1)  # (n_surf, 3)
        sigma = 0.15 * r_max_per[n_ext:n_ext + n_surf]  # (n_surf, 1)
        p_surf = radii[n_ext:n_ext + n_surf] * unit_surf \
                 + torch.randn(n_surf, 3, device=dev) * sigma

        # 3. Innenraum: gleichmäßig im Ellipsoidvolumen
        #    Würfelwurzel-Trick für gleichmäßige Kugelverteilung
        u_vol = torch.rand(n_int, device=dev) ** (1.0 / 3.0)
        dir_in = torch.randn(n_int, 3, device=dev)
        dir_in = dir_in / dir_in.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        p_int = radii[n_ext + n_surf:] * dir_in * u_vol.unsqueeze(-1)

        points = torch.cat([p_ext, p_surf, p_int], dim=0)  # (N, 3)

        # Exakte SDF-Labels via Bisektionsverfahren auf der GPU
        with torch.no_grad():
            sdf = sdf_exact_torch(points, radii)

        # Normierung für Netzwerkeingabe
        r_max_f = r_max_per.squeeze(-1).clamp(min=1e-8)
        inp = torch.cat([radii / r_max_per, points.abs() / r_max_per], dim=-1)
        target = sdf / r_max_f

        return inp, target

    # ── Trainingsschleife ─────────────────────────────────────────────────

    def train(
        self,
        n_steps: int = 50_000,
        batch_size: int = 32_768,
        r_min: float = 0.05,
        r_max: float = 1.0,
        lr: float = 1e-3,
        log_every: int = 500,
        save_path: Optional[str] = None,
    ) -> "EllipsoidSDFNet":
        """
        Trainiere das Netz.

        Args:
            n_steps:    Anzahl Gradientenschritte
            batch_size: Datenpunkte pro Schritt
            r_min:      Minimale Halbachsenlänge in den Trainingsdaten
            r_max:      Maximale Halbachsenlänge in den Trainingsdaten
            lr:         Anfangslernrate (Kosinus-Abfall bis 1e-5)
            log_every:  Verlustausgabe alle N Schritte
            save_path:  Wenn angegeben, wird das Modell danach gespeichert

        Returns:
            Das trainierte :class:`EllipsoidSDFNet`.
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=1e-5
        )

        self.net.train()
        for step in range(1, n_steps + 1):
            inp, target = self._generate_batch(batch_size, r_min, r_max)

            pred = self.net(inp)
            loss = nn.functional.l1_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Step {step:6d}/{n_steps}  "
                      f"loss={loss.item():.6f}  "
                      f"lr={lr_now:.2e}")

        self.net.eval()
        if save_path:
            self.save(save_path)
        return self.net

    # ── Persistenz ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Speichere Gewichte und Architektur-Hyperparameter."""
        torch.save({
            "hidden":     self.net.hidden,
            "depth":      self.net.depth,
            "state_dict": self.net.state_dict(),
        }, path)
        print(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str,
             device: Optional[str] = None) -> "EllipsoidSDFTrainer":
        """Lade Trainer (inkl. Netz) von der Festplatte."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=True)
        trainer = cls(hidden=ckpt["hidden"], depth=ckpt["depth"], device=device)
        trainer.net.load_state_dict(ckpt["state_dict"])
        trainer.net.eval()
        return trainer
