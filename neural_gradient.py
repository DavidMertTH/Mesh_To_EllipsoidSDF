"""
neural_gradient.py — Neuronales Netz fuer Ellipsoid-Gradienten.

Statt eines skalaren SDF-Wertes liefert das Netz einen 3D-Vektor, der vom
Anfragepunkt auf den naechsten Punkt der Ellipsoidoberflaeche zeigt
(Verschiebungsvektor q − p).

Eingabe (normiert):  [rx/rmax, ry/rmax, rz/rmax, |x|/rmax, |y|/rmax, |z|/rmax]
Ausgabe (normiert):  (q − |p|) / rmax  im positiven Oktanten

Die Vorzeichen-Symmetrie wird beim ``predict_np`` automatisch zurueckgerechnet
(elementweise Multiplikation mit ``sign(p)``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Exakter Verschiebungsvektor (Bisektionsverfahren) ────────────────────────

def vec_to_surface_exact_torch(points: torch.Tensor, radii: torch.Tensor,
                                n_bisect: int = 64) -> torch.Tensor:
    """
    Exakter Verschiebungsvektor q − p fuer ein achsen-ausgerichtetes Ellipsoid
    am Ursprung.

    Berechnet den Fusspunkt q auf der Oberflaeche per Bisektion (gleicher
    Algorithmus wie ``sdf_exact_torch``) und liefert ``q − p`` zurueck.

    Args:
        points:   (N, 3) Anfragepunkte im lokalen Ellipsoidrahmen
        radii:    (N, 3) oder (1, 3) Halbachsen
        n_bisect: Bisektionsiterationen (64 reicht fuer float32)

    Returns:
        (N, 3) float32 — Verschiebungsvektoren (zeigen zur Oberflaeche)
    """
    r = radii.double()
    p_orig = points.double()
    r2 = r ** 2

    inside = ((p_orig / r) ** 2).sum(dim=-1) < 1.0

    eps = 1e-10 * r.min(dim=-1, keepdim=True).values
    p_abs = p_orig.abs().clamp(min=eps)

    T_max = r.max(dim=-1).values * p_abs.norm(dim=-1) + r2.sum(dim=-1)
    T_min = -r2.min(dim=-1).values * (1.0 - 1e-15)

    t_lo = torch.where(inside, T_min, torch.zeros_like(T_min))
    t_hi = torch.where(inside, torch.zeros_like(T_max), T_max)

    for _ in range(n_bisect):
        t_mid = 0.5 * (t_lo + t_hi)
        denom = r2 + t_mid.unsqueeze(-1)
        F = ((r * p_abs / denom) ** 2).sum(dim=-1) - 1.0
        t_lo = torch.where(F > 0, t_mid, t_lo)
        t_hi = torch.where(F <= 0, t_mid, t_hi)

    t = 0.5 * (t_lo + t_hi)
    q_abs = r2 * p_abs / (r2 + t.unsqueeze(-1))   # Fusspunkt im positiven Oktanten

    # Zurueck in den Original-Oktanten
    sign = torch.sign(p_orig)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q_abs * sign
    return (q - p_orig).float()


# ── Netzwerk ──────────────────────────────────────────────────────────────────

class EllipsoidGradientNet(nn.Module):
    """
    MLP: 6 normierte Eingaben → 3 normierte Ausgaben.

    Eingabe:  [rx/rmax, ry/rmax, rz/rmax, |x|/rmax, |y|/rmax, |z|/rmax]
    Ausgabe:  (q − |p|) / rmax  im positiven Oktanten

    Die Komponenten-Vorzeichen werden ausserhalb des Netzes mit ``sign(p)``
    zurueck reflektiert (Punkt-Symmetrie des Ellipsoids).
    """

    def __init__(self, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(6, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 6) → (N, 3) normierter Verschiebungsvektor (positiver Oktant)."""
        return self.net(x)

    @staticmethod
    def normalize(radii: torch.Tensor, points: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Erstelle Eingabefeatures und Vorzeichen-Reflexion.

        Returns:
            inp:   (N, 6) Netzwerk-Eingabe
            r_max: (N,)   Skalierungsfaktor zum Ruecknormieren
            sign:  (N, 3) komponentenweises Vorzeichen von points
        """
        r_max = radii.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        inp = torch.cat([radii / r_max, points.abs() / r_max], dim=-1)
        sign = torch.sign(points)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return inp, r_max.squeeze(-1), sign

    def predict_np(self, radii: np.ndarray, points: np.ndarray,
                   device: Optional[str] = None) -> np.ndarray:
        """
        Vektor-zur-Oberflaeche-Vorhersage fuer mehrere (Halbachsen, Punkt)-Paare.

        Punkte muessen bereits im lokalen Ellipsoidrahmen vorliegen
        (zentriert und de-rotiert).

        Args:
            radii:  (3,) oder (N, 3) float32 — Halbachsen
            points: (N, 3) float32 — Anfragepunkte
            device: torch-Geraet (Standard: Geraet des Netzes)

        Returns:
            (N, 3) float32 — Verschiebungsvektor q − p im Original-Oktanten
        """
        if device is None:
            device = next(self.parameters()).device
        points = np.asarray(points, np.float32)
        N = len(points)
        radii_np = np.broadcast_to(np.asarray(radii, np.float32), (N, 3))

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
            inp, r_max, sign = self.normalize(r_t, p_t)
            v_norm = self(inp)                                 # (N, 3) abs. Oktant
            v = v_norm * r_max.unsqueeze(-1) * sign            # (N, 3) Original
        return v.cpu().numpy()


# ── Trainer ───────────────────────────────────────────────────────────────────

class EllipsoidGradientTrainer:
    """
    Trainiert ein :class:`EllipsoidGradientNet` zur Vorhersage des
    Verschiebungsvektors q − p.

    Datengenerierung (identisch zum SDF-Trainer):
      - 20 % gleichmaessig in einer Box [−3·rmax, 3·rmax]³ (Aussenraum)
      - 50 % oberflaechennah: Ellipsoidpunkt + Gaussrauschen
      - 30 % im Inneren: gleichmaessig im Ellipsoidvolumen

    Verlust: L1 auf den drei Vektorkomponenten im positiven Oktanten.
    """

    def __init__(self, hidden: int = 128, depth: int = 3,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = EllipsoidGradientNet(hidden=hidden,
                                         depth=depth).to(self.device)

    # ── Datengenerierung ──────────────────────────────────────────────────

    def _generate_batch(self, batch_size: int, r_min: float,
                        r_max_val: float
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Erzeuge (Eingabe, Ziel)-Paare fuer einen Trainingsschritt.

        Returns:
            inp:    (N, 6) normierte Netzwerk-Eingabe
            target: (N, 3) normierter Verschiebungsvektor (positiver Oktant)
        """
        dev = self.device
        N = batch_size

        radii = torch.empty(N, 3, device=dev).uniform_(r_min, r_max_val)
        r_max_per = radii.max(dim=-1, keepdim=True).values    # (N, 1)

        n_ext  = int(0.2 * N)
        n_surf = int(0.5 * N)
        n_int  = N - n_ext - n_surf

        # 1. Aussenraum
        p_ext = (torch.rand(n_ext, 3, device=dev) * 2.0 - 1.0) \
                * 3.0 * r_max_per[:n_ext]

        # 2. Oberflaechennah
        theta = torch.acos(torch.empty(n_surf, device=dev).uniform_(-1.0, 1.0))
        phi   = torch.empty(n_surf, device=dev).uniform_(0.0, 2.0 * np.pi)
        unit  = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1)
        sigma = 0.15 * r_max_per[n_ext:n_ext + n_surf]
        p_surf = radii[n_ext:n_ext + n_surf] * unit \
                 + torch.randn(n_surf, 3, device=dev) * sigma

        # 3. Innenraum (gleichverteilt im Volumen)
        u_vol  = torch.rand(n_int, device=dev) ** (1.0 / 3.0)
        dir_in = torch.randn(n_int, 3, device=dev)
        dir_in = dir_in / dir_in.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        p_int  = radii[n_ext + n_surf:] * dir_in * u_vol.unsqueeze(-1)

        points = torch.cat([p_ext, p_surf, p_int], dim=0)     # (N, 3)

        # Ground Truth: Verschiebungsvektor im Original-Oktanten
        with torch.no_grad():
            vec = vec_to_surface_exact_torch(points, radii)   # (N, 3)

        # In den positiven Oktanten reflektieren — dort arbeitet das Netz.
        sign = torch.sign(points)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        vec_abs = vec * sign

        # Normieren
        inp    = torch.cat([radii / r_max_per,
                            points.abs() / r_max_per], dim=-1)
        target = vec_abs / r_max_per

        return inp, target

    # ── Trainingsschleife ─────────────────────────────────────────────────

    def train(self, n_steps: int = 50_000, batch_size: int = 32_768,
              r_min: float = 0.05, r_max: float = 1.0, lr: float = 1e-3,
              log_every: int = 500,
              save_path: Optional[str] = None) -> "EllipsoidGradientNet":
        """Standard-Trainingsschleife (CLI-Variante; UI-Variante: train_neural_gradient.py)."""
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
    def load(cls, path: str, device: Optional[str] = None
             ) -> "EllipsoidGradientTrainer":
        """Lade Trainer (inkl. Netz) von der Festplatte."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=True)
        trainer = cls(hidden=ckpt["hidden"], depth=ckpt["depth"], device=device)
        trainer.net.load_state_dict(ckpt["state_dict"])
        trainer.net.eval()
        return trainer
