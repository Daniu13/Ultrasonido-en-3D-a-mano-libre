import numpy as np
import cv2
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

GEL, SKIN, FAT, MUSCLE, FASCIA, CORTEX, MARROW, BLOOD, MASS = range(9)

# (Z [MRayl], alpha [dB/(cm*MHz)], sigma_s)
TISSUE_TABLE = {
        GEL: (1.50, 0.00, 0.00),
        SKIN: (1.99, 1.50, 0.60),
        FAT: (1.38, 0.60, 0.20),
        MUSCLE: (1.70, 1.00, 0.35),
        FASCIA: (1.85, 1.00, 0.90),
        CORTEX: (7.80, 20.0, 0.10),
        MARROW: (1.50, 1.00, 0.15),
        BLOOD: (1.65, 0.15, 0.02),
        MASS: (1.75, 1.20, 0.90),
}
GEL_LNZ = float(np.log(TISSUE_TABLE[GEL][0]))

@dataclass
class Phantom:
        dx: float           # Metros por px           
        x0: float           # x (W) de centro del px (0,0)
        y0: float           # ¨¨¨¨
        labels: np.ndarray
        lnZ: np.ndarray
        alpha: np.ndarray
        scat: np.ndarray    # Campo de dispersores

def make_limb_phantom(size=0.12, dx=5e-5, seed=5):
        rng = np.random.default_rng(seed)
        n = int(round(size / dx))
        x0, y0 = -size / 2, size / 2

        # x = x0 + columnas*dx
        # y = y0 - filas*dx (mismo size pq cuadrado)
        X = (x0 + np.arange(n)*dx).astype(np.float32)[None,:] 
        Y = (y0 - np.arange(n)*dx).astype(np.float32)[:,None]

        a, b = 0.05, 0.04
        rho = np.sqrt((X / a)**2 + (Y / b)**2)
        ang = np.arctan2(Y / b, X / a)
        labels = np.zeros((n, n), np.uint8)
        labels[rho <= 1.0] = SKIN
        labels[rho <= 0.97] = FAT
        fat_inner = (0.8 + 0.06*np.cos(ang - 0.8)
                    + 0.03*np.cos(2*ang + 0.3))
        labels[rho <= fat_inner] = MUSCLE

        def disc(cx, cy, r):
                return (X - cx)**2 + (Y - cy)**2 <= r**2

        def seg_dist(p0, p1):
                p0 = np.array(p0, np.float32)
                d = np.array(p1, np.float32) - p0
                t = np.clip((X - p0[0])*d[0]
                            + (Y - p0[1]*d[1] / (d @ d), 0, 1))
                return np.hypot(X - (p0 + t*d[0]),
                Y - (p0[1] + t*d[1]))

        for p0, p1 in [((-0.035, 0.020), (0.010, -0.028)),
                       ((0.005, 0.030), (0.030, -0.015))]:
                labels[(seg_dist(p0, p1) <= 3e-4) & (labels == MUSCLE)] = FASCIA

        for cx, cy, r in [(-0.008, 0.002, 0.009),
                          (0.016, -0.004, 0.005)]:
                labels[disc(cx, cy, r)] = CORTEX
                labels[disc(cx, cy, r - 0.0025)] = MARROW

        for cx, cy, r in [(0.022, 0.012, 0.003),
                          (0.028, 0.004, 0.004),
                          (-0.025, -0.015, 0.0025)]:
                labels[disc(cx, cy, r)] = BLOOD
        labels[disc(-0.020, 0.015, 0.0035)] = MASS

        K = len(TISSUE_TABLE)
        Z = np.array([TISSUE_TABLE[k][0] for k in range(K)],
                        np.float32)
        AL = np.array([TISSUE_TABLE[k][1] for k in range(K)],
                        np.float32)
        SG = np.array([TISSUE_TABLE[k][2] for k in range(K)],
                        np.float32)

        m = int(round(size / 1e-3))     # textura suave a 1 mm/px
        g = gaussian_filter(rng.standard_normal((m, m)), 3.0)
        g = ((g - g.mean()) / g.std()).astype(np.float32)
        g = cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC)
        sigma_map = SG[labels] * np.exp(0.35*g)
        scat = sigma_map * rng.standard_normal((n, n), dtype=np.float32)

        return Phantom(dx, x0, y0, labels, np.log(Z)[labels],
                       AL[labels], scat.astype(np.float32))