import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class ProbeModel:
        R_c: float = 0.040
        depth: float = 0.080
        fov_deg: float = 60.0
        n_lines: int = 384
        dr: float = 5e-5
        f0: float = 3.0e6
        c: float = 1540.0
        frac_bw: float = 0.6
        f_number: float = 4.0
        focus: float = 0.035
        img_pix: float = 2e-4

        def __post_init__(self):
                self.half_fov = np.deg2rad(self.fov_deg) / 2

                self.n_r = int(round(self.depth / self.dr))
                self.r = (self.R_c
                        + self.dr*np.arange(self.n_r))
                self.r_max = self.r[-1]

                self.phi = np.linspace(-self.half_fov,
                            self.half_fov, self.n_lines)
                self.dphi = self.phi[1] - self.phi[0]

                RR, PP = np.meshgrid(self.t, self.phi,
                                     indexing="ij")

                self.xp = (RR*np.cos(PP)).astype(np.float32)
                self.yp = (RR*np.sin(PP)).astype(np.float32)
                self._build_scan_conversion()

        def _build_scan_conversion(self):
                h, p = self.half_fov, self.img_pix

                self.x_min = self.R_c * np.cos(h)
                self.x_max = self.r_max
                self.y_max = self.r_max * np.sin(h)

                n_rows = int(np.ceil((self.x_max
                            - self.x_min) / p))
                n_cols = int(np.ceil(2 * self.y_max / p))

                xs = self.x_min + (np.arange(n_rows + 0.5))*p
                ys = -self.y_max + (np.arange(n_cols) + 0.5)*p

                X, Y = np.meshgrid(xs, ys, indexing="ij")

                rho, phi = np.hypot(X, Y), np.arctan2(Y, X)
                self.map_x = ((phi + h) / self.phi).astype(np.float32)
                self.map_y = ((rho - self.R_c) / self.dr).astype(np.float32)
                self.fan_mask = ((rho >= self.R_c)
                            & (rho <= self.r_max)
                            & (np.abs(phi <= h)))

        def scan_convert(self, polar):
            img = cv2.remap(
                   polar.astype(np.float32),
                   self.map_x,
                   self.map_y,
                   cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT,
                   borderValue=0,
            )
            return np.where(
                   self.fan_mask,
                   img,
                   0
            ).astype(np.float32)