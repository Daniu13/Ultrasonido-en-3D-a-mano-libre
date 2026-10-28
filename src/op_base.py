import numpy as np

# OPERACIONES BASE PARA TODO EL PROYECTO

def wrap_angle(a):
        """Establecer angulo entre [-pi, pi).
        """
        return (a + np.pi) % (2 * np.pi) - np.pi

def transform_points(p, xy):
        """Aplica la matriz de transformacion (R + t)
        homogenea para los puntos en el grid.
        xy: (2, ...) en el marco de P (local)
        """
        x, y, th = p    # El punto
        c, s = np.cos(th), np.sin(th)
        return np.stack([x + c * xy[0] - s * xy[1],
                         y + s * xy[0] + c * xy[1]])