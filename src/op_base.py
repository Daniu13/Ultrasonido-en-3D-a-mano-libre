import numpy as np

# OPERACIONES BASE PARA TODO EL PROYECTO

def wrap_angle(a):
        """Establecer angulo entre [-pi, pi).
        """
        return (a + np.pi) % (2 * np.pi) - np.pi

def pose_to_mat(p):
        """Una pose [x, y, theta] la vuelve matriz de
        transformación homogenea (R y t).
        """
        x, y, th = p
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, x], [s, c, y],
                         [0.0, 0.0, 1.0]])

def mat_to_pose(T):
        """Inversa de pose_to_mat
        """
        return np.array([T[0, 2], T[1, 2],
                         np.arctan2(T[1, 0], T[0, 0])])

def compose(a, b):
        """Composición entre 2 poses (Ta * Tb).
        """
        return mat_to_pose(pose_to_mat(a) @ pose_to_mat(b))

def transform_points(p, xy):
        """Aplica la matriz de transformacion (R + t)
        homogenea para los puntos en el grid.
        xy: (2, ...) en el marco de P (local)
        return [2, coordenadas x, coordenadas y]
        """
        x, y, th = p    # El punto
        c, s = np.cos(th), np.sin(th)
        return np.stack([x + c * xy[0] - s * xy[1],
                         y + s * xy[0] + c * xy[1]])

