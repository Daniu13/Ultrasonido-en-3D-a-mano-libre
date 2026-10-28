import numpy as np

# OPERACIONES BASE PARA TODO EL PROYECTO
# RECORDAR RE-CHECKEAR CADA FUNCION EN MI LIBRETA DE APUNTES

def wrap_angle(a):
        """Establecer angulo entre [-pi, pi).
        """
        return (a + np.pi) % (2*np.pi) - np.pi

def pose_to_mat(p):
        """Una pose [x, y, theta] la vuelve matriz de
        transformación homogenea (R y t).
        """
        x, y, th = p
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, x],
                         [s, c, y],
                         [0.0, 0.0, 1.0]])

def mat_to_pose(T):
        """Inversa de pose_to_mat
        """
        return np.array([T[0, 2],
                         T[1, 2],
                         np.arctan2(T[1, 0], T[0, 0])])

def compose(a, b):
        """Composición entre 2 poses (Ta * Tb).
        """
        return mat_to_pose(pose_to_mat(a) @ pose_to_mat(b))

def inverse(p):
        """Inversa de una Pose, P -> W.
        """
        x, y, th = p
        c, s = np.cos(th), np.sin(th)
        return np.array([-(c*x + s*y),
                         -(-s*x + c*y),
                         wrap_angle(-th)])

def relative(a, b):
        # Determina lo que mide la odometría
        # o transformación relativa
        return compose(inverse(a), b)

def transform_points(p, xy):
        """Aplica la matriz de transformacion (R + t)
        homogenea para los puntos en el grid.
        xy: (2, ...) en el marco de P (local)
        return [2, coordenadas x, coordenadas y]
        """
        x, y, th = p    # El punto
        c, s = np.cos(th), np.sin(th)
        return np.stack([x + c*xy[0] - s*xy[1],
                         y + s*xy[0] + c*xy[1]])

def _ab(w):
        """Calcula los coeficientes a, b usados en la
        matriz de curvatura (V) generada por omega (w).
        """
        if abs(w) < 1e-3:
                return 1 - w**2 / 6, w / 2 - w**3 / 24
        return np.sin(w) / w, (1 - np.cos(w)) / w

def exp_map(xi):
        vx, vy, w = xi
        a, b = _ab(w)
        return np.array([a*vx - b*vy,
                         b*vx + a*vy,
                         wrap_angle(w)])

def log_map(p):
        x, y, th = p
        w = wrap_angle(th)
        a, b = _ab(w)

        d = a**2 + b**2
        return np.array([(a*x + b*y) / d,
                         (-b*x + a*y) / d, w])

if __name__ == "__main__":
        # PRUEBAS
        rng = np.random.default_rng(0)
        for _ in range(2000):
                p = np.array([*rng.normal(size=2), rng.uniform(-3, 3)])
                q = np.array([*rng.normal(size=2), rng.uniform(-3, 3)])
                assert np.allclose(compose(p, inverse(p)), 0, atol=1e-12)
                assert np.allclose(exp_map(log_map(p)), p, atol=1e-9)
                assert np.allclose(compose(p, relative(p, q))[:2],
                                   q[:2], atol=1e-9)
        print("operaciones OK")
        