import numpy as np
import matplotlib.pyplot as plt

class MobiusStrip:
    def __init__(self, R, w, n):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._mesh()

    def _mesh(self):
        u, v = self.U, self.V
        r = self.R
        x = (r + v * np.cos(u / 2)) * np.cos(u)
        y = (r + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def surface_area(self):
        du = (2 * np.pi) / (self.n - 1)
        dv = self.w / (self.n - 1)

        dxu = np.gradient(self.X, axis=1) / du
        dyu = np.gradient(self.Y, axis=1) / du
        dzu = np.gradient(self.Z, axis=1) / du

        dxv = np.gradient(self.X, axis=0) / dv
        dyv = np.gradient(self.Y, axis=0) / dv
        dzv = np.gradient(self.Z, axis=0) / dv

        cx = dyu * dzv - dzu * dyv
        cy = dzu * dxv - dxu * dzv
        cz = dxu * dyv - dyu * dxv

        dA = np.sqrt(cx**2 + cy**2 + cz**2)
        return np.sum(dA) * du * dv

    def edge_length(self):
        def edge(sign):
            u = self.u
            x = (self.R + sign * self.w / 2 * np.cos(u / 2)) * np.cos(u)
            y = (self.R + sign * self.w / 2 * np.cos(u / 2)) * np.sin(u)
            z = sign * self.w / 2 * np.sin(u / 2)
            return np.vstack([x, y, z])

        def length(e):
            d = np.diff(e, axis=1)
            return np.sum(np.linalg.norm(d, axis=0))

        return length(edge(1)) + length(edge(-1))

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, color='cyan', edgecolor='gray', alpha=0.8)
        plt.show()

if __name__ == '__main__':
    m = MobiusStrip(R=5, w=1, n=200)
    print(f"Area: {m.surface_area():.4f}")
    print(f"Edge Length: {m.edge_length():.4f}")
    m.plot()
