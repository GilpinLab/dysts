# type: ignore
"""
Various low-dimensional dynamical systems in Python.
For flows that occur on unbounded intervals (eg non-autonomous systems),
coordinates are transformed to a basis where the domain remains bounded
"""

import warnings

import numpy as np

from .base import DynSys, staticjit
from .utils.native_utils import _REMOVED_DELAY_SYSTEMS, _removed_delay_error


class Lorenz(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, rho, sigma):
        xdot = sigma * y - sigma * x
        ydot = rho * x - x * z - y
        zdot = x * y - beta * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, beta, rho, sigma):
        row1 = [-sigma, sigma, 0]
        row2 = [rho - z, -1, -x]
        row3 = [y, x, -beta]
        return row1, row2, row3


class LorenzBounded(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, r, rho, sigma):
        xdot = (
            sigma * y
            - sigma * x
            - sigma / r**2 * y * x**2
            - sigma / r**2 * y**3
            - sigma / r**2 * y * z**2
            + sigma / r**2 * x**3
            + sigma / r**2 * x * y**2
            + sigma / r**2 * x * z**2
        )
        ydot = (
            rho * x
            - x * z
            - y
            - rho / r**2 * x**3
            - rho / r**2 * x * y**2
            - rho / r**2 * x * z**2
            + 1 / r**2 * z * x**3
            + 1 / r**2 * x * z * y**2
            + 1 / r**2 * x * z**3
            + 1 / r**2 * y * x**2
            + 1 / r**2 * y**3
            + 1 / r**2 * y * z**2
        )
        zdot = (
            x * y
            - beta * z
            - 1 / r**2 * y * x**3
            - 1 / r**2 * x * y**3
            - 1 / r**2 * x * y * z**2
            + beta / r**2 * z * x**2
            + beta / r**2 * z * y**2
            + beta / r**2 * z**3
        )
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, beta, r, rho, sigma):
        c0 = y**2
        c1 = x**2
        c2 = 3 * c1
        c3 = 2 * y
        c4 = c3 * x
        c5 = z**2
        c6 = r**2
        c7 = c5 - c6
        c8 = -c4 + c7
        c9 = c6 ** (-1.0)
        c10 = c9 * sigma
        c11 = 3 * c0
        c12 = c1 + c11
        c13 = 2 * z
        c14 = c4 * z
        c15 = c5 * x
        c16 = -c6 * x + x**3
        row1 = [c10 * (c0 + c2 + c8), c10 * (-c12 - c8), c10 * c13 * (x - y)]
        row2 = [
            c9
            * (
                -c0 * rho
                + c0 * z
                - c2 * rho
                + c2 * z
                + c4
                - c5 * rho
                + c6 * (rho - z)
                + z**3
            ),
            c9 * (c12 + c14 - c4 * rho + c7),
            c9 * (c0 * x - c13 * rho * x + 3 * c15 + c16 + c3 * z),
        ]
        row3 = [
            c9 * (2 * beta * x * z - c2 * y - c5 * y + c6 * y - y**3),
            c9 * (2 * beta * y * z - c11 * x - c15 - c16),
            c9 * (beta * c0 + beta * c1 + 3 * beta * c5 - beta * c6 - c14),
        ]
        return row1, row2, row3


class LorenzCoupled(DynSys):
    @staticjit
    def _rhs(x1, y1, z1, x2, y2, z2, t, beta, eps, rho, rho1, rho2, sigma):
        x1dot = sigma * y1 - sigma * x1
        y1dot = rho1 * x1 - x1 * z1 - y1
        z1dot = x1 * y1 - beta * z1
        x2dot = sigma * y2 - sigma * x2 + eps * x1 - eps * x2
        y2dot = rho2 * x2 - x2 * z2 - y2
        z2dot = x2 * y2 - beta * z2
        return x1dot, y1dot, z1dot, x2dot, y2dot, z2dot

    @staticjit
    def _jac(x1, y1, z1, x2, y2, z2, t, beta, eps, rho, rho1, rho2, sigma):
        row1 = [-sigma, sigma, 0, 0, 0, 0]
        row2 = [rho1 - z1, -1, -x1, 0, 0, 0]
        row3 = [y1, x1, -beta, 0, 0, 0]
        row4 = [eps, 0, 0, -eps - sigma, sigma, 0]
        row5 = [0, 0, 0, rho2 - z2, -1, -x2]
        row6 = [0, 0, 0, y2, x2, -beta]
        return row1, row2, row3, row4, row5, row6


class Lorenz96(DynSys):
    def rhs(self, X, t):
        Xdot = np.zeros_like(X)
        Xdot[0] = (X[1] - X[-2]) * X[-1] - X[0] + self.f
        Xdot[1] = (X[2] - X[-1]) * X[0] - X[1] + self.f
        Xdot[-1] = (X[0] - X[-3]) * X[-2] - X[-1] + self.f
        Xdot[2:-1] = (X[3:] - X[:-3]) * X[1:-2] - X[2:-1] + self.f
        return Xdot


class Lorenz84(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, f, g):
        xdot = -a * x - y**2 - z**2 + a * f
        ydot = -y + x * y - b * x * z + g
        zdot = -z + b * x * y + x * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, f, g):
        row1 = [-a, -2 * y, -2 * z]
        row2 = [y - b * z, x - 1, -b * x]
        row3 = [b * y + z, b * x, -1 + x]
        return row1, row2, row3


class Rossler(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = -y - z
        ydot = x + a * y
        zdot = b + z * x - c * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [0, -1, -1]
        row2 = [1, a, 0]
        row3 = [z, 0, x - c]
        return row1, row2, row3


class Thomas(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a * x + b * np.sin(y)
        ydot = -a * y + b * np.sin(z)
        zdot = -a * z + b * np.sin(x)
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [-a, b * np.cos(y), 0]
        row2 = [0, -a, b * np.cos(z)]
        row3 = [b * np.cos(x), 0, -a]
        return row1, row2, row3


class ThomasLabyrinth(Thomas):
    pass


class DoublePendulum(DynSys):
    @staticjit
    def _rhs(th1, th2, p1, p2, t, g, l1, l2, m1, m2):
        cos_diff = np.cos(th1 - th2)
        sin_diff = np.sin(th1 - th2)
        denom = l1 * l2 * (m1 + m2 * sin_diff**2)

        th1dot = (l2 * p1 - l1 * p2 * cos_diff) / (l1 * denom)
        th2dot = ((m1 + m2) * l1 * p2 - m2 * l2 * p1 * cos_diff) / (m2 * l2 * denom)

        h1 = p1 * p2 * sin_diff / denom
        h2 = (m2 * l2 * p1**2) / (2 * l1 * denom**2)
        h2 += 0.5 * m2 * p2 * l2 * l1 * th2dot / denom
        h2 *= np.sin(2 * (th1 - th2))

        p1dot = -(m1 + m2) * g * l1 * np.sin(th1) - h1 + h2
        p2dot = -m2 * g * l2 * np.sin(th2) + h1 - h2

        return th1dot, th2dot, p1dot, p2dot

    @staticmethod
    @staticjit
    def _postprocessing(th1, th2, p1, p2):
        return np.sin(th1), np.sin(th2), p1, p2

    @staticjit
    def _jac(th1, th2, p1, p2, t, g, l1, l2, m1, m2):
        c0 = th1 - th2
        c1 = np.sin(c0)
        c2 = c1**2 * m2
        c3 = c2 + m1
        c4 = l1 * p2
        c5 = l2 * p1
        c6 = np.cos(c0)
        c7 = 2 * c6
        c8 = c3 * c4 + c7 * m2 * (c4 * c6 - c5)
        c9 = l1**2
        c10 = c9 ** (-1.0)
        c11 = l2 ** (-1.0)
        c12 = c3 ** (-2.0)
        c13 = c11 * c12
        c14 = c1 * c10 * c13
        c15 = c3 ** (-1.0)
        c16 = l1 ** (-1.0)
        c17 = c11 * c16
        c18 = c15 * c17 * c6
        c19 = -c18
        c20 = m1 + m2
        c21 = c20 * c4
        c22 = c6 * m2
        c23 = -c22 * c5
        c24 = c21 + c23
        c25 = c3 * c5
        c26 = -c24 * c7 + c25
        c27 = l2 ** (-2.0)
        c28 = c16 * c27
        c29 = c12 * c28
        c30 = c1 * c29
        c31 = l1 ** (-3.0)
        c32 = p1**2
        c33 = c1 / c3**3
        c34 = (
            -0.5 * c1 * c11 * c12 * c16 * m2 * p1 * p2
            + c11 * c31 * c32 * c33 * c7 * m2**2
            + 2.0 * c22 * c24 * c28 * c33 * p2
        )
        c35 = 2 * th1 - 2 * th2
        c36 = np.sin(c35)
        c37 = c13 * c31
        c38 = 2 * (0.5 * c24 * c29 * p2 + (1 / 2) * c32 * c37 * m2) * np.cos(c35)
        c39 = p1 * p2
        c40 = c18 * c39
        c41 = c12 * c17 * c2 * c39 * c7
        c42 = c38 - c40 + c41
        c43 = -c34 * c36 + c42
        c44 = c34 * c36
        c45 = c9 * p2
        c46 = c1 * c3 * c45 + c36 * m2 * (0.5 * c45 * c6 - p1)
        c47 = c1 * c25 - 0.5 * c36 * (2 * c21 + c23)
        row1 = [c14 * c8, -c14 * c8, c10 * c15, c19]
        row2 = [c26 * c30, -c26 * c30, c19, c15 * c20 * c27 / m2]
        row3 = [-c20 * g * l1 * np.cos(th1) + c43, -c42 + c44, -c37 * c46, -c29 * c47]
        row4 = [
            -c43,
            c38 - c40 + c41 - c44 - g * l2 * m2 * np.cos(th2),
            c37 * c46,
            c29 * c47,
        ]
        return row1, row2, row3, row4


class SwingingAtwood(DynSys):
    @staticjit
    def _rhs(r, th, pr, pth, t, m1, m2):
        g = 9.82
        rdot = pr / (m1 + m2)
        thdot = pth / (m1 * r**2)
        prdot = pth**2 / (m1 * r**3) - m2 * g + m1 * g * np.cos(th)
        pthdot = -m1 * g * r * np.sin(th)
        return rdot, thdot, prdot, pthdot

    @staticjit
    def _postprocessing(r, th, pr, pth):
        return r, np.sin(th), pr, pth

    @staticjit
    def _jac(r, th, pr, pth, t, m1, m2):
        row1 = [0, 0, (m1 + m2) ** (-1.0), 0]
        row2 = [-2 * pth / (m1 * r**3), 0, 0, 1 / (m1 * r**2)]
        row3 = [
            -3 * pth**2 / (m1 * r**4),
            -9.82 * m1 * np.sin(th),
            0,
            2 * pth / (m1 * r**3),
        ]
        row4 = [-9.82 * m1 * np.sin(th), -9.82 * m1 * r * np.cos(th), 0, 0]
        return row1, row2, row3, row4


class GlycolyticOscillation(DynSys):
    @staticjit
    def _rhs(a, b, c, t, d, k, l1, l2, nu, q1, q2, s1, s2):
        phi = (a * (1 + a) * (1 + b) ** 2) / (l1 + (1 + a) ** 2 * (1 + b) ** 2)
        eta = b * (1 + d * b) * (1 + c) ** 2 / (l2 + (1 + d * b) ** 2 * (1 + c) ** 2)
        adot = nu - s1 * phi
        bdot = q1 * s1 * phi - s2 * eta
        cdot = q2 * s2 * eta - k * c
        return adot, bdot, cdot

    @staticjit
    def _jac(a, b, c, t, d, k, l1, l2, nu, q1, q2, s1, s2):
        c0 = 2 * a
        c1 = a + 1
        c2 = b + 1
        c3 = c2**2
        c4 = c1**2 * c3
        c5 = c4 + l1
        c6 = c0 * c4 - c5 * (c0 + 1)
        c7 = s1 / c5**2
        c8 = c3 * c7
        c9 = c8 * q1
        c10 = b * d
        c11 = c10 + 1
        c12 = c + 1
        c13 = c12**2
        c14 = c11**2
        c15 = c13 * c14
        c16 = c15 + l2
        c17 = c16 ** (-1.0)
        c18 = c13 * s2
        c19 = c17 * c18
        c20 = 2 * b
        c21 = c20 + 2
        c22 = c16 ** (-2.0)
        c23 = 2 * c10
        c24 = c18 * c22 * q2
        c25 = 2 * c + 2
        row1 = [c6 * c8, -c0 * c1 * c2 * c7 * l1, 0]
        row2 = [
            -c6 * c9,
            -a * c1**3 * c21 * c9
            + a * c1 * c21 * q1 * s1 / c5
            + 2 * b * c12**4 * c14 * c22 * d * s2
            - c10 * c19
            - c11 * c19,
            -c11 * c12 * c20 * c22 * l2 * s2,
        ]
        row3 = [
            0,
            c24 * (-c15 * c23 + c16 * (c23 + 1)),
            -b * c11**3 * c24 * c25 + b * c11 * c17 * c25 * q2 * s2 - k,
        ]
        return row1, row2, row3


class GuckenheimerHolmes(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, e, f):
        xdot = a * x - b * y + c * z * x + d * z * x**2 + d * z * y**2
        ydot = a * y + b * x + c * z * y
        zdot = e - z**2 - f * x**2 - f * y**2 - a * z**3
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d, e, f):
        row1 = [
            a + c * z + 2 * d * x * z,
            -b + 2 * d * y * z,
            c * x + d * x**2 + d * y**2,
        ]
        row2 = [b, a + c * z, c * y]
        row3 = [-2 * f * x, -2 * f * y, z * (-3 * a * z - 2)]
        return row1, row2, row3


class HenonHeiles(DynSys):
    @staticjit
    def _rhs(x, y, px, py, t, lam):
        xdot = px
        ydot = py
        pxdot = -x - 2 * lam * x * y
        pydot = -y - lam * x**2 + lam * y**2
        return xdot, ydot, pxdot, pydot

    @staticjit
    def _jac(x, y, px, py, t, lam):
        row1 = [0, 0, 1, 0]
        row2 = [0, 0, 0, 1]
        row3 = [-1 - 2 * lam * y, -2 * lam * x, 0, 0]
        row4 = [-2 * lam * x, 2 * lam * y - 1, 0, 0]
        return row1, row2, row3, row4


class Halvorsen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a * x - b * y - b * z - y**2
        ydot = -a * y - b * z - b * x - z**2
        zdot = -a * z - b * x - b * y - x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [-a, -b - 2 * y, -b]
        row2 = [-b, -a, -b - 2 * z]
        row3 = [-b - 2 * x, -b, -a]
        return row1, row2, row3


class Chua(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, m0, m1):
        ramp_x = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
        xdot = alpha * (y - x - ramp_x)
        ydot = x - y + z
        zdot = -beta * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, beta, m0, m1):
        dramp_xdx = m1 + 0.5 * (m0 - m1) * (np.sign(x + 1) - np.sign(x - 1))
        row1 = [-alpha - alpha * dramp_xdx, alpha, 0]
        row2 = [1, -1, 1]
        row3 = [0, -beta, 0]
        return row1, row2, row3


class MultiChua(DynSys):
    def diode(self, x):
        m, c = self.m, self.c
        total = m[-1] * x
        for i in range(1, 6):
            total += 0.5 * (m[i - 1] - m[i]) * (np.abs(x + c[i]) - np.abs(x - c[i]))
        return total

    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a * (y - self.diode(x))
        ydot = x - y + z
        zdot = -self.b * y
        return (xdot, ydot, zdot)


class Duffing(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, delta, gamma, omega):
        xdot = y
        ydot = -delta * y - beta * x - alpha * x**3 + gamma * np.cos(z)
        zdot = omega
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, beta, delta, gamma, omega):
        row1 = [0, 1, 0]
        row2 = [-3 * alpha * x**2 - beta, -delta, -gamma * np.sin(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.cos(z)


class DoubleGyre(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, eps, omega):
        a = eps * np.sin(z)
        b = 1 - 2 * eps * np.sin(z)
        f = a * x**2 + b * x
        dx = -alpha * np.pi * np.sin(np.pi * f) * np.cos(np.pi * y)
        dy = alpha * np.pi * np.cos(np.pi * f) * np.sin(np.pi * y) * (2 * a * x + b)
        dz = omega
        return dx, dy, dz

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)

    @staticjit
    def _jac(x, y, z, t, alpha, eps, omega):
        c0 = 2 * x
        c1 = eps * np.sin(z)
        c2 = 2 * c1
        c3 = 1 - c2
        c4 = c0 * c1 + c3
        c5 = np.pi * x
        c6 = c5 * (c1 * x + c3)
        c7 = np.cos(c6)
        c8 = np.pi**2
        c9 = np.pi * y
        c10 = alpha * c7 * c8 * np.cos(c9)
        c11 = np.sin(c6)
        c12 = alpha * np.sin(c9)
        c13 = x - 2
        c14 = eps * np.cos(z)
        c15 = np.pi * c12
        row1 = [-c10 * c4, c11 * c12 * c8, -c10 * c13 * c14 * x]
        row2 = [
            c15 * (-np.pi * c11 * c4**2 + c2 * c7),
            c10 * c4,
            c14 * c15 * (-c11 * c13 * c4 * c5 + c7 * (c0 - 2)),
        ]
        row3 = [0, 0, 0]
        return row1, row2, row3


class BlinkingRotlet(DynSys):
    @staticjit
    def _rotlet(r, theta, a, b, bc):
        """A rotlet velocity field"""
        kappa = a**2 + (b**2 * r**2) / a**2 - 2 * b * r * np.cos(theta)
        gamma = (1 - r**2 / a**2) * (a**2 - (b**2 * r**2) / a**2)
        iota = (b**2 * r) / a**2 - b * np.cos(theta)
        zeta = b**2 + r**2 - 2 * b * r * np.cos(theta)
        nu = a**2 + b**2 - (2 * b**2 * r**2) / a**2
        vr = b * np.sin(theta) * (-bc * (gamma / kappa**2) - 1 / kappa + 1 / zeta)
        vth = (
            bc * (gamma * iota) / kappa**2
            + bc * r * nu / (a**2 * kappa)
            + iota / kappa
            - (r - b * np.cos(theta)) / zeta
        )
        return vr, vth

    @staticjit
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * np.tanh(tau * stiffness * np.sin(2 * np.pi * t / tau))

    def rhs(self, X, t):
        r, theta, tt = X
        weight = self._protocol(tt, self.tau)
        dr1, dth1 = self._rotlet(r, theta, self.a, self.b, self.bc)
        dr2, dth2 = self._rotlet(r, theta, self.a, -self.b, self.bc)
        dr = weight * dr1 + (1 - weight) * dr2
        dth = (weight * dth1 + (1 - weight) * dth2) / r
        dtt = 1
        return self.sigma * dr, self.sigma * dth, dtt

    def _postprocessing(self, r, th, tt):
        return r * np.cos(th), r * np.sin(th), np.sin(2 * np.pi * tt / self.tau)


class LidDrivenCavityFlow(DynSys):
    @staticjit
    def _lid(x, y, a, b, tau, u1, u2):
        """The velocity field when the left domain drives"""
        prefactor1 = (
            2
            * u1
            * np.sin(np.pi * x / a)
            / (2 * b * np.pi + a * np.sinh(2 * np.pi * b / a))
        )
        prefactor2 = (
            2
            * u2
            * np.sin(2 * np.pi * x / a)
            / (4 * b * np.pi + a * np.sinh(4 * np.pi * b / a))
        )
        vx1 = -b * np.pi * np.sinh(np.pi * b / a) * np.sinh(np.pi * y / a) + np.cosh(
            np.pi * b / a
        ) * (np.pi * y * np.cosh(np.pi * y / a) + a * np.sinh(np.pi * y / a))
        vx2 = -2 * b * np.pi * np.sinh(2 * np.pi * b / a) * np.sinh(
            2 * np.pi * y / a
        ) + np.cosh(2 * np.pi * b / a) * (
            2 * np.pi * y * np.cosh(2 * np.pi * y / a) + a * np.sinh(2 * np.pi * y / a)
        )
        vx = prefactor1 * vx1 + prefactor2 * vx2

        prefactor1 = (
            2
            * np.pi
            * u1
            * np.cos(np.pi * x / a)
            / (2 * b * np.pi + a * np.sinh(2 * np.pi * b / a))
        )
        prefactor2 = (
            4
            * np.pi
            * u2
            * np.cos(2 * np.pi * x / a)
            / (4 * b * np.pi + a * np.sinh(4 * np.pi * b / a))
        )
        vy1 = b * np.sinh(np.pi * b / a) * np.cosh(np.pi * y / a) - np.cosh(
            np.pi * b / a
        ) * y * np.sinh(np.pi * y / a)
        vy2 = b * np.sinh(2 * np.pi * b / a) * np.cosh(2 * np.pi * y / a) - np.cosh(
            2 * np.pi * b / a
        ) * y * np.sinh(2 * np.pi * y / a)
        vy = prefactor1 * vy1 + prefactor2 * vy2

        # vy1 = b * np.sinh(np.pi * b / a) * np.cosh(np.pi * y / a) - np.cosh(np.pi * b / a) * y * np.sinh(np.pi * y / a)
        # vy2 = b * np.sinh(2 * np.pi * b / a) * np.cosh(2 * np.pi * y / a) - np.cosh(2 * np.pi * b / a) * y * np.sinh(2 * np.pi * y / a)
        # vy = np.pi * prefactor1 * vy1 + 2 * np.pi * prefactor2 * vy2

        return vx, vy

    # @staticjit
    # def _right(x, y, a, b, tau, u1, u2):
    #     """The velocity field when the right domain drives"""
    #     prefactor1 = 2 * u1 * np.sin(np.pi * x / a) / (2 * b * np.pi + a * np.sinh(2 * np.pi * b / a))
    #     prefactor2 = 2 * u2 * np.sin(2 * np.pi * x / a) / (4 * b * np.pi + a * np.sinh(4 * np.pi * b / a))
    #     vx1 = -b * np.pi * np.sinh(np.pi * b / a) * np.sinh(np.pi * y / a) - np.cosh(np.pi * b / a) * (np.pi * y * np.cosh(np.pi * y /a) + a * np.sinh(np.pi * y /a))
    #     vx2 = -4 * b * np.pi * np.sinh(2 * np.pi * b / a) * np.sinh(2 * np.pi * y / a) - np.cosh(2 * np.pi * b / a) * (2 * np.pi * y * np.cosh(2 * np.pi * y /a) + a * np.sinh(2 * np.pi * y /a))
    #     vx = prefactor1 * vx1 - prefactor2 * vx2

    #     prefactor1 = 2 * np.pi * u1 * np.cos(np.pi * x / a) / (2 * b * np.pi + a * np.sinh(2 * np.pi * b / a))
    #     prefactor2 = 4 * np.pi * u2 * np.cos(2 * np.pi * x / a) / (4 * b * np.pi + a * np.sinh(4 * np.pi * b / a))
    #     vy1 = -b * np.sinh(np.pi * b / a) * np.cosh(np.pi * y / a) + np.cosh(np.pi * b / a) * y * np.sinh(np.pi * y / a)
    #     vy2 = -2 * b * np.sinh(2 * np.pi * b / a) * np.cosh(2 * np.pi * y / a) + np.cosh(2 * np.pi * b / a) * 2 * y * np.sinh(2 * np.pi * y / a)
    #     vy = prefactor1 * vy1 + prefactor2 * vy2

    #     return vx, vy

    @staticjit
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * np.tanh(tau * stiffness * np.sin(2 * np.pi * t / tau))

    def rhs(self, X, t):
        x, y, tt = X
        weight = self._protocol(tt, self.tau)
        dx1, dy1 = self._lid(x, y, self.a, self.b, self.tau, self.u1, self.u2)
        dx2, dy2 = self._lid(x, y, self.a, self.b, self.tau, -self.u1, self.u2)
        dx = weight * dx1 + (1 - weight) * dx2
        dy = weight * dy1 + (1 - weight) * dy2
        dtt = 1
        return dx, dy, dtt

    def _postprocessing(self, x, y, tt):
        return x, y, np.sin(2 * np.pi * tt / self.tau)


class BlinkingVortex(BlinkingRotlet):
    pass


class InteriorSquirmer(DynSys):
    @staticjit
    def _rhs_static(r, th, t, a, g):
        n = a.shape[0]
        nvals = np.arange(1, n + 1)
        sinvals, cosvals = np.sin(th * nvals), np.cos(th * nvals)
        rnvals = r**nvals

        vrn = g * cosvals + a * sinvals
        vrn *= (nvals * rnvals * (r**2 - 1)) / r

        vth = 2 * r + (r**2 - 1) * nvals / r
        vth *= a * cosvals - g * sinvals
        vth *= rnvals

        return np.sum(vrn), np.sum(vth) / r

    @staticjit
    def _jac_static(r, th, t, a, g):
        n = a.shape[0]
        nvals = np.arange(1, n + 1)
        sinvals, cosvals = np.sin(th * nvals), np.cos(th * nvals)
        rnvals = r**nvals
        trigsum = a * sinvals + g * cosvals
        trigskew = a * cosvals - g * sinvals

        j11 = np.copy(trigsum)
        j11 *= nvals * rnvals * (2 * r**2 + (r**2 - 1) * (nvals - 1))
        j11 = (1 / r**2) * np.sum(j11)

        j12 = np.copy(trigskew)
        j12 *= -(nvals**2) * rnvals * (1 - r**2) / r
        j12 = np.sum(j12)

        j21 = 2 * rnvals * (2 * nvals + 1) * (-np.copy(trigskew))
        j21 += (n * (1 - r**2) * rnvals * (nvals - 1) / r**2) * np.copy(
            g * sinvals + a * cosvals
        )
        j21 = -np.sum(j21)

        j22 = np.copy(trigsum)
        j22 *= -nvals * rnvals * (2 * r + (r**2 - 1) * nvals / r)
        j22 = np.sum(j22)
        # (1 / r**2) *

        ## Correct for polar coordinates
        vth = np.copy(trigskew)
        vth *= 2 * r + (r**2 - 1) * nvals / r
        vth *= rnvals
        vth = np.sum(vth) / r
        j21 = j21 / r - vth / r
        j22 /= r

        return np.array([[j11, j12], [j21, j22]])

    @staticjit
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * np.tanh(tau * stiffness * np.sin(2 * np.pi * t / tau))

    def _postprocessing(self, r, th, tt):
        return r * np.cos(th), r * np.sin(th), np.sin(2 * np.pi * tt / self.tau)

    # def jac(self, X, t):
    #     r, th = X[0], X[1]
    #     phase = self._protocol(t, self.tau)
    #     return self._jac_static(r, th, t, self.a * phase, self.g * (1 - phase), self.n)

    def rhs(self, X, t):
        r, th, tt = X
        phase = self._protocol(tt, self.tau)
        dtt = 1
        dr, dth = self._rhs_static(r, th, t, self.a * phase, self.g * (1 - phase))
        return dr, dth, dtt


class OscillatingFlow(DynSys):
    @staticjit
    def _rhs(x, y, z, t, b, k, omega, u):
        f = x + b * np.sin(z)
        dx = u * np.cos(k * y) * np.sin(k * f)
        dy = -u * np.sin(k * y) * np.cos(k * f)
        dz = omega
        return dx, dy, dz

    def _postprocessing(self, x, y, z):
        return np.cos(self.k * x), y, np.sin(z)

    @staticjit
    def _jac(x, y, z, t, b, k, omega, u):
        row1 = [
            k * u * np.cos(k * y) * np.cos(k * (b * np.sin(z) + x)),
            -k * u * np.sin(k * y) * np.sin(k * (b * np.sin(z) + x)),
            b * k * u * np.cos(z) * np.cos(k * y) * np.cos(k * (b * np.sin(z) + x)),
        ]
        row2 = [
            k * u * np.sin(k * y) * np.sin(k * (b * np.sin(z) + x)),
            -k * u * np.cos(k * y) * np.cos(k * (b * np.sin(z) + x)),
            b * k * u * np.sin(k * y) * np.sin(k * (b * np.sin(z) + x)) * np.cos(z),
        ]
        row3 = [0, 0, 0]
        return row1, row2, row3


class BickleyJet(DynSys):
    @staticjit
    def _rhs(y, x, z, t, ell, eps, k, omega, sigma, u):
        sechy = 1 / np.cosh(y / ell)
        un = k * (x - z * sigma)
        dx = u * sechy**2 * (-1 - 2 * np.dot(np.cos(un), eps) * np.tanh(y / ell))
        dy = ell * u * sechy**2 * np.dot(eps * k, np.sin(un))
        dz = omega
        return dy, dx, dz

    def _postprocessing(self, x, y, z):
        km = np.min(self.k)
        sm = np.min(self.sigma)
        return x, np.sin(km * y), np.sin(self.omega * z * km * sm)


class ArnoldBeltramiChildress(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        dx = a * np.sin(z) + c * np.cos(y)
        dy = b * np.sin(x) + a * np.cos(z)
        dz = c * np.sin(y) + b * np.cos(x)
        return dx, dy, dz

    @staticjit
    def _postprocessing(x, y, z):
        return np.sin(x), np.cos(y), np.sin(z)

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [0, -c * np.sin(y), a * np.cos(z)]
        row2 = [b * np.cos(x), 0, -a * np.sin(z)]
        row3 = [-b * np.sin(x), c * np.cos(y), 0]
        return row1, row2, row3


class JerkCircuit(DynSys):
    @staticjit
    def _rhs(x, y, z, t, eps, y0):
        xdot = y
        ydot = z
        zdot = -z - x - eps * (np.exp(y / y0) - 1)
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, eps, y0):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, -eps * np.exp(y / y0) / y0, -1]
        return row1, row2, row3


class ForcedBrusselator(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, f, w):
        xdot = a + x**2 * y - (b + 1) * x + f * np.cos(z)
        ydot = b * x - x**2 * y
        zdot = w
        return xdot, ydot, zdot

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)

    @staticjit
    def _jac(x, y, z, t, a, b, f, w):
        row1 = [-b + 2 * x * y - 1, x**2, -f * np.sin(z)]
        row2 = [b - 2 * x * y, -(x**2), 0]
        row3 = [0, 0, 0]
        return row1, row2, row3


class WindmiReduced(DynSys):
    @staticjit
    def _rhs(i, v, p, t, a1, b1, b2, b3, d1, vsw):
        idot = a1 * (vsw - v)
        vdot = b1 * i - b2 * p**1 / 2 - b3 * v
        pdot = vsw**2 - p ** (5 / 4) * vsw ** (1 / 2) * (1 + np.tanh(d1 * (i - 1))) / 2
        return idot, vdot, pdot

    @staticjit
    def _jac(i, v, p, t, a1, b1, b2, b3, d1, vsw):
        row1 = [0, -a1, 0]
        row2 = [b1, -b3, -1 / 2 * b2]
        row3 = [
            -1 / 2 * d1 * p**1.25 * vsw**0.5 / np.cosh(d1 * (i - 1)) ** 2,
            0,
            -0.625 * p**0.25 * vsw**0.5 * (np.tanh(d1 * (i - 1)) + 1),
        ]
        return row1, row2, row3


class MooreSpiegel(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, eps):
        xdot = y
        ydot = a * z
        zdot = -z + eps * y - y * x**2 - b * x
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, eps):
        row1 = [0, 1, 0]
        row2 = [0, 0, a]
        row3 = [-2 * x * y - b, eps - x**2, -1]
        return row1, row2, row3


class CoevolvingPredatorPrey(DynSys):
    @staticjit
    def _rhs(x, y, alpha, t, a1, a2, a3, b1, b2, d1, d2, delta, k1, k2, k4, vv):
        xdot = x * (
            -((a3 * y) / (1 + b2 * x))
            + (a1 * alpha * (1 - k1 * x * (-alpha + alpha * delta))) / (1 + b1 * alpha)
            - d1
            * (
                1
                - k2 * (-(alpha**2) + (alpha * delta) ** 2)
                + k4 * (-(alpha**4) + (alpha * delta) ** 4)
            )
        )
        ydot = (-d2 + (a2 * x) / (1 + b2 * x)) * y
        alphadot = vv * (
            -((a1 * k1 * x * alpha * delta) / (1 + b1 * alpha))
            - d1 * (-2 * k2 * alpha * delta**2 + 4 * k4 * alpha**3 * delta**4)
        )
        return xdot, ydot, alphadot

    @staticjit
    def _jac(x, y, alpha, t, a1, a2, a3, b1, b2, d1, d2, delta, k1, k2, k4, vv):
        c0 = b2 * x + 1
        c1 = c0 ** (-2.0)
        c2 = alpha * delta
        c3 = -alpha + c2
        c4 = alpha * b1 + 1
        c5 = a1 / c4
        c6 = alpha * c5
        c7 = alpha**4
        c8 = delta**4
        c9 = alpha**2
        c10 = delta**2
        c11 = c0 ** (-1.0)
        c12 = a3 * c11
        c13 = k1 * x
        c14 = c4**2
        c15 = c14 ** (-1.0)
        c16 = alpha * c13 * (delta - 1)
        c17 = c9 * k4
        row1 = [
            -c12 * y
            + c6 * (-c13 * c3 + 1)
            - d1 * (-k2 * (c10 * c9 - c9) + k4 * (c7 * c8 - c7) + 1)
            + x * (a3 * b2 * c1 * y - c3 * c6 * k1),
            -c12 * x,
            c15
            * x
            * (
                a1 * alpha * b1 * (c16 - 1)
                - a1 * c4 * (2 * c16 - 1)
                - 2 * alpha * c14 * d1 * (2 * c17 * (c8 - 1) - k2 * (c10 - 1))
            ),
        ]
        row2 = [a2 * c1 * y, a2 * c11 * x - d2, 0]
        row3 = [
            -c2 * c5 * k1 * vv,
            0,
            -a1 * c13 * c15 * delta * vv
            + 2 * c10 * d1 * k2 * vv
            - 12 * c17 * c8 * d1 * vv,
        ]
        return row1, row2, row3


class KawczynskiStrizhak(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, gamma, kappa, mu):
        xdot = gamma * y - gamma * x**3 + 3 * mu * gamma * x
        ydot = -2 * mu * x - y - z + beta
        zdot = kappa * x - kappa * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, beta, gamma, kappa, mu):
        row1 = [-3 * gamma * x**2 + 3 * mu * gamma, gamma, 0]
        row2 = [-2 * mu, -1, -1]
        row3 = [kappa, 0, -kappa]
        return row1, row2, row3


class BelousovZhabotinsky(DynSys):
    @staticjit
    def _rhs(
        x,
        z,
        v,
        t,
        c1,
        c10,
        c11,
        c12,
        c13,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        ci,
        kf,
        t0,
        y0,
        yb1,
        yb2,
        yb3,
        z0,
    ):
        ybar = (1 / y0) * yb1 * z * v / (yb2 * x + yb3 + kf)
        x = max(0, x)
        rf = (ci - z0 * z) * np.sqrt(x)
        xdot = c1 * x * ybar + c2 * ybar + c3 * x**2 + c4 * rf + c5 * x * z - kf * x
        zdot = (c6 / z0) * rf + c7 * x * z + c8 * z * v + c9 * z - kf * z
        vdot = c10 * x * ybar + c11 * ybar + c12 * x**2 + c13 * z * v - kf * v
        return xdot * t0, zdot * t0, vdot * t0


class IsothermalChemical(DynSys):
    @staticmethod
    def _rhs(alpha, beta, gamma, t, delta, kappa, mu, sigma):
        alphadot = mu * (kappa + gamma) - alpha * beta**2 - alpha
        betadot = (alpha * beta**2 + alpha - beta) / sigma
        gammadot = (beta - gamma) / delta
        return alphadot, betadot, gammadot

    @staticmethod
    def _jac(alpha, beta, gamma, t, delta, kappa, mu, sigma):
        row1 = [-(beta**2) - 1, -2 * alpha * beta, mu]
        row2 = [(beta**2 + 1) / sigma, (2 * alpha * beta - 1) / sigma, 0]
        row3 = [0, delta ** (-1.0), -1 / delta]
        return row1, row2, row3


class VallisElNino(DynSys):
    @staticmethod
    def _rhs(x, y, z, t, b, c, p):
        xdot = b * y - c * x - c * p
        ydot = -y + x * z
        zdot = -z - x * y + 1
        return xdot, ydot, zdot

    @staticmethod
    def _jac(x, y, z, t, b, c, p):
        row1 = [-c, b, 0]
        row2 = [z, -1, x]
        row3 = [-y, -x, -1]
        return row1, row2, row3


class RabinovichFabrikant(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, g):
        xdot = y * z - y + y * x**2 + g * x
        ydot = 3 * x * z + x - x**3 + g * y
        zdot = -2 * a * z - 2 * x * y * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, g):
        row1 = [g + 2 * x * y, x**2 + z - 1, y]
        row2 = [-3 * x**2 + 3 * z + 1, g, 3 * x]
        row3 = [-2 * y * z, -2 * x * z, -2 * a - 2 * x * y]
        return row1, row2, row3


class NoseHoover(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = -x + y * z
        zdot = a - y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 1, 0]
        row2 = [-1, z, y]
        row3 = [0, -2 * y, 0]
        return row1, row2, row3


class Dadras(DynSys):
    @staticjit
    def _rhs(x, y, z, t, c, e, o, p, r):
        xdot = y - p * x + o * y * z
        ydot = r * y - x * z + z
        zdot = c * x * y - e * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, c, e, o, p, r):
        row1 = [-p, o * z + 1, o * y]
        row2 = [-z, r, 1 - x]
        row3 = [c * y, c * x, -e]
        return row1, row2, row3


class RikitakeDynamo(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, mu):
        xdot = -mu * x + y * z
        ydot = -mu * y - a * x + x * z
        zdot = 1 - x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, mu):
        row1 = [-mu, z, y]
        row2 = [-a + z, -mu, x]
        row3 = [-y, -x, 0]
        return row1, row2, row3


class NuclearQuadrupole(DynSys):
    @staticjit
    def _rhs(q1, q2, p1, p2, t, a, b, d):
        q1dot = a * p1
        q2dot = a * p2
        p1dot = (
            -a * q1
            + 3 / np.sqrt(2) * b * q1**2
            - 3 / np.sqrt(2) * b * q2**2
            - d * q1**3
            - d * q1 * q2**2
        )
        p2dot = -a * q2 - 3 * np.sqrt(2) * b * q1 * q2 - d * q2 * q1**2 - d * q2**3
        return q1dot, q2dot, p1dot, p2dot

    @staticjit
    def _jac(q1, q2, p1, p2, t, a, b, d):
        row1 = [0, 0, a, 0]
        row2 = [0, 0, 0, a]
        row3 = [
            -a + 3 * np.sqrt(2) * b * q1 - 3 * d * q1**2 - d * q2**2,
            q2 * (-3 * np.sqrt(2) * b - 2 * d * q1),
            0,
            0,
        ]
        row4 = [
            q2 * (-3 * np.sqrt(2) * b - 2 * d * q1),
            -a - 3 * np.sqrt(2) * b * q1 - d * q1**2 - 3 * d * q2**2,
            0,
            0,
        ]
        return row1, row2, row3, row4


class PehlivanWei(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y - y * z
        ydot = y + y * z - 2 * x
        zdot = 2 - x * y - y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, 1 - z, -y]
        row2 = [-2, 1 + z, y]
        row3 = [-y, -x - 2 * y, 0]
        return row1, row2, row3


class SprottTorus(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y + 2 * x * y + x * z
        ydot = 1 - 2 * x**2 + y * z
        zdot = x - x**2 - y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [2 * y + z, 2 * x + 1, x]
        row2 = [-4 * x, z, y]
        row3 = [1 - 2 * x, -2 * y, 0]
        return row1, row2, row3


class SprottJerk(DynSys):
    @staticjit
    def _rhs(x, y, z, t, mu):
        xdot = y
        ydot = z
        zdot = -x + y**2 - mu * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, mu):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, 2 * y, -mu]
        return row1, row2, row3


## Not chaotic
# class JerkCircuit(DynSys):
#     def rhs(self, X, t):
#         x, y, z = X
#         xdot = y
#         ydot = z
#         zdot = -z - x  - self.eps*(np.exp(y/self.y0) - 1)
#         return (xdot, ydot, zdot)


class SprottA(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y
        ydot = -x + y * z
        zdot = 1 - y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, 1, 0]
        row2 = [-1, z, y]
        row3 = [0, -2 * y, 0]
        return row1, row2, row3


class SprottB(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y * z
        ydot = x - y
        zdot = 1 - x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, z, y]
        row2 = [1, -1, 0]
        row3 = [-y, -x, 0]
        return row1, row2, row3


class SprottC(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y * z
        ydot = x - y
        zdot = 1 - x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, z, y]
        row2 = [1, -1, 0]
        row3 = [-2 * x, 0, 0]
        return row1, row2, row3


class SprottD(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -y
        ydot = x + z
        zdot = x * z + 3 * y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, -1, 0]
        row2 = [1, 0, 1]
        row3 = [z, 6 * y, x]
        return row1, row2, row3


class SprottE(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y * z
        ydot = x**2 - y
        zdot = 1 - 4 * x
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, z, y]
        row2 = [2 * x, -1, 0]
        row3 = [-4, 0, 0]
        return row1, row2, row3


class SprottF(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y + z
        ydot = -x + a * y
        zdot = x**2 - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 1, 1]
        row2 = [-1, a, 0]
        row3 = [2 * x, 0, -1]
        return row1, row2, row3


class SprottG(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = a * x + z
        ydot = x * z - y
        zdot = -x + y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [a, 0, 1]
        row2 = [z, -1, x]
        row3 = [-1, 1, 0]
        return row1, row2, row3


class SprottH(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -y + z**2
        ydot = x + a * y
        zdot = x - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, -1, 2 * z]
        row2 = [1, a, 0]
        row3 = [1, 0, -1]
        return row1, row2, row3


class SprottI(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -a * y
        ydot = x + z
        zdot = x + y**2 - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, -a, 0]
        row2 = [1, 0, 1]
        row3 = [1, 2 * y, -1]
        return row1, row2, row3


class SprottJ(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = 2 * z
        ydot = -2 * y + z
        zdot = -x + y + y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, 0, 2]
        row2 = [0, -2, 1]
        row3 = [-1, 1 + 2 * y, 0]
        return row1, row2, row3


class SprottK(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = x * y - z
        ydot = x - y
        zdot = x + a * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [y, x, -1]
        row2 = [1, -1, 0]
        row3 = [1, 0, a]
        return row1, row2, row3


class SprottL(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = y + b * z
        ydot = a * x**2 - y
        zdot = 1 - x
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, 1, b]
        row2 = [2 * a * x, -1, 0]
        row3 = [-1, 0, 0]
        return row1, row2, row3


class SprottM(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -z
        ydot = -(x**2) - y
        zdot = a + a * x + y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 0, -1]
        row2 = [-2 * x, -1, 0]
        row3 = [a, 1, 0]
        return row1, row2, row3


class SprottN(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -2 * y
        ydot = x + z**2
        zdot = 1 + y - 2 * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, -2, 0]
        row2 = [1, 0, 2 * z]
        row3 = [0, 1, -2]
        return row1, row2, row3


class SprottO(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = x - z
        zdot = x + x * z + a * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 1, 0]
        row2 = [1, 0, -1]
        row3 = [1 + z, a, x]
        return row1, row2, row3


class SprottP(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = a * y + z
        ydot = -x + y**2
        zdot = x + y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, a, 1]
        row2 = [-1, 2 * y, 0]
        row3 = [1, 1, 0]
        return row1, row2, row3


class SprottQ(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -z
        ydot = x - y
        zdot = a * x + y**2 + b * z
        return (xdot, ydot, zdot)

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, 0, -1]
        row2 = [1, -1, 0]
        row3 = [a, 2 * y, b]
        return row1, row2, row3


class SprottR(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = a - y
        ydot = b + z
        zdot = x * y - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, -1, 0]
        row2 = [0, 0, 1]
        row3 = [y, x, -1]
        return row1, row2, row3


class SprottS(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -x - 4 * y
        ydot = x + z**2
        zdot = 1 + x
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [-1, -4, 0]
        row2 = [1, 0, 2 * z]
        row3 = [1, 0, 0]
        return row1, row2, row3


class SprottMore(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y
        ydot = -x - np.sign(z) * y
        zdot = y**2 - np.exp(-(x**2))
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, 1, 0]
        row2 = [-1, -np.sign(z), 0]
        row3 = [2 * x * np.exp(-(x**2)), 2 * y, 0]
        return row1, row2, row3


class Arneodo(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d):
        xdot = y
        ydot = z
        zdot = -a * x - b * y - c * z + d * x**3
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-a + 3 * d * x**2, -b, -c]
        return row1, row2, row3


class Coullet(Arneodo):
    pass


class Rucklidge(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a * x + b * y - y * z
        ydot = x
        zdot = -z + y**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [-a, b - z, -y]
        row2 = [1, 0, 0]
        row3 = [0, 2 * y, -1]
        return row1, row2, row3


class Sakarya(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, h, p, q, r, s):
        xdot = a * x + h * y + s * y * z
        ydot = -b * y - p * x + q * x * z
        zdot = c * z - r * x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, h, p, q, r, s):
        row1 = [a, h + s * z, s * y]
        row2 = [-p + q * z, -b, q * x]
        row3 = [-r * y, -r * x, c]
        return row1, row2, row3


class LiuChen(Sakarya):
    pass


class RayleighBenard(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, r):
        xdot = a * y - a * x
        ydot = r * y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, r):
        row1 = [-a, a, 0]
        row2 = [-z, r, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class Finance(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = (1 / b - a) * x + z + x * y
        ydot = -b * y - x**2
        zdot = -x - c * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [(1 / b - a) + y, x, 1]
        row2 = [-2 * x, -b, 0]
        row3 = [-1, 0, -c]
        return row1, row2, row3


class Bouali2(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, bb, c, g, m, y0):
        xdot = a * y0 * x - a * x * y - b * z
        ydot = -g * y + g * y * x**2
        zdot = -1.5 * m * x + m * bb * x * z - c * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, bb, c, g, m, y0):
        row1 = [a * (-y + y0), -a * x, -b]
        row2 = [2 * g * x * y, g * (x**2 - 1), 0]
        row3 = [m * (bb * z - 1.5), 0, bb * m * x - c]
        return row1, row2, row3


class Bouali(Bouali2):
    pass


class LuChenCheng(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = -(a * b) / (a + b) * x - y * z + c
        ydot = a * y + x * z
        zdot = b * z + x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-(a * b) / (a + b), -z, -y]
        row2 = [z, a, x]
        row3 = [y, x, b]
        return row1, row2, row3


class LuChen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * y - a * x
        ydot = -x * z + c * y
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-a, a, 0]
        row2 = [-z, c, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class QiChen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * y - a * x + y * z
        ydot = c * x + y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-a, a + z, y]
        row2 = [c - z, 1, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class ZhouChen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, e):
        xdot = a * x + b * y + y * z
        ydot = c * y - x * z + d * y * z
        zdot = e * z - x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d, e):
        row1 = [a, b + z, y]
        row2 = [-z, c + d * z, -x + d * y]
        row3 = [-y, -x, e]
        return row1, row2, row3


class BurkeShaw(DynSys):
    @staticjit
    def _rhs(x, y, z, t, e, n):
        xdot = -n * x - n * y
        ydot = y - n * x * z
        zdot = n * x * y + e
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, e, n):
        row1 = [-n, -n, 0]
        row2 = [-n * z, 1, -n * x]
        row3 = [n * y, n * x, 0]
        return row1, row2, row3


class Chen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * y - a * x
        ydot = (c - a) * x - x * z + c * y
        zdot = x * y - b * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-a, a, 0]
        row2 = [c - a - z, c, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class ChenLee(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * x - y * z
        ydot = b * y + x * z
        zdot = c * z + (1 / 3) * x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [a, -z, -y]
        row2 = [z, b, x]
        row3 = [(1 / 3) * y, (1 / 3) * x, c]
        return row1, row2, row3


class WangSun(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, d, e, f, q):
        xdot = a * x + q * y * z
        ydot = b * x + d * y - x * z
        zdot = e * z + f * x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, d, e, f, q):
        row1 = [a, q * z, q * y]
        row2 = [b - z, d, -x]
        row3 = [f * y, f * x, e]
        return row1, row2, row3


class YuWang(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d):
        xdot = a * (y - x)
        ydot = b * x - c * x * z
        zdot = np.exp(x * y) - d * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d):
        row1 = [-a, a, 0]
        row2 = [b - c * z, 0, -c * x]
        row3 = [y * np.exp(x * y), x * np.exp(x * y), -d]
        return row1, row2, row3


class YuWang2(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d):
        xdot = a * (y - x)
        ydot = b * x - c * x * z
        zdot = np.cosh(x * y) - d * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d):
        row1 = [-a, a, 0]
        row2 = [b - c * z, 0, -c * x]
        row3 = [y * np.sinh(x * y), x * np.sinh(x * y), -d]
        return row1, row2, row3


class SanUmSrisuchinwong(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y - x
        ydot = -z * np.tanh(x)
        zdot = -a + x * y + np.abs(y)
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [-1, 1, 0]
        row2 = [-z * (1 - np.tanh(x) ** 2), 0, -np.tanh(x)]
        row3 = [y, x + np.sign(y), 0]
        return row1, row2, row3


class DequanLi(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, c, d, eps, f, k):
        xdot = a * y - a * x + d * x * z
        ydot = k * x + f * y - x * z
        zdot = c * z + x * y - eps * x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, c, d, eps, f, k):
        row1 = [-a + d * z, a, d * x]
        row2 = [k - z, f, -x]
        row3 = [y - 2 * eps * x, x, c]
        return row1, row2, row3


class PanXuZhou(DequanLi):
    pass


class Tsucs2(DequanLi):
    pass


class ArnoldWeb(DynSys):
    @staticjit
    def _rhs(p1, p2, x1, x2, z, t, mu, w):
        denom = 4 + np.cos(z) + np.cos(x1) + np.cos(x2)
        p1dot = -mu * np.sin(x1) / denom**2
        p2dot = -mu * np.sin(x2) / denom**2
        x1dot = p1
        x2dot = p2
        zdot = w
        return p1dot, p2dot, x1dot, x2dot, zdot

    @staticjit
    def _postprocessing(p1, p2, x1, x2, z):
        return p1, p2, np.sin(x1), np.sin(x2), np.cos(z)

    @staticjit
    def _jac(p1, p2, x1, x2, z, t, mu, w):
        c0 = np.sin(x1)
        c1 = np.cos(x1)
        c2 = np.cos(x2)
        c3 = c1 + c2 + np.cos(z) + 4
        c4 = mu / c3**3
        c5 = np.sin(x2)
        c6 = 2 * c0 * c4
        c7 = -c5 * c6
        c8 = np.sin(z)
        row1 = [0, 0, c4 * (-2 * c0**2 - c1 * c3), c7, -c6 * c8]
        row2 = [0, 0, c7, c4 * (-c2 * c3 - 2 * c5**2), -2 * c4 * c5 * c8]
        row3 = [1, 0, 0, 0, 0]
        row4 = [0, 1, 0, 0, 0]
        row5 = [0, 0, 0, 0, 0]
        return row1, row2, row3, row4, row5


class NewtonLiepnik(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a * x + y + 10 * y * z
        ydot = -x - 0.4 * y + 5 * x * z
        zdot = b * z - 5 * x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [-a, 10 * z + 1, 10 * y]
        row2 = [5 * z - 1, -0.4, 5 * x]
        row3 = [-5 * y, -5 * x, b]
        return row1, row2, row3


class HyperRossler(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = -y - z
        ydot = x + a * y + w
        zdot = b + x * z
        wdot = -c * z + d * w
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a, b, c, d):
        row1 = [0, -1, -1, 0]
        row2 = [1, a, 0, 1]
        row3 = [z, 0, x, 0]
        row4 = [0, 0, -c, d]
        return row1, row2, row3, row4


class HyperLorenz(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = a * y - a * x + w
        ydot = -x * z + c * x - y
        zdot = -b * z + x * y
        wdot = d * w - x * z
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a, b, c, d):
        row1 = [-a, a, 0, 1]
        row2 = [c - z, -1, -x, 0]
        row3 = [y, x, -b, 0]
        row4 = [-z, 0, -x, d]
        return row1, row2, row3, row4


class HyperCai(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d, e):
        xdot = a * y - a * x
        ydot = b * x + c * y - x * z + w
        zdot = -d * z + y**2
        wdot = -e * x
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a, b, c, d, e):
        row1 = [-a, a, 0, 0]
        row2 = [b - z, c, -x, 1]
        row3 = [0, 2 * y, -d, 0]
        row4 = [-e, 0, 0, 0]
        return row1, row2, row3, row4


class HyperBao(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d, e):
        xdot = a * y - a * x + w
        ydot = c * y - x * z
        zdot = x * y - b * z
        wdot = e * x + d * y * z
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a, b, c, d, e):
        row1 = [-a, a, 0, 1]
        row2 = [-z, c, -x, 0]
        row3 = [y, x, -b, 0]
        row4 = [e, d * z, d * y, 0]
        return row1, row2, row3, row4


class HyperJha(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = a * y - a * x + w
        ydot = -x * z + b * x - y
        zdot = x * y - c * z
        wdot = -x * z + d * w
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a, b, c, d):
        row1 = [-a, a, 0, 1]
        row2 = [b - z, -1, -x, 0]
        row3 = [y, x, -c, 0]
        row4 = [-z, 0, -x, d]
        return row1, row2, row3, row4


class HyperQi(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d, e, f):
        xdot = a * y - a * x + y * z
        ydot = b * x + b * y - x * z
        zdot = -c * z - e * w + x * y
        wdot = -d * w + f * z + x * y
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a, b, c, d, e, f):
        row1 = [-a, a + z, y, 0]
        row2 = [b - z, b, -x, 0]
        row3 = [y, x, -c, -e]
        row4 = [y, x, f, -d]
        return row1, row2, row3, row4


class Qi(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = a * y - a * x + y * z * w
        ydot = b * x + b * y - x * z * w
        zdot = -c * z + x * y * w
        wdot = -d * w + x * y * z
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a, b, c, d):
        row1 = [-a, a + w * z, w * y, y * z]
        row2 = [b - w * z, b, -w * x, -x * z]
        row3 = [w * y, w * x, -c, x * y]
        row4 = [y * z, x * z, x * y, -d]
        return row1, row2, row3, row4


class LorenzStenflo(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = a * y - a * x + d * w
        ydot = c * x - x * z - y
        zdot = x * y - b * z
        wdot = -x - a * w
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a, b, c, d):
        row1 = [-a, a, 0, d]
        row2 = [c - z, -1, -x, 0]
        row3 = [y, x, -b, 0]
        row4 = [-1, 0, 0, -a]
        return row1, row2, row3, row4


class HyperYangChen(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=30, b=3, c=35, d=8):
        xdot = a * y - a * x
        ydot = c * x - x * z + w
        zdot = -b * z + x * y
        wdot = -d * x
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a=30, b=3, c=35, d=8):
        row1 = [-a, a, 0, 0]
        row2 = [c - z, 0, -x, 1]
        row3 = [y, x, -b, 0]
        row4 = [-d, 0, 0, 0]
        return row1, row2, row3, row4


class HyperYan(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=37, b=3, c=26, d=38):
        xdot = a * y - a * x
        ydot = (c - a) * x - x * z + c * y
        zdot = -b * z + x * y - y * z + x * z - w
        wdot = -d * w + y * z - x * z
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a=37, b=3, c=26, d=38):
        row1 = [-a, a, 0, 0]
        row2 = [-a + c - z, c, -x, 0]
        row3 = [y + z, x - z, -b + x - y, -1]
        row4 = [-z, z, -x + y, -d]
        return row1, row2, row3, row4


class HyperXu(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=10, b=40, c=2.5, d=2, e=16):
        xdot = a * y - a * x + w
        ydot = b * x + e * x * z
        zdot = -c * z - x * y
        wdot = x * z - d * y
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a=10, b=40, c=2.5, d=2, e=16):
        row1 = [-a, a, 0, 1]
        row2 = [b + e * z, 0, e * x, 0]
        row3 = [-y, -x, -c, 0]
        row4 = [z, -d, x, 0]
        return row1, row2, row3, row4


class HyperWang(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=10, b=40, c=2.5, d=10.6, e=4):
        xdot = a * y - a * x
        ydot = -x * z + b * x + w
        zdot = -c * z + e * x**2
        wdot = -d * x
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a=10, b=40, c=2.5, d=10.6, e=4):
        row1 = [-a, a, 0, 0]
        row2 = [b - z, 0, -x, 1]
        row3 = [2 * e * x, 0, -c, 0]
        row4 = [-d, 0, 0, 0]
        return row1, row2, row3, row4


class HyperPang(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=36, b=3, c=20, d=2):
        xdot = a * y - a * x
        ydot = -x * z + c * y + w
        zdot = x * y - b * z
        wdot = -d * x - d * y
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a=36, b=3, c=20, d=2):
        row1 = [-a, a, 0, 0]
        row2 = [-z, c, -x, 1]
        row3 = [y, x, -b, 0]
        row4 = [-d, -d, 0, 0]
        return row1, row2, row3, row4


class HyperLu(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=36, b=3, c=20, d=1.3):
        xdot = a * y - a * x + w
        ydot = -x * z + c * y
        zdot = x * y - b * z
        wdot = d * w + x * z
        return xdot, ydot, zdot, wdot

    @staticjit
    def _jac(x, y, z, w, t, a=36, b=3, c=20, d=1.3):
        row1 = [-a, a, 0, 1]
        row2 = [-z, c, -x, 0]
        row3 = [y, x, -b, 0]
        row4 = [z, 0, x, d]
        return row1, row2, row3, row4


class SaltonSea(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, d, k, lam, m, mu, r, th):
        xdot = r * x * (1 - (x + y) / k) - lam * x * y
        ydot = lam * x * y - m * y * z / (y + a) - mu * y
        zdot = th * y * z / (y + a) - d * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, d, k, lam, m, mu, r, th):
        row1 = [(-k * lam * y - r * x + r * (k - x - y)) / k, x * (-k * lam - r) / k, 0]
        row2 = [lam * y, -a * m * z / (a + y) ** 2 + lam * x - mu, -m * y / (a + y)]
        row3 = [0, a * th * z / (a + y) ** 2, (-d * (a + y) + th * y) / (a + y)]
        return row1, row2, row3


class ExcitableCell(DynSys):
    def rhs(self, X, t):
        v, n, c = X

        alpham = self.ams * (self.amvb + v) / (1 - np.exp(-self.ar * v - self.amvo))
        betam = self.bms * np.exp(-self.bmr * v - self.bmo)
        minf = alpham / (alpham + betam)

        alphah = self.ahs * np.exp(-self.ahr * v - self.aho)
        betah = 1 / (1 + np.exp(-self.bhr * v - self.bho))
        hinf = alphah / (alphah + betah)

        alphan = self.ans * (self.anvb + v) / (1 - np.exp(-self.ar * v - self.anvo))
        betan = self.bns * np.exp(-self.bnr * v - self.bno)
        ninf = alphan / (alphan + betan)
        tau = 1 / (self.tscl * (alphan + betan))

        ca = c / (1 + c)

        vdot = (
            self.gi * minf**3 * hinf * (self.vi - v)
            + self.gkv * n**4 * (self.vk - v)
            + self.gkc * ca * (self.vk - v)
            + self.gl * (self.vl - v)
        )
        ndot = (ninf - n) / tau
        cdot = self.rho * (minf**3 * hinf * (self.vc - v) - self.kc * c)
        return vdot, ndot, cdot


class CaTwoPlus(DynSys):
    def rhs(self, X, t):
        z, y, a = X
        Vin = self.V0 + self.V1 * self.beta
        V2 = self.Vm2 * (z**2) / (self.K2**2 + z**2)
        V3 = (
            (self.Vm3 * (z**self.m) / (self.Kz**self.m + z**self.m))
            * (y**2 / (self.Ky**2 + y**2))
            * (a**4 / (self.Ka**4 + a**4))
        )
        V5 = (
            self.Vm5
            * (a**self.p / (self.K5**self.p + a**self.p))
            * (z**self.n / (self.Kd**self.n + z**self.n))
        )
        zdot = Vin - V2 + V3 + self.kf * y - self.k * z
        ydot = V2 - V3 - self.kf * y
        adot = self.beta * self.V4 - V5 - self.eps * a
        return (zdot, ydot, adot)


class CellCycle(DynSys):
    def rhs(self, X, t):
        c1, m1, x1, c2, m2, x2 = X
        Vm1, Um1 = 2 * [self.Vm1]
        vi1, vi2 = 2 * [self.vi]
        H1, H2, H3, H4 = 4 * [self.K]
        K1, K2, K3, K4 = 4 * [self.K]
        V2, U2 = 2 * [self.V2]
        Vm3, Um3 = 2 * [self.Vm3]
        V4, U4 = 2 * [self.V4]
        Kc1, Kc2 = 2 * [self.Kc]
        vd1, vd2 = 2 * [self.vd]
        Kd1, Kd2 = 2 * [self.Kd1]
        kd1, kd2 = 2 * [self.kd1]
        Kim1, Kim2 = 2 * [self.Kim]
        V1 = Vm1 * c1 / (Kc1 + c1)
        U1 = Um1 * c2 / (Kc2 + c2)
        V3 = m1 * Vm3
        U3 = m2 * Um3
        c1dot = vi1 * Kim1 / (Kim1 + m2) - vd1 * x1 * c1 / (Kd1 + c1) - kd1 * c1
        c2dot = vi2 * Kim2 / (Kim2 + m1) - vd2 * x2 * c2 / (Kd2 + c2) - kd2 * c2
        m1dot = V1 * (1 - m1) / (K1 + (1 - m1)) - V2 * m1 / (K2 + m1)
        m2dot = U1 * (1 - m2) / (H1 + (1 - m2)) - U2 * m2 / (H2 + m2)
        x1dot = V3 * (1 - x1) / (K3 + (1 - x1)) - V4 * x1 / (K4 + x1)
        x2dot = U3 * (1 - x2) / (H3 + (1 - x2)) - U4 * x2 / (H4 + x2)
        return c1dot, m1dot, x1dot, c2dot, m2dot, x2dot


class CircadianRhythm(DynSys):
    def rhs(self, X, t):
        m, fc, fs, fn, th = X
        vs = self.vc * ((0.5 + 0.5 * np.cos(th)) + self.vmin) * (self.vmax - self.vmin)
        mdot = vs * (self.Ki**self.n) / (self.Ki**self.n + fn**self.n) - self.vm * m / (
            self.km + m
        )
        fcdot = self.ks * m - self.k1 * fc + self.k2 * fn - self.k * fc
        fsdot = self.k * fc - self.vd * fs / (self.kd + fs)
        fndot = self.k1 * fc - self.k2 * fn - self.vdn * fn / (self.kdn + fn)
        thdot = 2 * np.pi / self.ps
        return mdot, fcdot, fsdot, fndot, thdot

    @staticjit
    def _postprocessing(m, fc, fs, fn, th):
        return m, fc, fs, fn, np.cos(th)


class FluidTrampoline(DynSys):
    @staticmethod
    def _rhs(x, y, th, t, gamma, psi, w):
        xdot = y
        ydot = -1 - np.heaviside(-x, 0) * (x + psi * y * np.abs(y)) + gamma * np.cos(th)
        thdot = w
        return xdot, ydot, thdot

    @staticjit
    def _postprocessing(x, y, th):
        return x, y, np.cos(th)


class Langford(DynSys):
    """Langford's (1984) normal form for an interacting cusp (hysteresis) and Hopf
    bifurcation, with a term breaking its axial symmetry.
    """

    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, e, f):
        xdot = x * z - b * x - d * y
        ydot = d * x + y * z - b * y
        zdot = (
            c
            + a * z
            - (z**3) / 3
            - x**2
            - y**2
            - e * z * x**2
            - e * z * y**2
            + f * z * x**3
        )
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d, e, f):
        row1 = [-b + z, -d, x]
        row2 = [d, -b + z, y]
        row3 = [
            x * (-2 * e * z + 3 * f * x * z - 2),
            2 * y * (-e * z - 1),
            a - e * x**2 - e * y**2 + f * x**3 - z**2,
        ]
        return row1, row2, row3


class AnishchenkoAstakhov(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        mu, eta = self.mu, self.eta
        xdot = mu * x + y - x * z
        ydot = -x
        zdot = -eta * z + eta * np.heaviside(x, 0) * x**2
        return xdot, ydot, zdot


class ShimizuMorioka(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = y
        ydot = x - a * y - x * z
        zdot = -b * z + x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, 1, 0]
        row2 = [1 - z, -a, -x]
        row3 = [2 * x, 0, -b]
        return row1, row2, row3


class GenesioTesi(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = y
        ydot = z
        zdot = -c * x - b * y - a * z + x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-c + 2 * x, -b, -a]
        return row1, row2, row3


class AtmosphericRegime(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, mu1, mu2, omega, sigma):
        xdot = mu1 * x + sigma * x * y
        ydot = mu2 * y + omega * z + alpha * y * z + beta * z**2 - sigma * x**2
        zdot = mu2 * z - omega * y - alpha * y**2 - beta * y * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, beta, mu1, mu2, omega, sigma):
        row1 = [mu1 + sigma * y, sigma * x, 0]
        row2 = [-2 * sigma * x, alpha * z + mu2, alpha * y + 2 * beta * z + omega]
        row3 = [0, -2 * alpha * y - beta * z - omega, -beta * y + mu2]
        return row1, row2, row3


class Hadley(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, f, g):
        xdot = -(y**2) - z**2 - a * x + a * f
        ydot = x * y - b * x * z - y + g
        zdot = b * x * y + x * z - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, f, g):
        row1 = [-a, -2 * y, -2 * z]
        row2 = [y - b * z, x - 1, -b * x]
        row3 = [b * y + z, b * x, x - 1]
        return row1, row2, row3


class ForcedVanDerPol(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, mu, w):
        ydot = mu * (1 - x**2) * y - x + a * np.sin(z)
        xdot = y
        zdot = w
        return xdot, ydot, zdot

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)

    @staticjit
    def _jac(x, y, z, t, a, mu, w):
        row1 = [0, 1, 0]
        row2 = [-2 * mu * x * y - 1, mu * (1 - x**2), a * np.cos(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class ForcedFitzHughNagumo(DynSys):
    @staticjit
    def _rhs(v, w, z, t, a, b, curr, f, gamma, omega):
        vdot = v - v**3 / 3 - w + curr + f * np.sin(z)
        wdot = gamma * (v + a - b * w)
        zdot = omega
        return vdot, wdot, zdot

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)

    @staticjit
    def _jac(v, w, z, t, a, b, curr, f, gamma, omega):
        row1 = [1 - v**2, -1, f * np.cos(z)]
        row2 = [gamma, -b * gamma, 0]
        row3 = [0, 0, 0]
        return row1, row2, row3


class HindmarshRose(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, s, tx, tz):
        xdot = -x + 1 / tx * y - a / tx * x**3 + b / tx * x**2 + 1 / tx * z
        ydot = -a * x**3 - (d - b) * x**2 + z
        zdot = -s / tz * x - 1 / tz * z + c / tz
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d, s, tx, tz):
        row1 = [(-3 * a * x**2 + 2 * b * x - tx) / tx, tx ** (-1.0), tx ** (-1.0)]
        row2 = [x * (-3 * a * x + 2 * b - 2 * d), 0, 1]
        row3 = [-s / tz, 0, -1 / tz]
        return row1, row2, row3


class Colpitts(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        u = z - (self.e - 1)
        fz = -u * (1 - np.heaviside(u, 0))
        xdot = y - self.a * fz
        ydot = self.c - x - self.b * y - z
        zdot = y - self.d * z
        return (xdot, ydot, zdot)


class Laser(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, h, k):
        xdot = a * y - a * x + b * y * z**2
        ydot = c * x + d * x * z**2
        zdot = h * z + k * x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d, h, k):
        row1 = [-a, a + b * z**2, 2 * b * y * z]
        row2 = [c + d * z**2, 0, 2 * d * x * z]
        row3 = [2 * k * x, 0, h]
        return row1, row2, row3


class Blasius(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, alpha1, alpha2, b, c, k1, k2, zs):
        xdot = a * x - alpha1 * x * y / (1 + k1 * x)
        ydot = -b * y + alpha1 * x * y / (1 + k1 * x) - alpha2 * y * z / (1 + k2 * y)
        zdot = -c * (z - zs) + alpha2 * y * z / (1 + k2 * y)
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, alpha1, alpha2, b, c, k1, k2, zs):
        row1 = [a - alpha1 * y / (k1 * x + 1) ** 2, -alpha1 * x / (k1 * x + 1), 0]
        row2 = [
            alpha1 * y / (k1 * x + 1) ** 2,
            alpha1 * x / (k1 * x + 1)
            + alpha2 * k2 * y * z / (k2 * y + 1) ** 2
            - alpha2 * z / (k2 * y + 1)
            - b,
            -alpha2 * y / (k2 * y + 1),
        ]
        row3 = [0, alpha2 * z / (k2 * y + 1) ** 2, alpha2 * y / (k2 * y + 1) - c]
        return row1, row2, row3


class TurchinHanski(DynSys):
    @staticjit
    def _rhs(n, p, z, t, a, d, e, g, h, lamb, r, s):
        ndot = (
            r * (1 - e * np.sin(z)) * n
            - r * (n**2)
            - g * (n**2) / (n**2 + h**2)
            - a * n * p / (n + d)
        )
        pdot = s * (1 - e * np.sin(z)) * p - s * (p**2) / n
        zdot = 2 * np.pi * lamb
        return ndot, pdot, zdot

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)

    @staticjit
    def _jac(n, p, z, t, a, d, e, g, h, lamb, r, s):
        row1 = [
            -a * d * p / (d + n) ** 2
            - e * r * np.sin(z)
            - 2 * g * h**2 * n / (h**2 + n**2) ** 2
            - 2 * n * r
            + r,
            -a * n / (d + n),
            -e * n * r * np.cos(z),
        ]
        row2 = [
            p**2 * s / n**2,
            s * (-e * n * np.sin(z) + n - 2 * p) / n,
            -e * p * s * np.cos(z),
        ]
        row3 = [0, 0, 0]
        return row1, row2, row3


class StickSlipOscillator(DynSys):
    def _t(self, v):
        return self.t0 * np.sign(v) - self.alpha * v + self.beta * v**3

    @staticjit
    def _rhs(x, v, th, t, a, alpha, b, beta, eps, gamma, t0, vs, w):
        tq = t0 * np.sign(v - vs) - alpha * v + beta * (v - vs) ** 3
        xdot = v
        vdot = eps * (gamma * np.cos(th) - tq) + a * x - b * x**3
        thdot = w
        return xdot, vdot, thdot

    @staticjit
    def _postprocessing(x, v, th):
        return x, v, np.cos(th)

    @staticjit
    def _jac(x, v, th, t, a, alpha, b, beta, eps, gamma, t0, vs, w):
        row1 = [0, 1, 0]
        row2 = [
            a - 3 * b * x**2,
            eps * (alpha - 3 * beta * (v - vs) ** 2),
            -eps * gamma * np.sin(th),
        ]
        row3 = [0, 0, 0]
        return row1, row2, row3


class HastingsPowell(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a1, a2, b1, b2, d1, d2):
        xdot = x * (1 - x) - y * a1 * x / (1 + b1 * x)
        ydot = y * a1 * x / (1 + b1 * x) - z * a2 * y / (1 + b2 * y) - d1 * y
        zdot = z * a2 * y / (1 + b2 * y) - d2 * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a1, a2, b1, b2, d1, d2):
        row1 = [-a1 * y / (b1 * x + 1) ** 2 - 2 * x + 1, -a1 * x / (b1 * x + 1), 0]
        row2 = [
            a1 * y / (b1 * x + 1) ** 2,
            a1 * x / (b1 * x + 1)
            + a2 * b2 * y * z / (b2 * y + 1) ** 2
            - a2 * z / (b2 * y + 1)
            - d1,
            -a2 * y / (b2 * y + 1),
        ]
        row3 = [0, a2 * z / (b2 * y + 1) ** 2, a2 * y / (b2 * y + 1) - d2]
        return row1, row2, row3


class CellularNeuralNetwork(DynSys):
    @staticjit
    def f(x):
        return 0.5 * (np.abs(x + 1) - np.abs(x - 1))

    def rhs(self, X, t):
        x, y, z = X
        xdot = -x + self.d * self.f(x) - self.b * self.f(y) - self.b * self.f(z)
        ydot = -y - self.b * self.f(x) + self.c * self.f(y) - self.a * self.f(z)
        zdot = -z - self.b * self.f(x) + self.a * self.f(y) + self.f(z)
        return (xdot, ydot, zdot)


class BeerRNN(DynSys):
    @staticjit
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def rhs(self, X, t):
        Xdot = (-X + np.matmul(self.w, self._sig(X + self.theta))) / self.tau
        return Xdot


class Torus(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, n, r):
        xdot = (-a * n * np.sin(n * t)) * np.cos(t) - (r + a * np.cos(n * t)) * np.sin(
            t
        )
        ydot = (-a * n * np.sin(n * t)) * np.sin(t) + (r + a * np.cos(n * t)) * np.cos(
            t
        )
        zdot = a * n * np.cos(n * t)
        return xdot, ydot, zdot


class CaTwoPlusQuasiperiodic(CaTwoPlus):
    pass


class Hopfield(DynSys):
    def f(self, x):
        return (1 + np.tanh(x)) / 2

    def rhs(self, X, t):
        Xdot = -X / self.tau + self.f(self.eps * np.matmul(self.k, X)) - self.beta
        return Xdot


class MacArthur(DynSys):
    def growth_rate(self, rr):
        u0 = rr / (self.k.T + rr)
        u = self.r * u0.T
        return np.min(u.T, axis=1)

    def rhs(self, X, t):
        nn, rr = X[:5], X[5:]
        mu = self.growth_rate(rr)
        nndot = nn * (mu - self.m)
        rrdot = self.d * (self.s - rr) - np.matmul(self.c, (mu * nn))
        return np.hstack([nndot, rrdot])


class ItikBanksTumor(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a12, a13, a21, a31, d3, k3, r2, r3):
        xdot = x * (1 - x) - a12 * x * y - a13 * x * z
        ydot = r2 * y * (1 - y) - a21 * x * y
        zdot = r3 * x * z / (x + k3) - a31 * x * z - d3 * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a12, a13, a21, a31, d3, k3, r2, r3):
        row1 = [-a12 * y - a13 * z - 2 * x + 1, -a12 * x, -a13 * x]
        row2 = [-a21 * y, -a21 * x - 2 * r2 * y + r2, 0]
        row3 = [
            -a31 * z + k3 * r3 * z / (k3 + x) ** 2,
            0,
            -a31 * x - d3 + r3 * x / (k3 + x),
        ]
        return row1, row2, row3


## Doesn't match described dynamics
# class CosmologyFriedmann(DynSys):
#     @staticjit
#     def _rhs(x, y, z, t, a, b, c, d, p):
#         xdot = y
#         ydot = -a * y**2 / x - b * x - c * x**3 + d * p * x
#         zdot = 3 * (y / x) * (p + z)
#         return xdot, ydot, zdot


## Doesn't match described dynamics
# class MixMasterUniverse(DynSys):
#     def rhs(self, X, t):
#         a, b, g, adot_, bdot_, gdot_ = X
#         adot = adot_
#         bdot = bdot_
#         gdot = gdot_
#         addot = (np.exp(2*b) - np.exp(2*g))**2 - np.exp(4*a)
#         bddot = (np.exp(2*g) - np.exp(2*a))**2 - np.exp(4*b)
#         gddot = (np.exp(2*a) - np.exp(2*b))**2 - np.exp(4*g)
#         return (adot, bdot, gdot, addot, bddot, gddot)

## Doesn't match described dynamics
# class Universe(DynSys):
#     def rhs(self, X, t):
#         #Xdot = X * np.matmul(self.a, 1 - X)
#         Xdot = self.r * X * (1 - np.matmul(self.a, X))
#         return Xdot

# class SeasonalSEIR:
#     """
#     This is extremely unstable for some reason
#     Olsen, Schaffer. Science 1990

#     eq = SeasonalSEIR()
#     tpts = np.linspace(0, 1000, 100000)
#     ic = (.579, .02, .001, 1e-6)
#     sol = integrate_dyn(eq, ic, tpts)

#     plt.plot(sol[0], sol[2], '.k', markersize=.1)

#     """
#     def __init__(self):
#         pass

#     def __call__(self, X, t):
#         (s, e, i, th) = X

#         ## measles
#         m = 0.02
#         a = 35.84
#         g = 100
#         b0 = 1800
#         b1 = 0.28

# #         ## chickenpox
# #         m = 0.02
# #         a = 36
# #         g = 34.3
# #         b0 = 537
# #         b1 = 0.3

#         b = b0*(1 + b1*np.cos(th))
#         sdot = m*(1 - s) - b*s*i
#         edot = b*s*i - (m + a)*e
#         idot = a*e - (m + g)*i
#         thdot = 2*np.pi
#         return (sdot, edot, idot, thdot)

# class SeasonalSEIR:
#     """
#     Seasonally forced SEIR model
#     Zhang, Q., Liu, C., & Zhang, X. (2012). Analysis and Control of an SEIR Epidemic System with Nonlinear Transmission Rate. Lecture Notes in Control and Information Sciences, 203–225. doi:10.1007/978-1-4471-2303-3_14
#     """
#     def __init__(self, b=0.02, beta0=1800, alpha=35.84, gamma=100.0, beta1=0.28):
#         self.b, self.beta0, self.alpha, self.gamma, self.beta1 = b, beta0, alpha, gamma, beta1
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         alpha is a
#         b is mu
#         """
#         s, e, i, th = X
#         beta = self.beta0*(1 + self.beta1*np.cos(th)) # seasonal forcing
#         sdot = self.b - self.b*s - beta*s*i
#         edot = beta*s*i - (self.alpha + self.b)*e
#         idot = self.alpha*e - (self.gamma + self.b)*i
#         thdot = 2*np.pi
#         return (sdot, edot, idot, thdot)

# class SeasonalSEIR:
#     """
#     Seasonally forced SEIR model
#     Zhang, Q., Liu, C., & Zhang, X. (2012). Analysis and Control of an SEIR Epidemic System with Nonlinear Transmission Rate. Lecture Notes in Control and Information Sciences, 203–225. doi:10.1007/978-1-4471-2303-3_14
#     """
#     def __init__(self, b=0.02, beta0=1800, alpha=35.84, gamma=100.0, beta1=0.28):
#         self.b, self.beta0, self.alpha, self.gamma, self.beta1 = b, beta0, alpha, gamma, beta1
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         alpha is a
#         b is mu
#         """
#         s, e, i, th = X
#         beta = self.beta0*(1 + self.beta1*np.cos(th)) # seasonal forcing
#         sdot = self.b - self.b*s - beta*s*i
#         edot = beta*s*i - (self.alpha + self.b)*e
#         idot = self.alpha*e - (self.gamma + self.b)*i
#         thdot = 2*np.pi
#         return (sdot, edot, idot, thdot)

# class SeasonalSEIR:
#     """
#     Seasonally forced SEIR model
#     """
#     def __init__(self, mu=0.02, b0=1800, a=35.84, g=100.0, d=0.28):
#         self.mu, self.b0, self.a, self.g, self.d = mu, b0, a, g, d
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         alpha is a
#         b is mu
#         """
#         s, e, i, th = X
#         b = self.b0*(1 + self.d*np.cos(th)) # seasonal forcing
#         sdot = self.mu - self.mu*s - b*s*i
#         edot = b*s*i - (self.mu + self.a)*e
#         idot = self.a*e - (self.mu + self.g)*i
# #         edot = 0
# #         idot = 0
#         thdot = 2*np.pi

#         return (sdot, edot, idot, thdot)

# class SeasonalSEIR:
#     """
#     Seasonally forced SEIR model
# Bifurcation analysis of periodic SEIR and SIR epidemic models
#     """
#     def __init__(self, mu=0.02, b0=1884.95*5, a=35.842, g=100.0, d=0.255):
#         self.mu, self.b0, self.a, self.g, self.d = mu, b0, a, g, d
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         s, e, i, th = X
#         b = self.b0*(1 + self.d*np.cos(th)) # seasonal forcing
#         sdot = self.mu - self.mu*s - b*s*i
#         edot = b*s*i - (self.mu + self.a)*e
#         idot = self.a*e - (self.mu + self.g)*i
#         thdot = 2*np.pi
#         return (sdot, edot, idot, thdot)


# class SeasonalSIR:
#     """
#     Seasonally forced SEIR model
#     """
#     def __init__(self, mu=0.02, b0=1884.95, a=35.842, g=100, d=0.255):
#         self.mu, self.b0, self.a, self.g, self.d = mu, b0, a, g, d
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         s, i, th = X
#         b = self.b0*(1 + self.d*np.sin(th)) # seasonal forcing
#         sdot = self.mu - self.mu*s - b*s*i
#         idot =  b*s*i - (self.mu + self.g)*i
#         thdot = 2*np.pi
#         return (sdot, idot, thdot)


# class Robinson:
#     """
#     C Robinson 1989 Nonlinearity
#     Was unable to find published parameters for which this has a stable attractor,
#     it may only be transient
#     """
#     def __init__(self, a=0.71, b=1.8587, v=1.0, gamma=0.7061, delta=0.1):
#         self.a, self.b, self.v, self.gamma, self.delta = a, b, v, gamma, delta
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         x, y, z = X
#         xdot = y
#         ydot = x - 2*x**3 - self.a*y + (self.b*x**2)*y - self.v*y*z
#         zdot = -self.gamma*z + self.delta*x**2
#         return (xdot, ydot, zdot)

# Sato: Cardiac model analogous to HH
# https://www.sciencedirect.com/science/article/pii/S1007570416300016
# hundreds of equations


# class HodgkinHuxley:
#     def __init__(self, i=7.92197):
#         self.i = i

#     def phi(self, x):
#         return x/(np.exp(x) - 1)

#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         v, m, n, h = X
#         i = self.i
#         vdot = i - (120*m**3*h*(v + 115) + 36*n**4*(v - 12) + 0.3*(v + 10.599))
#         mdot = (1 - m)*self.phi((v + 25)/10) - m*(4*np.exp(v/18))
#         ndot = (1 - n)*0.1*self.phi((v + 10)/10) - n*0.125*np.exp(v/80)
#         hdot = (1 - h)*0.07*np.exp(v/20) - h/(1 + np.exp((v + 30)/10))
#         return (vdot, mdot, ndot, hdot)

##############################
##
## Quasiperiodic systems
##
##############################

# class SymmetricKuramoto(DynSys):
#     def coupling(self, x, n=4):
#         k = 1 + np.arange(n)
#         return np.sum(self.a[:, None, None] * np.cos(k[:, None, None]*x[None, ...] - self.eta[:, None, None]), axis=0)
#     def rhs(self, X, t):
#         phase_diff = X[:, None] - X[None, :]
#         Xdot = self.w + np.mean(self.coupling(phase_diff), axis=0) ## need to apply coupling element-wise
#         return Xdot

#     "SymmetricKuramoto": {
#         "initial_conditions": [
#             0.1,
#             0.01,
#             0.01,
#             0.01
#         ],
#         "dt": 0.01,
#         "parameters": {
#             "w" : 0.0,
#             "a": [-2, -2, -1,  -0.88],
#             "eta": [0.1104, -0.1104, 0.669, 0.669]
#         },
#         "citation": "Bick, Christian, et al. Chaos in symmetric phase oscillator networks. Physical review letters 107.24 (2011): 244101.",
#         "period": 7737.7
#     }

# class InterfacialFlight:
#     """
#     """
#     def __init__(self):
#         pass
#     def __call__(self, X, t):
#         x, y, z = X
#         rclaw = 57 #um
#         m = 2.2
#         cl = 1.55
#         ly = 73

#         l0 = 137*1e6 # convert from uN/mg into um/s^2
#         f = 116
#         r = 0.15
#         phi0 = np.pi/2

#         cdleg = 3
#         rhow = 1e-9 # water density in mg/um^3

#         sigma = 72800 # water surface tension mg/s^2

#         kinv = 2709 # um
#         hclaw = rclaw*np.log(2*kinv/rclaw)

#         sech = lambda pp : 1 / np.cosh(pp)

#         phi_arg = 2*np.pi*f*t + phi0
#         sin_term = np.sin(phi_arg)
#         hsin_term = np.heaviside(sin_term, 0)
#         zdot = (l0/m)*np.cos(phi_arg)*(hsin_term + r*(1 - hsin_term))
#         zdot += -(8*np.pi*sigma*rclaw/m)*sech((x - hclaw)/rclaw)*np.sign(x)
#         zdot += -(2*rhow*cdleg*np.pi*rclaw**2/m)*y*np.sign(y)

#         xdot = 1
#         ydot = x
#         return (xdot, ydot, zdot)

## Not chaotic
# class Robinson(DynSys):
#     @staticjit
#     def _rhs(x, y, z, t, a, b, c, d, v):
#         xdot = y
#         ydot = x - 2 * x**3 - a * y + b * x**2 * y - v * y * z
#         zdot = -c * z + d * x**2
#         return (xdot, ydot, zdot)


class Liu(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, h, k):
        xdot = a * (y - x)
        ydot = b * x - k * x * z
        zdot = -c * z + h * x**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, h, k):
        row1 = [-a, a, 0]
        row2 = [b - k * z, 0, -k * x]
        row3 = [2 * h * x, 0, -c]
        return row1, row2, row3


class DrivenPendulum(DynSys):
    @staticjit
    def _rhs(x, y, z, t, g, omega, q):
        xdot = y
        ydot = -y / q - np.sin(x) + g * np.cos(z)
        zdot = omega
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, g, omega, q):
        row1 = [0, 1, 0]
        row2 = [-np.cos(x), -1 / q, -g * np.sin(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return np.sin(x), y, np.cos(z)


class Rabinovich(DynSys):
    @staticjit
    def _rhs(x, y, z, t, h, nu1, nu2):
        xdot = h * y - nu1 * x - y * z
        ydot = h * x - nu2 * y + x * z
        zdot = -z + x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, h, nu1, nu2):
        row1 = [-nu1, h - z, -y]
        row2 = [h + z, -nu2, x]
        row3 = [y, x, -1]
        return row1, row2, row3


class WangChen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y * z + a
        ydot = x**2 - y
        zdot = 1 - 4 * x
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, z, y]
        row2 = [2 * x, -1, 0]
        row3 = [-4, 0, 0]
        return row1, row2, row3


class Wei(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -y
        ydot = x + z
        zdot = 2 * y**2 + x * z - a
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, -1, 0]
        row2 = [1, 0, 1]
        row3 = [z, 4 * y, x]
        return row1, row2, row3


class JafariSprott(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = y
        ydot = -x + y * z
        zdot = -x - a * x * y - b * x * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, 1, 0]
        row2 = [-1, z, y]
        row3 = [-1 - a * y - b * z, -a * x, -b * x]
        return row1, row2, row3


class HideSkeldonAcheson(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, kappa, lam):
        xdot = x * (y - 1) - beta * z
        ydot = alpha * (1 - x**2) - kappa * y
        zdot = x - lam * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, beta, kappa, lam):
        row1 = [y - 1, x, -beta]
        row2 = [-2 * alpha * x, -kappa, 0]
        row3 = [1, 0, -lam]
        return row1, row2, row3


class GlukhovskyDolzhansky(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, r, sigma):
        xdot = -sigma * (x - y) - a * y * z
        ydot = r * x - y - x * z
        zdot = -z + x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, r, sigma):
        row1 = [-sigma, sigma - a * z, -a * y]
        row2 = [r - z, -1, -x]
        row3 = [y, x, -1]
        return row1, row2, row3


class Tigan(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * (y - x)
        ydot = (c - a) * x - a * x * z
        zdot = -b * z + x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-a, a, 0]
        row2 = [c - a - a * z, 0, -a * x]
        row3 = [y, x, -b]
        return row1, row2, row3


class WillamowskiRossler(DynSys):
    @staticjit
    def _rhs(x, y, z, t, k1, k2, k3, k4, k5, km1, km2, km3, km4, km5):
        xdot = k1 * x - km1 * x**2 - k2 * x * y + km2 * y**2 - k4 * x * z + km4
        ydot = k2 * x * y - km2 * y**2 - k3 * y + km3
        zdot = -k4 * x * z + km4 + k5 * z - km5 * z**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, k1, k2, k3, k4, k5, km1, km2, km3, km4, km5):
        row1 = [k1 - 2 * km1 * x - k2 * y - k4 * z, -k2 * x + 2 * km2 * y, -k4 * x]
        row2 = [k2 * y, k2 * x - 2 * km2 * y - k3, 0]
        row3 = [-k4 * z, 0, -k4 * x + k5 - 2 * km5 * z]
        return row1, row2, row3


class ForcedRosenzweigMacArthur(DynSys):
    @staticjit
    def _rhs(x, y, th, t, a, b, d, e, eps, k0, omega, r):
        kk = k0 * (1 + eps * np.sin(th))
        xdot = r * x * (1 - x / kk) - a * x * y / (b + x)
        ydot = e * a * x * y / (b + x) - d * y
        thdot = omega
        return xdot, ydot, thdot

    @staticjit
    def _jac(x, y, th, t, a, b, d, e, eps, k0, omega, r):
        kk = k0 * (1 + eps * np.sin(th))
        dkk = k0 * eps * np.cos(th)
        row1 = [
            r * (1 - 2 * x / kk) - a * y * b / (b + x) ** 2,
            -a * x / (b + x),
            r * x**2 * dkk / kk**2,
        ]
        row2 = [e * a * y * b / (b + x) ** 2, e * a * x / (b + x) - d, 0]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, th):
        return x, y, np.sin(th)


class LinzSprott(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = z
        zdot = -a * z - y + np.abs(x) - 1
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [np.sign(x), -1, -a]
        return row1, row2, row3


class Gissinger(DynSys):
    @staticjit
    def _rhs(q, d, v, t, gamma, mu, nu):
        qdot = mu * q - v * d
        ddot = -nu * d + v * q
        vdot = gamma - v + q * d
        return qdot, ddot, vdot

    @staticjit
    def _jac(q, d, v, t, gamma, mu, nu):
        row1 = [mu, -v, -d]
        row2 = [v, -nu, q]
        row3 = [d, q, -1]
        return row1, row2, row3


class MuraliLakshmananChua(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, f, m0, m1, omega, sigma):
        hx = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
        xdot = y - hx
        ydot = -sigma * y - beta * x + f * np.sin(z)
        zdot = omega
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, beta, f, m0, m1, omega, sigma):
        dh = m1 + 0.5 * (m0 - m1) * (np.sign(x + 1) - np.sign(x - 1))
        row1 = [-dh, 1, 0]
        row2 = [-beta, -sigma, f * np.cos(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)


class ShawVanDerPol(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, omega):
        xdot = b * y + (c - d * y**2) * x
        ydot = -x + a * np.sin(z)
        zdot = omega
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c, d, omega):
        row1 = [c - d * y**2, b - 2 * d * x * y, 0]
        row2 = [-1, 0, a * np.cos(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)


class UedaAkamatsu(DynSys):
    @staticjit
    def _rhs(x, y, z, t, b, gamma, mu, nu):
        xdot = y
        ydot = mu * (1 - gamma * x**2) * y - x**3 + b * np.cos(z)
        zdot = nu
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, b, gamma, mu, nu):
        row1 = [0, 1, 0]
        row2 = [
            -2 * mu * gamma * x * y - 3 * x**2,
            mu * (1 - gamma * x**2),
            -b * np.sin(z),
        ]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.cos(z)


class ParametricPendulum(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, omega, p):
        xdot = y
        ydot = -beta * y - (1 + p * np.cos(z)) * np.sin(x)
        zdot = omega
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, beta, omega, p):
        row1 = [0, 1, 0]
        row2 = [-(1 + p * np.cos(z)) * np.cos(x), -beta, p * np.sin(z) * np.sin(x)]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return np.sin(x), y, np.cos(z)


class ForcedToda(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, alpha, mu, omega):
        xdot = y
        ydot = -mu * y - alpha * (np.exp(x) - 1) + a * np.sin(z)
        zdot = omega
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, alpha, mu, omega):
        row1 = [0, 1, 0]
        row2 = [-alpha * np.exp(x), -mu, a * np.cos(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)


class ModulatedLossLaser(DynSys):
    @staticjit
    def _rhs(u, w, z, t, eps, m, mu, omega):
        udot = w + (m / eps) * np.cos(z)
        wdot = mu - (1 + eps * w) * (1 + np.exp(u))
        zdot = omega
        return udot, wdot, zdot

    @staticjit
    def _jac(u, w, z, t, eps, m, mu, omega):
        row1 = [0, 1, -(m / eps) * np.sin(z)]
        row2 = [-(1 + eps * w) * np.exp(u), -eps * (1 + np.exp(u)), 0]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(u, w, z):
        return u, w, np.cos(z)


class McCannYodzis(DynSys):
    @staticjit
    def _rhs(r, c, p, t, c0, k, r0, xc, xp, yc, yp):
        fr = r / (r + r0)
        fc = c / (c + c0)
        rdot = r * (1 - r / k) - xc * yc * c * fr
        cdot = xc * c * (yc * fr - 1) - xp * yp * p * fc
        pdot = xp * p * (yp * fc - 1)
        return rdot, cdot, pdot

    @staticjit
    def _jac(r, c, p, t, c0, k, r0, xc, xp, yc, yp):
        fr = r / (r + r0)
        fc = c / (c + c0)
        dfr = r0 / (r + r0) ** 2
        dfc = c0 / (c + c0) ** 2
        row1 = [1 - 2 * r / k - xc * yc * c * dfr, -xc * yc * fr, 0]
        row2 = [
            xc * c * yc * dfr,
            xc * (yc * fr - 1) - xp * yp * p * dfc,
            -xp * yp * fc,
        ]
        row3 = [0, xp * p * yp * dfc, xp * (yp * fc - 1)]
        return row1, row2, row3


class UpadhyayRai(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, a1, a2, b1, c, i, i1, w, w1, w2, w3):
        fx = x**2 / i + x + a
        gy = y**2 / i1 + y + a
        xdot = a1 * x - b1 * x**2 - w * x * y / fx
        ydot = -a2 * y + w1 * x * y / fx - w2 * y * z / gy
        zdot = -c * z + w3 * y * z / gy
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, a1, a2, b1, c, i, i1, w, w1, w2, w3):
        fx = x**2 / i + x + a
        gy = y**2 / i1 + y + a
        dfx = (a - x**2 / i) / fx**2  # d(x/fx)/dx
        dgy = (a - y**2 / i1) / gy**2  # d(y/gy)/dy
        row1 = [a1 - 2 * b1 * x - w * y * dfx, -w * x / fx, 0]
        row2 = [w1 * y * dfx, -a2 + w1 * x / fx - w2 * z * dgy, -w2 * y / gy]
        row3 = [0, w3 * z * dgy, -c + w3 * y / gy]
        return row1, row2, row3


class RosslerPrototype4(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -y - z
        ydot = x
        zdot = a * (y - y**2) - b * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, -1, -1]
        row2 = [1, 0, 0]
        row3 = [0, a * (1 - 2 * y), -b]
        return row1, row2, row3


class UnifiedChaotic(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha):
        xdot = (25 * alpha + 10) * (y - x)
        ydot = (28 - 35 * alpha) * x - x * z + (29 * alpha - 1) * y
        zdot = x * y - (alpha + 8) / 3 * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha):
        row1 = [-(25 * alpha + 10), 25 * alpha + 10, 0]
        row2 = [28 - 35 * alpha - z, 29 * alpha - 1, -x]
        row3 = [y, x, -(alpha + 8) / 3]
        return row1, row2, row3


# ---------------------------------------------------------------------------
# Reduced-order models of physical, biological and engineered processes
# ---------------------------------------------------------------------------


class TimmermannJin(DynSys):
    @staticjit
    def _rhs(tw, te, hw, t, alpha, eps, h, hstar, mbl, mu, r, tr, tr0, z0, zeta):
        h2 = hw + mbl * (te - tw)
        tsub = tr - (tr - tr0) / 2 * (1 - np.tanh((h + h2 - z0) / hstar))
        twdot = -alpha * (tw - tr) - eps * mu * (te - tw) ** 2
        tedot = -alpha * (te - tr) + zeta * mu * (te - tw) * (te - tsub)
        hwdot = r * (-hw - mbl * (te - tw) / 2)
        return twdot, tedot, hwdot

    @staticjit
    def _jac(tw, te, hw, t, alpha, eps, h, hstar, mbl, mu, r, tr, tr0, z0, zeta):
        c0 = te - tw
        c1 = hstar ** (-1.0)
        c2 = c0 * mbl
        c3 = np.tanh(c1 * (c2 + h + hw - z0))
        c4 = c3 - 1
        c5 = tr - tr0
        c6 = c3**2 - 1
        c7 = c5 * c6
        c8 = mu * zeta
        c9 = (1 / 2) * c1
        c10 = (1 / 2) * tr - 1 / 2 * tr0
        c11 = c0 * c8
        c12 = (1 / 2) * mbl * r
        row1 = [-alpha + 2 * c0 * eps * mu, -2 * c0 * eps * mu, 0]
        row2 = [
            c8 * c9 * (-c2 * c7 + hstar * (c4 * c5 - 2 * te + 2 * tr)),
            -alpha + c11 * (c1 * c10 * c6 * mbl + 1) + c8 * (-c10 * c4 + te - tr),
            c11 * c7 * c9,
        ]
        row3 = [c12, -c12, -r]
        return row1, row2, row3


class SugamaHorton(DynSys):
    @staticjit
    def _rhs(u, logk, logf, t, c, q):
        su = np.sqrt(u)
        m = u**-1.5 * (0.95 + 0.05 * u**2.5)
        udot = q - su * np.exp(logk)
        logkdot = su - c * np.exp(logf) / su - np.exp(logk) / u
        logfdot = c * np.exp(logk) / su - c * m
        return udot, logkdot, logfdot

    @staticjit
    def _jac(u, logk, logf, t, c, q):
        row1 = [-1 / 2 * np.exp(logk) / np.sqrt(u), -np.sqrt(u) * np.exp(logk), 0]
        row2 = [
            (1 / 2) * c * np.exp(logf) / u ** (3 / 2)
            + np.exp(logk) / u**2
            + (1 / 2) / np.sqrt(u),
            -np.exp(logk) / u,
            -c * np.exp(logf) / np.sqrt(u),
        ]
        row3 = [
            1.4249999999999998 * c * u ** (-2.5)
            - 0.5 * c * u ** (-1.5) * np.exp(logk)
            - 0.04999999999999999 * c,
            c * np.exp(logk) / np.sqrt(u),
            0,
        ]
        return row1, row2, row3


class Hyperion(DynSys):
    @staticjit
    def _rhs(th, om, f, t, e, w0sq):
        r = (1 - e**2) / (1 + e * np.cos(f))
        thdot = om
        omdot = -(w0sq / (2 * r**3)) * np.sin(2 * (th - f))
        fdot = (1 + e * np.cos(f)) ** 2 / (1 - e**2) ** 1.5
        return thdot, omdot, fdot

    @staticjit
    def _postprocessing(th, om, f):
        return np.sin(2 * th), om, np.cos(f)

    @staticjit
    def _jac(th, om, f, t, e, w0sq):
        row1 = [0, 1, 0]
        row2 = [
            w0sq * (e * np.cos(f) + 1) ** 3 * np.cos(2 * f - 2 * th) / (e**2 - 1) ** 3,
            0,
            (1 / 2)
            * w0sq
            * (e * np.cos(f) + 1) ** 2
            * (
                3 * e * np.sin(f) * np.sin(2 * f - 2 * th)
                - 2 * (e * np.cos(f) + 1) * np.cos(2 * f - 2 * th)
            )
            / (e**2 - 1) ** 3,
        ]
        row3 = [0, 0, -2 * e * (1 - e**2) ** (-1.5) * (e * np.cos(f) + 1) * np.sin(f)]
        return row1, row2, row3


class KarneyIonHeating(DynSys):
    @staticjit
    def _rhs(y, v, ph, t, alpha, nu):
        ydot = v
        vdot = -y + alpha * np.cos(y - ph)
        phdot = nu
        return ydot, vdot, phdot

    @staticjit
    def _postprocessing(y, v, ph):
        return y, v, np.cos(ph)

    @staticjit
    def _jac(y, v, ph, t, alpha, nu):
        row1 = [0, 1, 0]
        row2 = [alpha * np.sin(ph - y) - 1, 0, -alpha * np.sin(ph - y)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class OneZonePulsator(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, eps, gam, om):
        r3p = (1 + x) ** 3 * (1 + z)
        al = a * (r3p - 1.2) + 21.6
        be = 3.6 * r3p * (r3p - 0.2)
        xdot = y
        ydot = om**2 * ((1 + x) ** 2 * (1 + z) - (1 + x) ** -2)
        zdot = -3 * gam * y * (1 + z) / (1 + x) - eps * (1 + x) ** -3 * (
            (1 + x) ** al * (1 + z) ** be - 1
        )
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, eps, gam, om):
        c0 = om**2
        c1 = x + 1
        c2 = c1**3
        c3 = c2 ** (-1.0)
        c4 = c1**4
        c5 = z + 1
        c6 = c1**2
        c7 = c1 ** (-1.0)
        c8 = c2 * c5
        c9 = a * (c8 - 1.2) + 21.6
        c10 = c5 * c6
        c11 = a * np.log(c1)
        c12 = c8 - 0.2
        c13 = 3.6 * c12
        c14 = c1**c9 * c5 ** (c13 * c8)
        c15 = np.log(c5)
        c16 = c3 * eps
        c17 = 3 * c7 * gam
        c18 = c13 * c2
        row1 = [0, 1, 0]
        row2 = [2 * c0 * c3 * (c4 * c5 + 1), 0, c0 * c6]
        row3 = [
            -c16
            * (
                c14 * c15 * (10.8 * c1**5 * c5**2 + 10.8 * c10 * c12)
                + c14 * (3 * c10 * c11 + c7 * c9)
            )
            + 3 * c5 * gam * y / c6
            + 3 * eps * (c14 - 1) / c4,
            -c17 * c5,
            -c16 * (c11 * c14 * c2 + c14 * (c15 * (3.6 * c1**6 * c5 + c18) + c18))
            - c17 * y,
        ]
        return row1, row2, row3


class DrivenAlfven(DynSys):
    @staticjit
    def _rhs(by, bz, th, t, a, lam, nu, om):
        b2 = by**2 + bz**2
        dhdby = (b2 - 1) * by - lam * (by - 1)
        dhdbz = (b2 - 1) * bz - lam * bz
        r1 = dhdbz + a * np.cos(th)
        r2 = -dhdby + a * np.sin(th)
        bydot = (r1 + nu * r2) / (1 + nu**2)
        bzdot = (r2 - nu * r1) / (1 + nu**2)
        thdot = om
        return bydot, bzdot, thdot

    @staticjit
    def _postprocessing(by, bz, th):
        return by, bz, np.sin(th)

    @staticjit
    def _jac(by, bz, th, t, a, lam, nu, om):
        row1 = [
            (2 * by * bz - nu * (3 * by**2 + bz**2 - lam - 1)) / (nu**2 + 1),
            (by**2 - 2 * by * bz * nu + 3 * bz**2 - lam - 1) / (nu**2 + 1),
            a * (nu * np.cos(th) - np.sin(th)) / (nu**2 + 1),
        ]
        row2 = [
            (-3 * by**2 - 2 * by * bz * nu - bz**2 + lam + 1) / (nu**2 + 1),
            (-2 * by * bz - nu * (by**2 + 3 * bz**2 - lam - 1)) / (nu**2 + 1),
            a * (nu * np.sin(th) + np.cos(th)) / (nu**2 + 1),
        ]
        row3 = [0, 0, 0]
        return row1, row2, row3


class BetaCellBurster(DynSys):
    @staticjit
    def _rhs(
        v,
        n,
        s,
        t,
        gca,
        gk,
        gs,
        sigma,
        tau,
        taus,
        thetam,
        thetan,
        thetas,
        vca,
        vk,
        vm,
        vn,
        vs,
    ):
        minf = 1 / (1 + np.exp((vm - v) / thetam))
        ninf = 1 / (1 + np.exp((vn - v) / thetan))
        sinf = 1 / (1 + np.exp((vs - v) / thetas))
        vdot = (-gca * minf * (v - vca) - gk * n * (v - vk) - gs * s * (v - vk)) / tau
        ndot = sigma * (ninf - n) / tau
        sdot = (sinf - s) / taus
        return vdot, ndot, sdot

    @staticjit
    def _jac(
        v,
        n,
        s,
        t,
        gca,
        gk,
        gs,
        sigma,
        tau,
        taus,
        thetam,
        thetan,
        thetas,
        vca,
        vk,
        vm,
        vn,
        vs,
    ):
        row1 = [
            (
                -gca / (np.exp((-v + vm) / thetam) + 1)
                - gca
                * (v - vca)
                * np.exp((-v + vm) / thetam)
                / (thetam * (np.exp((-v + vm) / thetam) + 1) ** 2)
                - gk * n
                - gs * s
            )
            / tau,
            gk * (-v + vk) / tau,
            gs * (-v + vk) / tau,
        ]
        row2 = [
            (1 / 4)
            * sigma
            / (tau * thetan * np.cosh((1 / 2) * (v - vn) / thetan) ** 2),
            -sigma / tau,
            0,
        ]
        row3 = [
            (1 / 4) / (taus * thetas * np.cosh((1 / 2) * (v - vs) / thetas) ** 2),
            0,
            -1 / taus,
        ]
        return row1, row2, row3


class GeneRegulatoryMotif(DynSys):
    @staticjit
    def _rhs(p1, p2, p3, t, h, k):
        a1 = p1**h / (p1**h + k**h)
        a3 = p3**h / (p3**h + k**h)
        r1 = k**h / (p1**h + k**h)
        r2 = k**h / (p2**h + k**h)
        r3 = k**h / (p3**h + k**h)
        p1dot = r2 - p1
        p2dot = a1 * r3 - p2
        p3dot = a3 * r1 - p3
        return p1dot, p2dot, p3dot

    @staticjit
    def _jac(p1, p2, p3, t, h, k):
        row1 = [-1, -h * k**h * p2 ** (h - 1) / (k**h + p2**h) ** 2, 0]
        row2 = [
            h
            * k**h
            * (p1 ** (h + 1) * (k**h + p1**h) - p1 ** (2 * h + 1))
            / (p1**2 * (k**h + p1**h) ** 2 * (k**h + p3**h)),
            -1,
            -h * k**h * p1**h * p3 ** (h - 1) / ((k**h + p1**h) * (k**h + p3**h) ** 2),
        ]
        row3 = [
            -h * k**h * p1 ** (h - 1) * p3**h / ((k**h + p1**h) ** 2 * (k**h + p3**h)),
            0,
            -h * k**h * p3 ** (2 * h) / (p3 * (k**h + p1**h) * (k**h + p3**h) ** 2)
            + h * k**h * p3**h / (p3 * (k**h + p1**h) * (k**h + p3**h))
            - 1,
        ]
        return row1, row2, row3


class ForcedGlycolysis(DynSys):
    @staticjit
    def _rhs(s, p, th, t, c, eps, etot, k, kd, kon, krp, krs, lallo, n, nu0, omega):
        alph = max(s, 0.0) / (krs + k / kon)
        gam = max(p, 0.0) / krp
        sig = (
            n
            * etot
            * alph
            * (1 + alph) ** (n - 1)
            * (1 + gam) ** n
            / (
                lallo * (1 + c * max(s, 0.0) / krs) ** n
                + (1 + alph) ** n * (1 + gam) ** n
            )
        )
        nu = max(nu0 * (1 + eps * np.sin(th)), 0.0)
        sdot = nu - k * sig
        pdot = k * sig - kd * p
        thdot = omega
        return sdot, pdot, thdot

    @staticjit
    def _jac(s, p, th, t, c, eps, etot, k, kd, kon, krp, krs, lallo, n, nu0, omega):
        # valid for s, p > 0 (the clipping in _rhs is inactive on the attractor)
        c0 = k / kon + krs
        c1 = 1.0 / c0
        c2 = c1 * n
        c3 = c / krs
        c4 = c3 * s + 1
        c5 = c4**n * lallo
        c6 = 1.0 / krp
        c7 = c6 * p + 1
        c8 = c7**n
        c9 = c1 * s
        c10 = c9 + 1
        c11 = c10**n
        c12 = c11 * c8
        c13 = c12 + c5
        c14 = n - 1
        c15 = c10**c14 * etot * k
        c16 = c15 * c8
        c17 = c16 / c13
        c18 = 1.0 / c10
        c19 = c9 / c13**2
        c20 = (
            c16 * c19 * n * (-c12 * c18 * c2 - c3 * c5 * n / c4)
            + c17 * c2
            + c14 * c17 * c18 * n * s / c0**2
        )
        c21 = c6 * n**2 / c7
        c22 = c11 * c15 * c19 * c21 * c7 ** (2 * n) - c17 * c21 * c9
        if nu0 * (1 + eps * np.sin(th)) > 0:
            dnu = eps * nu0 * np.cos(th)
        else:
            dnu = 0.0
        row1 = [-c20, c22, dnu]
        row2 = [c20, -c22 - kd, 0.0]
        row3 = [0.0, 0.0, 0.0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(s, p, th):
        return s, p, np.sin(th)


class DengLoladze(DynSys):
    @staticjit
    def _rhs(x, y1, y2, t, a1, a2, c1, c2, d1, d2, e1, e2, k, p, q, r, s1, s2):
        free = p - s1 * y1 - s2 * y2
        g1 = c1 * x * y1 / (a1 + x)
        g2 = c2 * x * y2 / (a2 + x)
        xdot = r * x * (1 - x / min(k, free / q)) - g1 - g2
        y1dot = e1 * min(1.0, free / (x * s1)) * g1 - d1 * y1
        y2dot = e2 * min(1.0, free / (x * s2)) * g2 - d2 * y2
        return xdot, y1dot, y2dot

    @staticjit
    def _jac(x, y1, y2, t, a1, a2, c1, c2, d1, d2, e1, e2, k, p, q, r, s1, s2):
        free = p - s1 * y1 - s2 * y2
        # carrying capacity min(k, free / q)
        if free / q < k:
            keff, ky1, ky2 = free / q, -s1 / q, -s2 / q
        else:
            keff, ky1, ky2 = k, 0.0, 0.0
        g1 = c1 * x * y1 / (a1 + x)
        g2 = c2 * x * y2 / (a2 + x)
        g1x, g1y1 = c1 * y1 * a1 / (a1 + x) ** 2, c1 * x / (a1 + x)
        g2x, g2y2 = c2 * y2 * a2 / (a2 + x) ** 2, c2 * x / (a2 + x)
        # food quality min(1, free / (x si))
        if free / (x * s1) < 1:
            q1, q1x, q1y1, q1y2 = (
                free / (x * s1),
                -free / (x**2 * s1),
                -1 / x,
                -s2 / (x * s1),
            )
        else:
            q1, q1x, q1y1, q1y2 = 1.0, 0.0, 0.0, 0.0
        if free / (x * s2) < 1:
            q2, q2x, q2y1, q2y2 = (
                free / (x * s2),
                -free / (x**2 * s2),
                -s1 / (x * s2),
                -1 / x,
            )
        else:
            q2, q2x, q2y1, q2y2 = 1.0, 0.0, 0.0, 0.0
        cap = r * x**2 / keff**2
        row1 = [r * (1 - 2 * x / keff) - g1x - g2x, cap * ky1 - g1y1, cap * ky2 - g2y2]
        row2 = [
            e1 * (q1x * g1 + q1 * g1x),
            e1 * (q1y1 * g1 + q1 * g1y1) - d1,
            e1 * q1y2 * g1,
        ]
        row3 = [
            e2 * (q2x * g2 + q2 * g2x),
            e2 * q2y1 * g2,
            e2 * (q2y2 * g2 + q2 * g2y2) - d2,
        ]
        return row1, row2, row3


class ArneodoVolterra(DynSys):
    @staticjit
    def _rhs(x1, x2, x3, t, a11, a12, a13, a21, a22, a23, a31, a32, a33, r1, r2, r3):
        x1dot = x1 * (r1 - a11 * x1 - a12 * x2 - a13 * x3)
        x2dot = x2 * (r2 - a21 * x1 - a22 * x2 - a23 * x3)
        x3dot = x3 * (r3 - a31 * x1 - a32 * x2 - a33 * x3)
        return x1dot, x2dot, x3dot

    @staticjit
    def _jac(x1, x2, x3, t, a11, a12, a13, a21, a22, a23, a31, a32, a33, r1, r2, r3):
        row1 = [-2 * a11 * x1 - a12 * x2 - a13 * x3 + r1, -a12 * x1, -a13 * x1]
        row2 = [-a21 * x2, -a21 * x1 - 2 * a22 * x2 - a23 * x3 + r2, -a23 * x2]
        row3 = [-a31 * x3, -a32 * x3, -a31 * x1 - a32 * x2 - 2 * a33 * x3 + r3]
        return row1, row2, row3


class RedQueen(DynSys):
    @staticjit
    def _rhs(
        x1,
        x2,
        x3,
        t,
        a21,
        a22,
        a23,
        a24,
        a25,
        a31,
        a32,
        a33,
        a34,
        a35,
        c0,
        c1,
        c2,
        d2,
        d3,
        e2,
        e3,
        k1,
        k2,
        k3,
        r,
    ):
        c = c0 + c2 * (x1 - c1) ** 2
        u = (x1 - a24) / a21
        v = (x2 - a25) / a22
        aa2 = np.exp(-(u**2) + 2 * a23 * u * v - v**2)
        w = (x2 - a34) / a31
        z = (x3 - a35) / a32
        aa3 = np.exp(-(w**2) + 2 * a33 * w * z - z**2)
        n2 = d3 / (e3 * aa3)
        n1 = (r - aa2 * n2) / c
        n3 = (e2 * aa2 * n1 - d2) / aa3
        x1dot = (
            k1 * n1 * (-2 * c2 * (x1 - c1) * n1 - (2 * aa2 / a21) * (a23 * v - u) * n2)
        )
        x2dot = (
            k2
            * n2
            * (
                e2 * (2 * aa2 / a22) * (a23 * u - v) * n1
                - (2 * aa3 / a31) * (a33 * z - w) * n3
            )
        )
        x3dot = k3 * n3 * (e3 * (2 * aa3 / a32) * (a33 * w - z) * n2)
        return x1dot, x2dot, x3dot

    @staticjit
    def _jac(
        x1,
        x2,
        x3,
        t,
        a21,
        a22,
        a23,
        a24,
        a25,
        a31,
        a32,
        a33,
        a34,
        a35,
        c0,
        c1,
        c2,
        d2,
        d3,
        e2,
        e3,
        k1,
        k2,
        k3,
        r,
    ):
        c3 = -x1
        c4 = -c1 - c3
        c5 = c0 + c2 * c4**2
        c6 = c5 ** (-1.0)
        c7 = a21 ** (-2.0)
        c8 = -a24 - c3
        c9 = c7 * c8**2
        c10 = a22 ** (-2.0)
        c11 = -x2
        c12 = -a25 - c11
        c13 = c10 * c12**2
        c14 = a21 ** (-1.0)
        c15 = a22 ** (-1.0)
        c16 = np.exp(2 * a23 * c12 * c14 * c15 * c8 - c13 - c9)
        c17 = a31 ** (-2.0)
        c18 = -a34 - c11
        c19 = c17 * c18**2
        c20 = a32 ** (-2.0)
        c21 = -a35 + x3
        c22 = c20 * c21**2
        c23 = a31 ** (-1.0)
        c24 = c18 * c23
        c25 = a32 ** (-1.0)
        c26 = c21 * c25
        c27 = a33 * c26
        c28 = c24 * c27
        c29 = c19 + c22 - 2 * c28
        c30 = np.exp(c29)
        c31 = e3 ** (-1.0)
        c32 = c30 * c31 * d3
        c33 = c16 * c32
        c34 = -c33 + r
        c35 = c34 * c6
        c36 = 2 * c2
        c37 = c35 * c36
        c38 = 2 * c33
        c39 = -2 * x1
        c40 = c34 * (-2 * c1 - c39) / c5**2
        c41 = c12 * c15
        c42 = a23 * c41
        c43 = 2 * c14 * c42 - c7 * (-2 * a24 - c39)
        c44 = c14 * (-c14 * c8 + c42)
        c45 = c38 * c44
        c46 = c43 * c6
        c47 = c36 * c4
        c48 = c33 * c47
        c49 = k1 * (-c37 * c4 - c45)
        c50 = c2 * c40
        c51 = c33 * c49
        c52 = -2 * x2
        c53 = -2 * a34 - c52
        c54 = c23 * c27
        c55 = -c17 * c53 + 2 * c54
        c56 = -c55
        c57 = c33 * c56
        c58 = 2 * a23 * c14 * c15 * c8 - c10 * (-2 * a25 - c52)
        c59 = c33 * c58
        c60 = -c57 - c59
        c61 = c6 * c60
        c62 = 2 * c15
        c63 = 2 * c44
        c64 = c20 * (-2 * a35 + 2 * x3)
        c65 = a33 * c24 * c25
        c66 = c64 - 2 * c65
        c67 = c6 * c66
        c68 = c16 * e2
        c69 = np.exp(4 * a23 * c12 * c14 * c15 * c8 - 2 * c13 - 2 * c9)
        c70 = c69 * e2
        c71 = c32 * c70
        c72 = c46 * c71
        c73 = c16 * c34 * c43 * c6 * e2 - c50 * c68 - c72
        c74 = np.exp(-c29)
        c75 = 2 * c74
        c76 = c30 * c75
        c77 = -c24 + c27
        c78 = c23 * c77
        c79 = c76 * c78
        c80 = a23 * c14 * c8 - c41
        c81 = c62 * c80
        c82 = c32 * k2
        c83 = c16 * c34 * c6 * e2 - d2
        c84 = c35 * c68
        c85 = c58 * c84 + c61 * c68
        c86 = c76 * c83
        c87 = c78 * c86
        c88 = c82 * (2 * c15 * c16 * c34 * c6 * c80 * e2 - c87)
        c89 = a33 * c23 * c25
        c90 = -c66
        c91 = np.exp(2 * c19 + 2 * c22 - 4 * c28)
        c92 = c75 * c91 * d3 * k3
        c93 = a33 * c18 * c23 - c26
        c94 = c25 * c93
        c95 = c92 * c94
        c96 = c83 * c92
        c97 = c94 * c96
        row1 = [
            c34
            * c6
            * k1
            * (2 * c2**2 * c4 * c40 - c37 + c38 * c7 - c43 * c45 + c46 * c48)
            - c46 * c51
            - c49 * c50,
            c35 * k1 * (-a23 * c14 * c33 * c62 - c47 * c61 - c57 * c63 - c59 * c63)
            + c49 * c61,
            c34 * c6 * k1 * (-c45 * c66 + c48 * c67) - c51 * c67,
        ]
        row2 = [
            c82
            * (
                2 * a23 * c14 * c15 * c16 * c34 * c6 * e2
                + 2 * c15 * c16 * c34 * c43 * c6 * c80 * e2
                - c15 * c36 * c40 * c68 * c80
                - c72 * c81
                - c73 * c79
            ),
            c56 * c88
            + c82
            * (
                -2 * c10 * c84
                + 2 * c15 * c16 * c34 * c58 * c6 * c80 * e2
                + 2 * c15 * c16 * c6 * c60 * c80 * e2
                + 2 * c17 * c30 * c74 * c83
                - c55 * c87
                - c56 * c87
                - c79 * c85
            ),
            c66 * c88
            + c82
            * (
                2 * c23 * c31 * c6 * c66 * c69 * c74 * c77 * c91 * d3 * e2
                - c66 * c87
                - c67 * c71 * c81
                - c86 * c89
                - c87 * c90
            ),
        ]
        row3 = [
            c73 * c95,
            c55 * c97 + c85 * c95 + c89 * c96 + c97 * (2 * c17 * c53 - 4 * c54),
            -c20 * c96
            + 2 * c25 * c74 * c83 * c90 * c91 * c93 * d3 * k3
            + 2 * c25 * c74 * c83 * c91 * c93 * d3 * k3 * (2 * c64 - 4 * c65)
            - c31 * c67 * c70 * c76 * c91 * c94 * d3**2 * k3,
        ]
        return row1, row2, row3


class InjectedLaser(DynSys):
    @staticjit
    def _rhs(x, y, n, t, alpha, b, gamma, k, omega):
        xdot = k + n * x / 2 - (alpha * n / 2 - omega) * y
        ydot = n * y / 2 + (alpha * n / 2 - omega) * x
        ndot = -2 * gamma * n - (1 + 2 * b * n) * (x**2 + y**2 - 1)
        return xdot, ydot, ndot

    @staticjit
    def _jac(x, y, n, t, alpha, b, gamma, k, omega):
        row1 = [
            (1 / 2) * n,
            -1 / 2 * alpha * n + omega,
            -1 / 2 * alpha * y + (1 / 2) * x,
        ]
        row2 = [
            (1 / 2) * alpha * n - omega,
            (1 / 2) * n,
            (1 / 2) * alpha * x + (1 / 2) * y,
        ]
        row3 = [
            -2 * x * (2 * b * n + 1),
            -2 * y * (2 * b * n + 1),
            -2 * b * (x**2 + y**2 - 1) - 2 * gamma,
        ]
        return row1, row2, row3


class ConsecutiveReactionCSTR(DynSys):
    @staticjit
    def _rhs(x, y, temp, t, alpha, b, beta, da, kappa, s, tc):
        r1 = da * x * np.exp(temp)
        r2 = da * s * y * np.exp(kappa * temp)
        xdot = 1 - x - r1
        ydot = -y + r1 - r2
        tempdot = -temp - beta * (temp - tc) + b * r1 - b * alpha * r2
        return xdot, ydot, tempdot

    @staticjit
    def _jac(x, y, temp, t, alpha, b, beta, da, kappa, s, tc):
        row1 = [-da * np.exp(temp) - 1, 0, -da * x * np.exp(temp)]
        row2 = [
            da * np.exp(temp),
            -da * s * np.exp(kappa * temp) - 1,
            da * (-kappa * s * y * np.exp(kappa * temp) + x * np.exp(temp)),
        ]
        row3 = [
            b * da * np.exp(temp),
            -alpha * b * da * s * np.exp(kappa * temp),
            -alpha * b * da * kappa * s * y * np.exp(kappa * temp)
            + b * da * x * np.exp(temp)
            - beta
            - 1,
        ]
        return row1, row2, row3


class SaturableAbsorberLaser(DynSys):
    @staticjit
    def _rhs(i, u, w, t, a, abar, b, eps, pump):
        idot = i * (u - abar / (1 + a * i) - 1)
        udot = eps * (w - u * (1 + i))
        wdot = eps * (pump + b * u - w)
        return idot, udot, wdot

    @staticjit
    def _jac(i, u, w, t, a, abar, b, eps, pump):
        row1 = [-abar / (a * i + 1) ** 2 + u - 1, i, 0]
        row2 = [-eps * u, -eps * (i + 1), eps]
        row3 = [0, b * eps, -eps]
        return row1, row2, row3


class GearRattle(DynSys):
    @staticjit
    def _rhs(q, v, th, t, fah, fm, omega, zeta):
        if q > 1:
            fq = q - 1
        elif q < -1:
            fq = q + 1
        else:
            fq = 0.0
        qdot = v
        vdot = -2 * zeta * v - fq + fm + fah * omega**2 * np.sin(th)
        thdot = omega
        return qdot, vdot, thdot

    @staticjit
    def _jac(q, v, th, t, fah, fm, omega, zeta):
        if q > 1 or q < -1:
            dfq = 1.0
        else:
            dfq = 0.0
        row1 = [0.0, 1.0, 0.0]
        row2 = [-dfq, -2 * zeta, fah * omega**2 * np.cos(th)]
        row3 = [0.0, 0.0, 0.0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(q, v, th):
        return q, v, np.sin(th)


class ArecchiFeedbackLaser(DynSys):
    @staticjit
    def _rhs(x, y, z, t, b0, b1, eps1, eps2, k1, p0):
        xdot = -eps1 * x * (1 + k1 * z**2 - p0 * y)
        ydot = -y - x * y + 1
        zdot = -eps2 * (z - b0 + b1 * x)
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, b0, b1, eps1, eps2, k1, p0):
        row1 = [eps1 * (-k1 * z**2 + p0 * y - 1), eps1 * p0 * x, -2 * eps1 * k1 * x * z]
        row2 = [-y, -x - 1, 0]
        row3 = [-b1 * eps2, 0, -eps2]
        return row1, row2, row3


class ACTAttractor(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, delta, mu):
        xdot = alpha * (x - y)
        ydot = -4 * alpha * y + x * z + mu * x**3
        zdot = -delta * alpha * z + x * y + beta * z**2
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, beta, delta, mu):
        row1 = [alpha, -alpha, 0]
        row2 = [3*mu*x**2 + z, -4*alpha, x]
        row3 = [y, x, -alpha*delta + 2*beta*z]
        return row1, row2, row3


class AFMCantilever(DynSys):
    @staticjit
    def _rhs(x, y, th, t, a0, d0, delta, gam, omega, sig):
        r = a0 + x
        xdot = y
        ydot = -x - delta * y - d0 / r**2 + sig**6 * d0 / (30 * r**8) + gam * np.cos(th)
        thdot = omega
        return xdot, ydot, thdot

    @staticjit
    def _postprocessing(x, y, th):
        return x, y, np.sin(th)

    @staticjit
    def _jac(x, y, th, t, a0, d0, delta, gam, omega, sig):
        row1 = [0, 1, 0]
        row2 = [-4/15*d0*sig**6/(a0 + x)**9 + 2*d0/(a0 + x)**3 - 1, -delta, -gam*np.sin(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class AzizAlaouiLeslieGower(DynSys):
    @staticjit
    def _rhs(lx, ly, lz, t, a0, a1, b0, c3, d0, d1, d2, v0, v1, v2, v3):
        # logarithms of the prey, specialist predator and top predator densities
        x = np.exp(lx)
        y = np.exp(ly)
        z = np.exp(lz)
        lxdot = a0 - b0 * x - v0 * y / (d0 + x)
        lydot = -a1 + v1 * x / (d0 + x) - v2 * z / (d1 + y)
        lzdot = z * (c3 - v3 / (d2 + y))
        return lxdot, lydot, lzdot

    @staticjit
    def _jac(lx, ly, lz, t, a0, a1, b0, c3, d0, d1, d2, v0, v1, v2, v3):
        row1 = [-b0*np.exp(lx) + v0*np.exp(lx + ly)/(d0 + np.exp(lx))**2, -v0*np.exp(ly)/(d0 + np.exp(lx)), 0]
        row2 = [d0*v1*np.exp(lx)/(d0 + np.exp(lx))**2, v2*np.exp(ly + lz)/(d1 + np.exp(ly))**2, -v2*np.exp(lz)/(d1 + np.exp(ly))]
        row3 = [0, v3*np.exp(ly + lz)/(d2 + np.exp(ly))**2, (c3 - v3/(d2 + np.exp(ly)))*np.exp(lz)]
        return row1, row2, row3


class BateHilker(DynSys):
    @staticjit
    def _rhs(ln, lp, li, t, dbeta, h, m, mu, r):
        # logarithms of prey density, predator density and infected fraction of predators
        n = np.exp(ln)
        pr = np.exp(lp)
        i = np.exp(li)
        fn = n / (h + n)
        lndot = r * (1 - n) - pr / (h + n)
        lpdot = fn - m - mu * i
        lidot = dbeta * (1 - i) - fn
        return lndot, lpdot, lidot

    @staticjit
    def _jac(ln, lp, li, t, dbeta, h, m, mu, r):
        row1 = [-r*np.exp(ln) + np.exp(ln + lp)/(h + np.exp(ln))**2, -np.exp(lp)/(h + np.exp(ln)), 0]
        row2 = [h*np.exp(ln)/(h**2 + 2*h*np.exp(ln) + np.exp(2*ln)), 0, -mu*np.exp(li)]
        row3 = [-h*np.exp(ln)/(h + np.exp(ln))**2, 0, -dbeta*np.exp(li)]
        return row1, row2, row3


class BlasiusHuppertStone(DynSys):
    @staticjit
    def _rhs(u, v, w, t, a, al1, al2, b, c, k1, k2, wstar):
        f1 = u * v / (1 + k1 * u)
        f2 = v * w / (1 + k2 * v)
        udot = a * u - al1 * f1
        vdot = -b * v + al1 * f1 - al2 * f2
        wdot = -c * (w - wstar) + al2 * f2
        return udot, vdot, wdot

    @staticjit
    def _jac(u, v, w, t, a, al1, al2, b, c, k1, k2, wstar):
        row1 = [a - al1*v/(k1*u + 1)**2, -al1*u/(k1*u + 1), 0]
        row2 = [al1*v/(k1*u + 1)**2, al1*u/(k1*u + 1) + al2*k2*v*w/(k2*v + 1)**2 - al2*w/(k2*v + 1) - b, -al2*v/(k2*v + 1)]
        row3 = [0, al2*w/(k2*v + 1)**2, al2*v/(k2*v + 1) - c]
        return row1, row2, row3


class CarrollRossler(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, gam, gam_x, lam, mu):
        gx = mu * np.maximum(x - 3, 0.0)
        xdot = -(gam_x * x + beta * y + lam * z)
        ydot = x + (gam - 0.02) * y
        zdot = gx - z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, beta, gam, gam_x, lam, mu):
        row1 = [-gam_x, -beta, -lam]
        row2 = [1, gam - 0.02, 0]
        row3 = [mu * 0.5 * (1 + np.sign(x - 3)), 0, -1]
        return row1, row2, row3


class ChuaCubic(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, c):
        xdot = alpha * (y - x**3 - c * x)
        ydot = x - y + z
        zdot = -beta * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, beta, c):
        row1 = [-alpha*(c + 3*x**2), alpha, 0]
        row2 = [1, -1, 1]
        row3 = [0, -beta, 0]
        return row1, row2, row3



class ConsecutiveAutocatalator(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, dax, day, daz):
        rx = dax * x * z
        ry = day * y * z**2
        xdot = 1 - x - rx
        ydot = rx - y - ry
        zdot = 1 - (1 + daz) * z + a * (rx + ry)
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, dax, day, daz):
        row1 = [-dax*z - 1, 0, -dax*x]
        row2 = [dax*z, -day*z**2 - 1, dax*x - 2*day*y*z]
        row3 = [a*dax*z, a*day*z**2, a*(dax*x + 2*day*y*z) - daz - 1]
        return row1, row2, row3


class DarukaDitlevsen(DynSys):
    @staticjit
    def _rhs(x, tau, th, t, a, al, kap, lam, omega):
        xdot = lam * tau
        taudot = -(al - x + x**3) - kap * tau - a * x * np.sin(th)
        thdot = omega
        return xdot, taudot, thdot

    @staticjit
    def _postprocessing(x, tau, th):
        return x, tau, np.sin(th)

    @staticjit
    def _jac(x, tau, th, t, a, al, kap, lam, omega):
        row1 = [0, lam, 0]
        row2 = [-a*np.sin(th) - 3*x**2 + 1, -kap, -a*x*np.cos(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class DoubleScrollJerk(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = z
        zdot = -a * (z + y + x - np.sign(x))
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-a, -a, -a]
        return row1, row2, row3


class DynamicHallInstability(DynSys):
    @staticjit
    def _rhs(ex, ez, n, t, beta, e0, eta, eth, exs, kappa, na, nd, nstar, t0, wfac, x0, xs):
        e2 = ex**2 + ez**2
        w = wfac * e2
        r = w / (2 * eth)
        xion = x0 * np.sqrt(r) / (1 + r) / (1 + np.exp((eth - 0.5 * w) / (2 * e0 / 3 + eta * w / 3)))
        trec = t0 * (e0 / (e0 + 0.5 * (eta + 1) * w)) ** 1.5
        exdot = kappa * ((1 + beta**2) * nstar * exs - n * ex + beta * n * ez)
        ezdot = kappa * (-n * ez - beta * n * ex)
        ndot = 1e-6 * xs * (na - n) - 10 * trec * n * (n + nd) + 10 * xion * n * (na - n)
        return exdot, ezdot, ndot

    @staticjit
    def _jac(ex, ez, n, t, beta, e0, eta, eth, exs, kappa, na, nd, nstar, t0, wfac, x0, xs):
        c0 = kappa*n
        c1 = -c0
        c2 = beta*c0
        c3 = ex*wfac
        c4 = 0.5*eta + 0.5
        c5 = ex**2 + ez**2
        c6 = c5*wfac
        c7 = (c4*c6 + e0)**(-1.0)
        c8 = (c7*e0)**1.5
        c9 = n + nd
        c10 = 30.0*c4*c7*c8*c9*n*t0
        c11 = (1/3)*c6*eta + (2/3)*e0
        c12 = c11**(-1.0)
        c13 = -0.5*c6 + eth
        c14 = np.exp(c12*c13)
        c15 = c14 + 1
        c16 = c15**(-1.0)
        c17 = c16*ex
        c18 = np.sqrt(c5)
        c19 = np.sqrt(2)
        c20 = wfac/eth
        c21 = np.sqrt(c20)
        c22 = (1/2)*c20*c5 + 1
        c23 = c22**(-1.0)
        c24 = -n + na
        c25 = 5*c19*c21*c23*c24*n*x0
        c26 = c25/c18
        c27 = 1.0*c12
        c28 = (2/3)*c13*eta/c11**2
        c29 = c14*c18*c25/c15**2
        c30 = 5*c18*c19*c20*c21*c24*n*x0/c22**2
        c31 = ez*wfac
        c32 = c16*ez
        c33 = 10*c8*t0
        row1 = [c1, c2, kappa*(beta*ez - ex)]
        row2 = [-c2, c1, -kappa*(beta*ex + ez)]
        row3 = [c10*c3 + c17*c26 - c17*c30 - c29*(-c27*c3 - c28*c3), c10*c31 + c26*c32 - c29*(-c27*c31 - c28*c31) - c30*c32, 5*c16*c18*c19*c21*c23*c24*x0 - 5*c16*c18*c19*c21*c23*n*x0 - c33*c9 - c33*n - 1e-06*xs]
        return row1, row2, row3


class DynasticCycle(DynSys):
    @staticjit
    def _rhs(x, y, z, t, b, d, e, g, h, q, r):
        xdot = x * (1 - x - y / (b + x) - h * z)
        ydot = q * y * (e * x / (b + x) - 1 - z / (d + y))
        zdot = r * (x * y / (b + x) - g * z)
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, b, d, e, g, h, q, r):
        row1 = [-b*y/(b + x)**2 - h*z - 2*x + 1, -x/(b + x), -h*x]
        row2 = [b*e*q*y/(b + x)**2, q*y*z/(d + y)**2 + q*(e*x/(b + x) - z/(d + y) - 1), -q*y/(d + y)]
        row3 = [b*r*y/(b + x)**2, r*x/(b + x), -g*r]
        return row1, row2, row3


class ElectrostaticMEMS(DynSys):
    @staticjit
    def _rhs(x, y, th, t, a, beta, gam, mu, omega):
        xdot = y
        ydot = (
            -mu * y
            - x
            - beta * x**3
            + gam * ((1 + a * np.sin(th)) ** 2 / (1 - x) ** 2 - 1 / (1 + x) ** 2)
        )
        thdot = omega
        return xdot, ydot, thdot

    @staticjit
    def _postprocessing(x, y, th):
        return x, y, np.sin(th)

    @staticjit
    def _jac(x, y, th, t, a, beta, gam, mu, omega):
        row1 = [0, 1, 0]
        row2 = [-3*beta*x**2 + 2*gam/(x + 1)**3 - 2*gam*(a*np.sin(th) + 1)**2/(x - 1)**3 - 1, -mu, 2*a*gam*(a*np.sin(th) + 1)*np.cos(th)/(x - 1)**2]
        row3 = [0, 0, 0]
        return row1, row2, row3


class ForcedMPR(DynSys):
    @staticjit
    def _rhs(r, v, th, t, delta, eta, i0, j, omega):
        rdot = delta / np.pi + 2 * r * v
        vdot = v**2 + eta + j * r + i0 * np.sin(th) - (np.pi * r) ** 2
        thdot = omega
        return rdot, vdot, thdot

    @staticjit
    def _postprocessing(r, v, th):
        return r, v, np.sin(th)

    @staticjit
    def _jac(r, v, th, t, delta, eta, i0, j, omega):
        row1 = [2*v, 2*r, 0]
        row2 = [j - 2*np.pi**2*r, 2*v, i0*np.cos(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class ForcedMorrisLecar(DynSys):
    @staticjit
    def _rhs(
        v, w, th, t,
        a1, c, f1, gca, gk, gl, idc, phi, v1, v2, v3, v4, vca, vk, vl,
    ):
        minf = 0.5 * (1 + np.tanh((v - v1) / v2))
        winf = 0.5 * (1 + np.tanh((v - v3) / v4))
        vdot = (
            -gca * minf * (v - vca)
            - gk * w * (v - vk)
            - gl * (v - vl)
            + idc
            + a1 * np.sin(th)
        ) / c
        wdot = phi * (winf - w) * np.cosh((v - v3) / (2 * v4))
        thdot = 2 * np.pi * f1
        return vdot, wdot, thdot

    @staticjit
    def _postprocessing(v, w, th):
        return v, w, np.sin(th)

    @staticjit
    def _jac(v, w, th, t, a1, c, f1, gca, gk, gl, idc, phi, v1, v2, v3, v4, vca, vk, vl,):
        row1 = [-(0.5*gca*(v - vca)/np.cosh((v - v1)/v2)**2 + v2*(0.5*gca*(np.tanh((v - v1)/v2) + 1) + gk*w + gl))/(c*v2), gk*(-v + vk)/c, a1*np.cos(th)/c]
        row2 = [(1/2)*phi*((-w + 0.5*np.tanh((v - v3)/v4) + 0.5)*np.sinh((1/2)*(v - v3)/v4) + 1.0*np.cosh((1/2)*(v - v3)/v4)/np.cosh((v - v3)/v4)**2)/v4, -phi*np.cosh((1/2)*(v - v3)/v4), 0]
        row3 = [0, 0, 0]
        return row1, row2, row3


class ForcedWilsonCowan(DynSys):
    @staticjit
    def _rhs(x, y, th, t, a, b, bf, c, d, omega, rx, ry):
        xdot = -x + 1 / (1 + np.exp(-(a * x - b * y + rx + bf * np.cos(th))))
        ydot = -y + 1 / (1 + np.exp(-(c * x - d * y + ry)))
        thdot = omega
        return xdot, ydot, thdot

    @staticjit
    def _postprocessing(x, y, th):
        return x, y, np.sin(th)

    @staticjit
    def _jac(x, y, th, t, a, b, bf, c, d, omega, rx, ry):
        row1 = [a*np.exp(-a*x + b*y - bf*np.cos(th) - rx)/(np.exp(-a*x + b*y - bf*np.cos(th) - rx) + 1)**2 - 1, -1/4*b/np.cosh((1/2)*a*x - 1/2*b*y + (1/2)*bf*np.cos(th) + (1/2)*rx)**2, -1/4*bf*np.sin(th)/np.cosh((1/2)*a*x - 1/2*b*y + (1/2)*bf*np.cos(th) + (1/2)*rx)**2]
        row2 = [(1/4)*c/np.cosh((1/2)*c*x - 1/2*d*y + (1/2)*ry)**2, -1/4*d/np.cosh((1/2)*c*x - 1/2*d*y + (1/2)*ry)**2 - 1, 0]
        row3 = [0, 0, 0]
        return row1, row2, row3


class GaspardNicolis(DynSys):
    @staticjit
    def _rhs(x, y, z, t, k1, k10, k2, k3, k4, k5, k6, k7, k8, k9, km, km9):
        w1 = k1 * x - km * x**2
        w2 = k2 * x**2 - km * x**3
        w3 = k3 * x * y - km * y
        w4 = k4 * x * z - km * z
        w5 = k5 * x * y - km * x * y**2
        w6 = k6 * y * z - km * y**2 * z
        w7 = k7 * y - km
        w8 = k8 * x - km * x * z
        w9 = k9 * z**2 - km9 * z**3
        w10 = k10 * z - km
        xdot = w1 + w2 - w3 - w4
        ydot = w5 + w6 - w7
        zdot = w8 + w9 - w10
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, k1, k10, k2, k3, k4, k5, k6, k7, k8, k9, km, km9):
        row1 = [k1 + 2*k2*x - k3*y - k4*z - 3*km*x**2 - 2*km*x, -k3*x + km, -k4*x + km]
        row2 = [y*(k5 - km*y), k5*x + k6*z - k7 - 2*km*x*y - 2*km*y*z, y*(k6 - km*y)]
        row3 = [k8 - km*z, 0, -k10 + 2*k9*z - km*x - 3*km9*z**2]
        return row1, row2, row3


class GilpinSpiral(DynSys):
    @staticjit
    def _rhs(lx, ly, lz, t, a11, a12, a13, a21, a22, a23, a31, a32, r1, r2, r3):
        # logarithms of the two prey and the predator
        x = np.exp(lx)
        y = np.exp(ly)
        z = np.exp(lz)
        lxdot = r1 - a11 * x - a12 * y - a13 * z
        lydot = r2 - a21 * x - a22 * y - a23 * z
        lzdot = -r3 + a31 * x + a32 * y
        return lxdot, lydot, lzdot

    @staticjit
    def _jac(lx, ly, lz, t, a11, a12, a13, a21, a22, a23, a31, a32, r1, r2, r3):
        row1 = [-a11*np.exp(lx), -a12*np.exp(ly), -a13*np.exp(lz)]
        row2 = [-a21*np.exp(lx), -a22*np.exp(ly), -a23*np.exp(lz)]
        row3 = [a31*np.exp(lx), a32*np.exp(ly), 0]
        return row1, row2, row3


class HaimNickel(DynSys):
    @staticjit
    def _rhs(e, th, eta, t, a, b, c, ch, g1, g2, r, v):
        ee = np.exp(e)
        i_f = (ch * np.exp(0.5 * e) / (1.0 + ch * ee) + a * ee) * (1.0 - th)
        edot = (v - e) / r - i_f
        thdot = (np.exp(0.5 * e) / (1.0 + ch * ee) * (1.0 - th) - b * ch * eta * ee) / g1
        etadot = (np.exp(2 * e) * (th - eta) - c * ch * eta * ee) / g2
        return edot, thdot, etadot

    @staticjit
    def _jac(e, th, eta, t, a, b, c, ch, g1, g2, r, v):
        c0 = 1.0 - th
        c1 = np.exp(e)
        c2 = a*c1
        c3 = c1*ch
        c4 = c3 + 1.0
        c5 = np.exp(1.5*e)/c4**2
        c6 = c4**(-1.0)
        c7 = np.exp(0.5*e)
        c8 = c6*c7
        c9 = c8*ch
        c10 = g1**(-1.0)
        c11 = b*c3
        c12 = c*ch
        c13 = g2**(-1.0)
        c14 = c1*c13
        row1 = [-c0*(c2 - c5*ch**2 + 0.5*c9) - 1/r, c2 + c9, 0]
        row2 = [c10*(-c0*c5*ch + 0.5*c0*c6*c7 - c11*eta), -c10*c8, -c10*c11]
        row3 = [c14*(2*c1*(-eta + th) - c12*eta), c13*np.exp(2*e), c14*(-c1 - c12)]
        return row1, row2, row3


class IceAgeVdPDuffing(DynSys):
    @staticjit
    def _rhs(x, y, th, t, alpha, beta, c3, gamma, k1, kappa, omega, tau):
        g = -y + c3 * y**3
        xdot = (gamma * k1 * np.sin(th) - beta - g) / (tau * kappa)
        ydot = alpha * (y - y**3 / 3 + x) / (tau * kappa)
        thdot = omega
        return xdot, ydot, thdot

    @staticjit
    def _postprocessing(x, y, th):
        return x, y, np.sin(th)

    @staticjit
    def _jac(x, y, th, t, alpha, beta, c3, gamma, k1, kappa, omega, tau):
        row1 = [0, (-3*c3*y**2 + 1)/(kappa*tau), gamma*k1*np.cos(th)/(kappa*tau)]
        row2 = [alpha/(kappa*tau), alpha*(1 - y**2)/(kappa*tau), 0]
        row3 = [0, 0, 0]
        return row1, row2, row3


class IkeuchiTomita(DynSys):
    @staticjit
    def _rhs(lx, lw, th, t, a, b, k, omega):
        # logarithms of the cold (x) and warm (w) phase fractions; hot fraction y = 1 - x - w
        x = np.exp(lx)
        w = np.exp(lw)
        y = 1 - x - w
        lxdot = a * w * (1 + k * np.cos(th)) / x - b * y**2
        lwdot = -a * (1 + k * np.cos(th)) + y
        thdot = omega
        return lxdot, lwdot, thdot

    @staticjit
    def _postprocessing(lx, lw, th):
        return lx, lw, np.sin(th)

    @staticjit
    def _jac(lx, lw, th, t, a, b, k, omega):
        row1 = [(-a*(k*np.cos(th) + 1)*np.exp(lw) - 2*b*(np.exp(lw) + np.exp(lx) - 1)*np.exp(2*lx))*np.exp(-lx), (a*(k*np.cos(th) + 1)*np.exp(lw) - 2*b*(np.exp(lw) + np.exp(lx) - 1)*np.exp(lw + lx))*np.exp(-lx), -a*k*np.exp(lw - lx)*np.sin(th)]
        row2 = [-np.exp(lx), -np.exp(lw), a*k*np.sin(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class InertialFitzHughNagumo(DynSys):
    @staticjit
    def _rhs(x, v, y, t, a, eps, k):
        xdot = v
        vdot = (y + x - x**3 / 3 - v) / k
        ydot = -eps * (x - a)
        return xdot, vdot, ydot

    @staticjit
    def _jac(x, v, y, t, a, eps, k):
        row1 = [0, 1, 0]
        row2 = [(1 - x**2)/k, -1/k, k**(-1.0)]
        row3 = [-eps, 0, 0]
        return row1, row2, row3


class IntraguildPredation(DynSys):
    @staticjit
    def _rhs(rr, c1, c2, t, a1r, a2c, a2r, b1r, b2c, b2r, k0, m1, m2, r, w1r, w2c, w2r):
        f1 = a1r * rr / (1 + b1r * rr)
        f2 = a2r * rr / (1 + b2r * rr)
        g = a2c * c1 / (1 + b2c * c1)
        rrdot = r * rr * (1 - rr / k0) - f1 * c1 - f2 * c2
        c1dot = w1r * f1 * c1 - g * c2 - m1 * c1
        c2dot = w2r * f2 * c2 + w2c * g * c2 - m2 * c2
        return rrdot, c1dot, c2dot

    @staticjit
    def _jac(rr, c1, c2, t, a1r, a2c, a2r, b1r, b2c, b2r, k0, m1, m2, r, w1r, w2c, w2r):
        row1 = [-a1r*c1/(b1r*rr + 1)**2 - a2r*c2/(b2r*rr + 1)**2 + r - 2*r*rr/k0, -a1r*rr/(b1r*rr + 1), -a2r*rr/(b2r*rr + 1)]
        row2 = [a1r*c1*w1r/(b1r*rr + 1)**2, a1r*rr*w1r/(b1r*rr + 1) + a2c*b2c*c1*c2/(b2c*c1 + 1)**2 - a2c*c2/(b2c*c1 + 1) - m1, -a2c*c1/(b2c*c1 + 1)]
        row3 = [a2r*c2*w2r/(b2r*rr + 1)**2, a2c*c2*w2c/(b2c*c1 + 1)**2, a2c*c1*w2c/(b2c*c1 + 1) + a2r*rr*w2r/(b2r*rr + 1) - m2]
        return row1, row2, row3


class IteratedPrisonersDilemma(DynSys):
    @staticjit
    def _rhs(
        x2, x3, x4, t,
        a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, mu,
    ):
        x1 = 1 - x2 - x3 - x4
        p1 = a11 * x1 + a12 * x2 + a13 * x3 + a14 * x4
        p2 = a21 * x1 + a22 * x2 + a23 * x3 + a24 * x4
        p3 = a31 * x1 + a32 * x2 + a33 * x3 + a34 * x4
        p4 = a41 * x1 + a42 * x2 + a43 * x3 + a44 * x4
        pbar = x1 * p1 + x2 * p2 + x3 * p3 + x4 * p4
        x2dot = (1 - 4 * mu / 3) * p2 * x2 - pbar * x2 + mu * pbar / 3
        x3dot = (1 - 4 * mu / 3) * p3 * x3 - pbar * x3 + mu * pbar / 3
        x4dot = (1 - 4 * mu / 3) * p4 * x4 - pbar * x4 + mu * pbar / 3
        return x2dot, x3dot, x4dot

    @staticjit
    def _jac(x2, x3, x4, t, a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, mu,):
        c0 = -a21 + a22
        c1 = x3*(-a31 + a32)
        c2 = x4*(-a41 + a42)
        c3 = -x2 - x3 - x4 + 1
        c4 = a21*c3 + a22*x2 + a23*x3 + a24*x4
        c5 = a11*c3
        c6 = a12*x2
        c7 = a13*x3
        c8 = a14*x4
        c9 = -c5 - c6 - c7 - c8
        c10 = c0*x2 + c1 + c2 + c3*(-a11 + a12) + c4 + c9
        c11 = 1 - 4/3*mu
        c12 = a31*c3 + a32*x2 + a33*x3 + a34*x4
        c13 = a41*c3 + a42*x2 + a43*x3 + a44*x4
        c14 = c12*x3 + c13*x4 + c3*(c5 + c6 + c7 + c8) + c4*x2
        c15 = x2*(-a21 + a23)
        c16 = -a31 + a33
        c17 = x4*(-a41 + a43)
        c18 = c12 + c15 + c16*x3 + c17 + c3*(-a11 + a13) + c9
        c19 = (1/3)*mu
        c20 = c18*c19
        c21 = x2*(-a21 + a24)
        c22 = x3*(-a31 + a34)
        c23 = -a41 + a44
        c24 = c13 + c21 + c22 + c23*x4 + c3*(-a11 + a14) + c9
        c25 = c19*c24
        c26 = c10*c19
        row1 = [c0*c11*x2 + (1/3)*c10*mu - c10*x2 + c11*c4 - c14, c11*c15 - c18*x2 + c20, c11*c21 - c24*x2 + c25]
        row2 = [c1*c11 - c10*x3 + c26, c11*c12 + c11*c16*x3 - c14 + (1/3)*c18*mu - c18*x3, c11*c22 - c24*x3 + c25]
        row3 = [-c10*x4 + c11*c2 + c26, c11*c17 - c18*x4 + c20, c11*c13 + c11*c23*x4 - c14 + (1/3)*c24*mu - c24*x4]
        return row1, row2, row3


class JosephsonRCL(DynSys):
    @staticjit
    def _rhs(ph, v, i, t, bc, bl, gn, gsg, i0, vg, w):
        # quasiparticle conductance steps from gsg to gn at |v| = vg, smoothed over width w
        g = gsg + (gn - gsg) * 0.5 * (1 + np.tanh((np.abs(v) - vg) / w))
        phdot = v
        vdot = (i0 - g * v - np.sin(ph) - i) / bc
        idot = (v - i) / bl
        return phdot, vdot, idot

    @staticjit
    def _postprocessing(ph, v, i):
        return np.sin(ph), v, i

    @staticjit
    def _jac(ph, v, i, t, bc, bl, gn, gsg, i0, vg, w):
        row1 = [0, 1, 0]
        row2 = [-np.cos(ph)/bc, (-gsg - v*(1 - np.tanh((-vg + abs(v))/w)**2)*(0.5*gn - 0.5*gsg)*np.sign(v)/w - (0.5*gn - 0.5*gsg)*(np.tanh((-vg + abs(v))/w) + 1))/bc, -1/bc]
        row3 = [0, bl**(-1.0), -1/bl]
        return row1, row2, row3


class KeynesianGrowthCycle(DynSys):
    @staticjit
    def _rhs(g, gw, s, t, alpha, beta, cstar, delta, eps, gamma, phi, pibar):
        gdot = alpha * ((cstar + phi * (g - gw)) * gw - s)
        gwdot = gamma * (g - gw)
        sdot = eps * (beta * (pibar - s) * s - gw) + delta * gamma * (g - gw)
        return gdot, gwdot, sdot

    @staticjit
    def _jac(g, gw, s, t, alpha, beta, cstar, delta, eps, gamma, phi, pibar):
        row1 = [alpha*gw*phi, alpha*(cstar - gw*phi + phi*(g - gw)), -alpha]
        row2 = [gamma, -gamma, 0]
        row3 = [delta*gamma, -delta*gamma - eps, beta*eps*(pibar - 2*s)]
        return row1, row2, row3


class KlebanoffHastings(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a1, a12, a2, a21, b1, b2, k, m, p):
        fx = a1 * x * z / (1 + b1 * x)
        fy = a2 * y * z / (1 + b2 * y)
        xdot = x - x**2 - a12 * x * y - fx
        ydot = p * y - k * y**2 - a21 * x * y - fy
        zdot = fx + fy - m * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a1, a12, a2, a21, b1, b2, k, m, p):
        row1 = [-a1*z/(b1*x + 1)**2 - a12*y - 2*x + 1, -a12*x, -a1*x/(b1*x + 1)]
        row2 = [-a21*y, -a2*z/(b2*y + 1)**2 - a21*x - 2*k*y + p, -a2*y/(b2*y + 1)]
        row3 = [a1*z/(b1*x + 1)**2, a2*z/(b2*y + 1)**2, a1*x/(b1*x + 1) + a2*y/(b2*y + 1) - m]
        return row1, row2, row3


class KrischerElectrochemical(DynSys):
    @staticjit
    def _rhs(phi, c, th, t, bb, eps, mu, phi0, rho, u):
        k = 0.00002 * phi**3 - 0.0094 * phi**2 + 1.12 * phi
        th0 = 1.0 / (1.0 + np.exp((phi - phi0) / bb))
        jf = c * k * (1.0 - th)
        phidot = (-jf + (u - phi) / rho) / eps
        cdot = (-jf + 1.0 - c) / mu
        thdot = th0 - th
        return phidot, cdot, thdot

    @staticjit
    def _jac(phi, c, th, t, bb, eps, mu, phi0, rho, u):
        c0 = eps**(-1.0)
        c1 = th - 1.0
        c2 = phi**2
        c3 = c*c1*(6.000000000000001e-05*c2 - 0.0188*phi + 1.12)
        c4 = phi*(2e-05*c2 - 0.0094*phi + 1.12)
        c5 = c1*c4
        c6 = c*c4
        c7 = mu**(-1.0)
        c8 = bb**(-1.0)
        c9 = np.exp(c8*(phi - phi0))
        row1 = [c0*(c3*rho - 1)/rho, c0*c5, c0*c6]
        row2 = [c3*c7, c7*(c5 - 1), c6*c7]
        row3 = [-1.0*c8*c9/(c9 + 1.0)**2, 0, -1]
        return row1, row2, row3


class LarterBreakspear(DynSys):
    @staticjit
    def _rhs(
        v, w, z, t,
        aee, aei, aie, ane, ani, b, dca, dk, dna, dv, dz, gca, gk, gl, gna, i, phi, rnmda,
        tauk, tca, tk, tna, vca, vk, vl, vna, vt, zt,
    ):
        mca = 0.5 * (1 + np.tanh((v - tca) / dca))
        mna = 0.5 * (1 + np.tanh((v - tna) / dna))
        mk = 0.5 * (1 + np.tanh((v - tk) / dk))
        qv = 0.5 * (1 + np.tanh((v - vt) / dv))
        qz = 0.5 * (1 + np.tanh((z - zt) / dz))
        vdot = (
            -(gca + rnmda * aee * qv) * mca * (v - vca)
            - gk * w * (v - vk)
            - gl * (v - vl)
            - (gna * mna + aee * qv) * (v - vna)
            - aie * z * qz
            + ane * i
        )
        wdot = phi * (mk - w) / tauk
        zdot = b * (ani * i + aei * v * qv)
        return vdot, wdot, zdot

    @staticjit
    def _jac(v, w, z, t, aee, aei, aie, ane, ani, b, dca, dk, dna, dv, dz, gca, gk, gl, gna, i, phi, rnmda, tauk, tca, tk, tna, vca, vk, vl, vna, vt, zt,):
        c0 = dv**(-1.0)
        c1 = c0*(v - vt)
        c2 = np.tanh(c1)
        c3 = aee*(0.5*c2 + 0.5)
        c4 = dna**(-1.0)
        c5 = -v
        c6 = np.tanh(c4*(-c5 - tna))
        c7 = 0.5*c0
        c8 = aee*c7*(1 - c2**2)
        c9 = dca**(-1.0)
        c10 = np.tanh(c9*(-c5 - tca))
        c11 = 0.5*c10 + 0.5
        c12 = -c3*rnmda - gca
        c13 = v - vca
        c14 = dz**(-1.0)
        c15 = c14*(z - zt)
        c16 = dk**(-1.0)
        c17 = phi/tauk
        row1 = [c11*c12 - c11*c13*c8*rnmda + 0.5*c12*c13*c9*(1 - c10**2) - c3 - gk*w - gl - gna*(0.5*c6 + 0.5) - (v - vna)*(0.5*c4*gna*(1 - c6**2) + c8), gk*(-v + vk), -0.5*aie*c14*(dz*(np.tanh(c15) + 1) + z/np.cosh(c15)**2)]
        row2 = [0.5*c16*c17/np.cosh(c16*(c5 + tk))**2, -c17, 0]
        row3 = [aei*b*c7*(dv*(c2 + 1) + v/np.cosh(c1)**2), 0, 0]
        return row1, row2, row3


class Malasoma(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = z
        zdot = -a * z + x * y**2 - x
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [y**2 - 1, 2*x*y, -a]
        return row1, row2, row3


class MankinHudson(DynSys):
    @staticjit
    def _rhs(c, temp, th, t, a, b, beta, da, omega, tc):
        rate = da * (1 - c) * np.exp(temp)
        cdot = -c + rate
        tempdot = -temp + b * rate - beta * (temp - tc - a * np.sin(th))
        thdot = omega
        return cdot, tempdot, thdot

    @staticjit
    def _postprocessing(c, temp, th):
        return c, temp, np.sin(th)

    @staticjit
    def _jac(c, temp, th, t, a, b, beta, da, omega, tc):
        row1 = [-da*np.exp(temp) - 1, da*(1 - c)*np.exp(temp), 0]
        row2 = [-b*da*np.exp(temp), -b*da*(c - 1)*np.exp(temp) - beta - 1, a*beta*np.cos(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class MarhlCalcium(DynSys):
    @staticjit
    def _rhs(
        cac, caer, cam, t,
        beta_er, beta_m, catot, k1, k2, kch, kin, kleak, kminus, kmm, kout, kplus, kpump,
        prtot, rho_er, rho_m,
    ):
        jch = kch * cac**2 / (k1**2 + cac**2) * (caer - cac)
        jleak = kleak * (caer - cac)
        jpump = kpump * cac
        jin = kin * cac**8 / (k2**8 + cac**8)
        jout = (kout * cac**2 / (k1**2 + cac**2) + kmm) * cam
        capr = catot - cac - (rho_er / beta_er) * caer - (rho_m / beta_m) * cam
        pr = prtot - capr
        cacdot = jch + jleak - jpump + jout - jin + kminus * capr - kplus * cac * pr
        caerdot = (beta_er / rho_er) * (jpump - jch - jleak)
        camdot = (beta_m / rho_m) * (jin - jout)
        return cacdot, caerdot, camdot

    @staticjit
    def _jac(cac, caer, cam, t, beta_er, beta_m, catot, k1, k2, kch, kin, kleak, kminus, kmm, kout, kplus, kpump, prtot, rho_er, rho_m,):
        c0 = cac**2
        c1 = k1**2
        c2 = c0 + c1
        c3 = c2**(-1.0)
        c4 = c0*c3
        c5 = c4*kch
        c6 = cac*kplus
        c7 = rho_er/beta_er
        c8 = rho_m/beta_m
        c9 = 2*kout
        c10 = cac**3
        c11 = c2**2
        c12 = c11**(-1.0)
        c13 = c10*c12
        c14 = k2**8
        c15 = c14 + cac**8
        c16 = c15**(-2.0)
        c17 = 8*cac**7*kin
        c18 = cac - caer
        c19 = -c18
        c20 = 2*kch
        c21 = kleak + kpump
        c22 = c5 + kleak
        c23 = c4*kout + kmm
        c24 = beta_er/rho_er
        c25 = beta_m/rho_m
        row1 = [-c13*c19*c20 + 8*c16*cac**15*kin + 2*c19*c3*cac*kch - c21 - c5 - c6 + cam*(-c13*c9 + 2*c3*cac*kout) - kminus - kplus*(c7*caer + c8*cam + cac - catot + prtot) - c17/c15, c22 - c6*c7 - c7*kminus, c23 - c6*c8 - c8*kminus]
        row2 = [c12*c24*(-c10*c18*c20 + c11*c21 + c2*cac*kch*(3*cac - 2*caer)), -c22*c24, 0]
        row3 = [-c1*c12*c25*c9*cac*cam + c14*c16*c17*c25, 0, -c23*c25]
        return row1, row2, row3


class NMRFeedbackBloch(DynSys):
    @staticjit
    def _rhs(x, y, z, t, c, d, g, g1, g2):
        xdot = d * y + g * z * (x * np.sin(c) - y * np.cos(c)) - x / g2
        ydot = -d * x - z + g * z * (x * np.cos(c) + y * np.sin(c)) - y / g2
        zdot = y - g * np.sin(c) * (x**2 + y**2) - (z - 1) / g1
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, c, d, g, g1, g2):
        row1 = [g*z*np.sin(c) - 1/g2, d - g*z*np.cos(c), g*(x*np.sin(c) - y*np.cos(c))]
        row2 = [-d + g*z*np.cos(c), g*z*np.sin(c) - 1/g2, g*(x*np.cos(c) + y*np.sin(c)) - 1]
        row3 = [-2*g*x*np.sin(c), -2*g*y*np.sin(c) + 1, -1/g1]
        return row1, row2, row3


class NaPKKMBurster(DynSys):
    @staticjit
    def _rhs(v, n, w, t, am, an, aw, bm, bn, bw, ek, el, ena, gk, gkm, gl, gna, i, taun, tauw):
        minf = 1 / (1 + np.exp((am - v) / bm))
        ninf = 1 / (1 + np.exp((an - v) / bn))
        winf = 1 / (1 + np.exp((aw - v) / bw))
        vdot = (
            -gna * minf * (v - ena)
            - gk * n * (v - ek)
            - gkm * w * (v - ek)
            - gl * (v - el)
            + i
        )
        ndot = (ninf - n) / taun
        wdot = (winf - w) / tauw
        return vdot, ndot, wdot

    @staticjit
    def _jac(v, n, w, t, am, an, aw, bm, bn, bw, ek, el, ena, gk, gkm, gl, gna, i, taun, tauw):
        row1 = [-gk*n - gkm*w - gl - gna/(np.exp((am - v)/bm) + 1) - gna*(-ena + v)*np.exp((am - v)/bm)/(bm*(np.exp((am - v)/bm) + 1)**2), gk*(ek - v), gkm*(ek - v)]
        row2 = [(1/4)/(bn*taun*np.cosh((1/2)*(an - v)/bn)**2), -1/taun, 0]
        row3 = [(1/4)/(bw*tauw*np.cosh((1/2)*(aw - v)/bw)**2), 0, -1/tauw]
        return row1, row2, row3


class ParallelAutocatalator(DynSys):
    @staticjit
    def _rhs(x, y, z, t, ax, ay, dax, day, daz):
        rx = dax * x * z**2
        ry = day * y * z**2
        xdot = 1 - x - rx
        ydot = 1 - y - ry
        zdot = 1 - (1 + daz) * z + ax * rx + ay * ry
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, ax, ay, dax, day, daz):
        row1 = [-dax*z**2 - 1, 0, -2*dax*x*z]
        row2 = [0, -day*z**2 - 1, -2*day*y*z]
        row3 = [ax*dax*z**2, ay*day*z**2, 2*ax*dax*x*z + 2*ay*day*y*z - daz - 1]
        return row1, row2, row3


class ParallelReactionCSTR(DynSys):
    @staticjit
    def _rhs(y, z, temp, t, a, b1, b2, da1, da2, g1, g2):
        r1 = da1 * y * np.exp(g1 * temp / (1 + temp))
        r2 = da2 * z * np.exp(g2 * temp / (1 + temp))
        ydot = 1 - y - r1
        zdot = 1 - z - r2
        tempdot = a * (b1 * r1 + b2 * r2 - temp)
        return ydot, zdot, tempdot

    @staticjit
    def _jac(y, z, temp, t, a, b1, b2, da1, da2, g1, g2):
        row1 = [-da1*np.exp(g1*temp/(temp + 1)) - 1, 0, -da1*g1*y*np.exp(g1*temp/(temp + 1))/(temp + 1)**2]
        row2 = [0, -da2*np.exp(g2*temp/(temp + 1)) - 1, -da2*g2*z*np.exp(g2*temp/(temp + 1))/(temp + 1)**2]
        row3 = [a*b1*da1*np.exp(g1*temp/(temp + 1)), a*b2*da2*np.exp(g2*temp/(temp + 1)), a*(b1*da1*g1*y*np.exp(g1*temp/(temp + 1)) + b2*da2*g2*z*np.exp(g2*temp/(temp + 1)) - (temp + 1)**2)/(temp + 1)**2]
        return row1, row2, row3


class PendularCavity(DynSys):
    @staticjit
    def _rhs(ph, dph, th, t, alpha, beta, d0, eb, q):
        lor = 1.0 / (1 + (d0 + ph + th) ** 2)
        phdot = dph
        dphdot = -dph / q - ph + alpha * lor
        thdot = -eb * (th + beta * lor)
        return phdot, dphdot, thdot

    @staticjit
    def _jac(ph, dph, th, t, alpha, beta, d0, eb, q):
        row1 = [0, 1, 0]
        row2 = [1.0*alpha*(-2*d0 - 2*ph - 2*th)/((d0 + ph + th)**2 + 1)**2 - 1, -1/q, -2.0*alpha*(d0 + ph + th)/((d0 + ph + th)**2 + 1)**2]
        row3 = [2.0*beta*eb*(d0 + ph + th)/((d0 + ph + th)**2 + 1)**2, 0, -eb*(1.0*beta*(-2*d0 - 2*ph - 2*th)/((d0 + ph + th)**2 + 1)**2 + 1)]
        return row1, row2, row3


class PikovskyRabinovich(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, delta, gamma, mu):
        xdot = y - delta * z
        ydot = -x + 2 * gamma * y + alpha * z + beta
        zdot = (x - z**3 + z) / mu
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, beta, delta, gamma, mu):
        row1 = [0, 1, -delta]
        row2 = [-1, 2*gamma, alpha]
        row3 = [mu**(-1.0), 0, (1 - 3*z**2)/mu]
        return row1, row2, row3


class QIFAdaptation(DynSys):
    @staticjit
    def _rhs(r, v, a, t, beta, delta, eta, j, taua):
        rdot = delta / (np.pi * (1 + beta)) + 2 * r * v
        vdot = v**2 - (np.pi * r) ** 2 + eta + j * r - a
        adot = (-(1 + beta) * a + beta * (eta + j * r)) / taua
        return rdot, vdot, adot

    @staticjit
    def _jac(r, v, a, t, beta, delta, eta, j, taua):
        row1 = [2*v, 2*r, 0]
        row2 = [j - 2*np.pi**2*r, 2*v, -1]
        row3 = [beta*j/taua, 0, (-beta - 1)/taua]
        return row1, row2, row3


class RayleighDuffing(DynSys):
    @staticjit
    def _rhs(x, y, th, t, a, gamma, mu, omega):
        xdot = y
        ydot = mu * (1 - gamma * y**2) * y - x**3 + a * np.sin(th)
        thdot = omega
        return xdot, ydot, thdot

    @staticjit
    def _postprocessing(x, y, th):
        return x, y, np.sin(th)

    @staticjit
    def _jac(x, y, th, t, a, gamma, mu, omega):
        row1 = [0, 1, 0]
        row2 = [-3*x**2, mu*(-3*gamma*y**2 + 1), a*np.cos(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class RfSquid(DynSys):
    @staticjit
    def _rhs(x, v, th, t, beta, gamma, omega, phiac, phidc):
        xdot = v
        vdot = (
            -gamma * v - x - beta * np.sin(2 * np.pi * x) + phidc + phiac * np.cos(th)
        )
        thdot = omega
        return xdot, vdot, thdot

    @staticjit
    def _postprocessing(x, v, th):
        return x, v, np.sin(th)

    @staticjit
    def _jac(x, v, th, t, beta, gamma, omega, phiac, phidc):
        row1 = [0, 1, 0]
        row2 = [-2*np.pi*beta*np.cos(2*np.pi*x) - 1, -gamma, -phiac*np.sin(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class SchefferDaphnia(DynSys):
    @staticjit
    def _rhs(a, z, th, t, d, e, eps, f, g, ha, hz, k, m, omega, r):
        s = (1 - eps * np.cos(th)) / (1 + eps)
        gr = g * a / (a + ha)
        adot = s * r * a * (1 - a / (s * k)) - z * s * gr + d * (s * k - a)
        zdot = e * s * gr * z - s * m * z - s**2 * f * z**2 / (z**2 + hz**2)
        thdot = omega
        return adot, zdot, thdot

    @staticjit
    def _postprocessing(a, z, th):
        return a, z, np.sin(th)

    @staticjit
    def _jac(a, z, th, t, d, e, eps, f, g, ha, hz, k, m, omega, r):
        c0 = k**(-1.0)
        c1 = a*r
        c2 = eps + 1
        c3 = c2**(-1.0)
        c4 = eps*np.cos(th) - 1
        c5 = -c4
        c6 = g*z
        c7 = a + ha
        c8 = c7**(-1.0)
        c9 = c3*c5
        c10 = c8*c9
        c11 = c7**(-2.0)
        c12 = a*g
        c13 = c3*c4
        c14 = d*k
        c15 = np.sin(th)
        c16 = c15*eps
        c17 = z**2
        c18 = c17 + hz**2
        c19 = c18**(-1.0)
        c20 = c2**(-2.0)
        c21 = 2*c20*c5**2*f
        row1 = [a*c11*c3*c5*g*z - c0*c1 - c10*c6 + c3*c5*r*(-a*c0*c2/c5 + 1) - d, c12*c13*c8, c16*(a**2*r + a*c14 - a*c6 + c1*ha + c14*ha)/(a*eps + c7 + eps*ha)]
        row2 = [-c11*c13*c6*e*ha, c10*c12*e - c19*c21*z - c9*m + c21*z**3/c18**2, a*c15*c3*c8*e*eps*g*z - 2*c16*c17*c19*c20*c5*f - c16*c3*m*z]
        row3 = [0, 0, 0]
        return row1, row2, row3


class SeasonalSIR(DynSys):
    @staticjit
    def _rhs(s, li, th, t, b0, b1, gamma, mu, omega):
        # susceptible fraction, log of the infected fraction, seasonal phase
        beta = b0 * (1 + b1 * np.sin(th))
        sdot = mu - beta * s * np.exp(li) - mu * s
        lidot = beta * s - (gamma + mu)
        thdot = omega
        return sdot, lidot, thdot

    @staticjit
    def _postprocessing(s, li, th):
        return s, li, np.sin(th)

    @staticjit
    def _jac(s, li, th, t, b0, b1, gamma, mu, omega):
        row1 = [-b0*b1*np.exp(li)*np.sin(th) - b0*np.exp(li) - mu, -b0*s*(b1*np.sin(th) + 1)*np.exp(li), -b0*b1*s*np.exp(li)*np.cos(th)]
        row2 = [b0*(b1*np.sin(th) + 1), 0, b0*b1*s*np.cos(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class ShenLarter(DynSys):
    @staticjit
    def _rhs(
        c, e, p, t,
        k1, k2, k3, k4, k5, kch, kin1, kin2, kleak, kminus, kout, kplus, kpump, r,
    ):
        jch = kch * p**4 / (p**4 + k1**4) * (k4 * c / ((c + k4) * (c + k5))) ** 3 * e
        jleak = kleak * e
        jpump = kpump * c**2 / (c**2 + k2**2)
        jin = kin1 * r + kin2
        jout = kout * c
        cdot = jch + jleak - jpump + jin - jout
        edot = jpump - jch - jleak
        pdot = kplus * r * c / (c + k3) - kminus * p
        return cdot, edot, pdot

    @staticjit
    def _jac(c, e, p, t, k1, k2, k3, k4, k5, kch, kin1, kin2, kleak, kminus, kout, kplus, kpump, r,):
        c0 = c**3
        c1 = c**2
        c2 = c1 + k2**2
        c3 = 2*kpump
        c4 = k4**3
        c5 = c + k4
        c6 = c5**(-3.0)
        c7 = c + k5
        c8 = c7**(-3.0)
        c9 = p**4
        c10 = k1**4
        c11 = c10 + c9
        c12 = c9/c11
        c13 = 3*c0*c12*c4*e*kch
        c14 = c*c3/c2 - c0*c3/c2**2 - 3*c1*c12*c4*c6*c8*e*kch + c13*c6/c7**4 + c13*c8/c5**4
        c15 = c0*c4*c6*c8*kch
        c16 = c12*c15 + kleak
        c17 = 4*c10*c15*e*p**3/c11**2
        row1 = [-c14 - kout, c16, c17]
        row2 = [c14, -c16, -c17]
        row3 = [k3*kplus*r/(c + k3)**2, 0, -kminus]
        return row1, row2, row3


class SpinValveOscillator(DynSys):
    @staticjit
    def _rhs(th, ph, ps, t, aac, aj, alpha, h, omega):
        st = np.sin(th)
        ct = np.cos(th)
        sp = np.sin(ph)
        cp = np.cos(ph)
        mx = st * cp
        my = st * sp
        mz = ct
        a = aj + aac * np.cos(ps)
        t1 = my * mz - a * (my**2 + mz**2)
        t2 = -mz * (h + mx) + a * mx * my
        t3 = my * h + a * mx * mz
        m1 = (t1 + alpha * (my * t3 - mz * t2)) / (1 + alpha**2)
        m2 = (t2 + alpha * (mz * t1 - mx * t3)) / (1 + alpha**2)
        m3 = (t3 + alpha * (mx * t2 - my * t1)) / (1 + alpha**2)
        thdot = m1 * ct * cp + m2 * ct * sp - m3 * st
        phdot = (-m1 * sp + m2 * cp) / st
        psdot = omega
        return thdot, phdot, psdot

    @staticjit
    def _postprocessing(th, ph, ps):
        return th, np.sin(ph), np.sin(ps)

    @staticjit
    def _jac(th, ph, ps, t, aac, aj, alpha, h, omega):
        c0 = np.cos(ph)
        c1 = np.sin(th)
        c2 = c0*c1
        c3 = c2 + h
        c4 = np.cos(th)
        c5 = c3*c4
        c6 = c1**2
        c7 = aac*np.cos(ps) + aj
        c8 = np.sin(ph)
        c9 = c0*c6*c7*c8 - c5
        c10 = c4**2
        c11 = c8**2
        c12 = c11*c6
        c13 = c10 + c12
        c14 = c13*c7
        c15 = c1*c4*c8 - c14
        c16 = c1*c15
        c17 = c1*c8
        c18 = c2*c4
        c19 = c17*h + c18*c7
        c20 = (alpha**2 + 1)**(-1.0)
        c21 = c20*c4
        c22 = c4*c9
        c23 = c0*c10
        c24 = 2*c4
        c25 = c1*c3 + c2*c24*c7*c8 - c23
        c26 = c15*c4
        c27 = c6*c8
        c28 = c1*c24
        c29 = c7*(c11*c28 - c28)
        c30 = c10*c8 - c27 - c29
        c31 = c4*c8
        c32 = c6*c7
        c33 = -c0*c32 + c23*c7 + c31*h
        c34 = c1*c20
        c35 = c19*c2
        c36 = c0*c27
        c37 = c36*c7
        c38 = alpha*(c26 - c35) + c37 - c5
        c39 = c17*c4
        c40 = alpha*(c1*c19*c8 - c22) - c14 + c39
        c41 = c1*c9
        c42 = alpha*(c17*c33 + c19*c31 - c25*c4 + c41) + c10*c8 - c27 - c29
        c43 = c0*c4
        c44 = alpha*(-c16 - c19*c43 - c2*c33 + c30*c4) + c25
        c45 = 2*c37
        c46 = c0*c1*c4 - c45
        c47 = c0**2
        c48 = -c12*c7 + c32*c47 + c39
        c49 = c2*h - c39*c7
        c50 = c0*c21
        c51 = alpha*(c17*c49 + c35 - c4*c48) + c18 - c45
        c52 = alpha*(c17*c19 - c2*c49 + c4*c46) + c48
        c53 = c21*c8
        c54 = aac*np.sin(ps)
        c55 = c47*c54
        c56 = c13*c54
        c57 = c1**(-1.0)
        c58 = c20*c8
        row1 = [c0*c20*c4*c42 - c17*c20*c38 - c2*c20*c40 + c20*c4*c44*c8 - c21*(alpha*(-c16*c8 + c2*c9) + c19) - c34*(alpha*(c0*c22 - c17*c30 + c2*c25 - c26*c8) + c33), -c34*(alpha*(c0*c1*c48 - c15*c2 - c17*c46 - c41*c8) + c49) + c38*c50 - c40*c53 + c50*c51 + c52*c53, -c34*(alpha*(-c1**3*c55*c8 - c17*c56) - c18*c54) + c50*c56 + c53*(alpha*(c4*c55*c6 + c4*c56) - c36*c54)]
        row2 = [-c4*(c0*c20*c38 - c40*c58)/c6 + c57*(c0*c20*c44 - c42*c58), c57*(-c0*c20*c40 + c0*c20*c52 - c38*c58 - c51*c58), c20*c54*c57*(alpha*c43 - c8)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class StokesDrop(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, theta, wmag):
        e11 = 1.0 / (1.0 + alpha)
        e22 = alpha / (1.0 + alpha)
        e33 = -1.0
        wx = wmag * np.sin(theta)
        wz = wmag * np.cos(theta)
        r2 = x**2 + y**2 + z**2
        xex = e11 * x**2 + e22 * y**2 + e33 * z**2
        c = 5.0 * r2 - 3.0
        xdot = 0.5 * (c * e11 * x - 2 * x * xex) - 0.5 * wz * y
        ydot = 0.5 * (c * e22 * y - 2 * y * xex) + 0.5 * (wz * x - wx * z)
        zdot = 0.5 * (c * e33 * z - 2 * z * xex) + 0.5 * wx * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, theta, wmag):
        c0 = z**2
        c1 = 3.5*c0
        c2 = x**2
        c3 = 1.0*c0
        c4 = y**2
        c5 = alpha*c4
        c6 = 1.0*c5
        c7 = 2.5*c4 - 1.5
        c8 = 1.0/(alpha + 1)
        c9 = alpha + 1.0
        c10 = c9**(-1.0)
        c11 = 0.5*wmag
        c12 = c11*c9*np.cos(theta)
        c13 = 2.0*alpha
        c14 = x*y
        c15 = c10*x*z
        c16 = 5.0*alpha
        c17 = 1.0*c2
        c18 = 2.5*c2
        c19 = y*z
        c20 = -c11*np.sin(theta)
        c21 = c10*c19
        row1 = [c8*(alpha*c3 + c1 + 4.5*c2 - c6 + c7), c10*(-c12 - c13*c14 + 5.0*x*y), c15*(c13 + 7.0)]
        row2 = [c10*(c12 + c14*c16 - 2.0*c14), c8*(alpha*c1 + alpha*c18 - 1.5*alpha - c17 + c3 + 4.5*c5), c16*c21 + 2.0*c19 + c20]
        row3 = [c15*(-c16 - 7.0), -c13*c21 - 5.0*c19 - c20, c10*(-c17 - c6 + c9*(-4.5*c0 - c18 - c7))]
        return row1, row2, row3


class StretchTwistFold(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta):
        xdot = alpha * z - 8 * x * y
        ydot = 11 * x**2 + 3 * y**2 + z**2 + beta * x * z - 3
        zdot = -alpha * x + 2 * y * z - beta * x * y
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, alpha, beta):
        row1 = [-8*y, -8*x, alpha]
        row2 = [beta*z + 22*x, 6*y, beta*x + 2*z]
        row3 = [-alpha - beta*y, -beta*x + 2*z, 2*y]
        return row1, row2, row3


class SugunaSinha(DynSys):
    @staticjit
    def _rhs(x, y, z, t, k, l, n, q, vmax):
        g = vmax * y * (1 + y) * (1 + z) ** 2 / (l + (1 + y) ** 2 * (1 + z) ** 2)
        xdot = 1 / (1 + z**n) - k * x
        ydot = x - g
        zdot = g - q * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, k, l, n, q, vmax):
        c0 = 2*y
        c1 = y + 1
        c2 = z + 1
        c3 = c2**2
        c4 = c1**2*c3
        c5 = c4 + l
        c6 = c0*c4 - c5*(c0 + 1)
        c7 = vmax/c5**2
        c8 = c3*c7
        c9 = 2*z + 2
        row1 = [-k, 0, -n*z**(n - 1)/(z**n + 1)**2]
        row2 = [1, c6*c8, -c0*c1*c2*c7*l]
        row3 = [0, -c6*c8, -c1**3*c8*c9*y + c1*c9*vmax*y/c5 - q]
        return row1, row2, row3


class SymmetricGyro(DynSys):
    @staticjit
    def _rhs(x, v, th, t, a2, b, c1, c2, f, omega):
        # (1 - cos x)^2 / sin^3 x, written in a form that is regular at x = 0
        gyro = 0.5 * np.tan(x / 2) / np.cos(x / 2) ** 2
        xdot = v
        vdot = -a2 * gyro - c1 * v - c2 * v**3 + (b + f * np.sin(th)) * np.sin(x)
        thdot = omega
        return xdot, vdot, thdot

    @staticjit
    def _postprocessing(x, v, th):
        return x, v, np.sin(th)

    @staticjit
    def _jac(x, v, th, t, a2, b, c1, c2, f, omega):
        row1 = [0, 1, 0]
        row2 = [-0.5*a2*np.sin((1/2)*x)*np.tan((1/2)*x)/np.cos((1/2)*x)**3 - 0.25*a2/np.cos((1/2)*x)**4 + b*np.cos(x) + f*np.sin(th)*np.cos(x), -c1 - 3*c2*v**2, f*np.sin(x)*np.cos(th)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class ThreeWaveDecay(DynSys):
    @staticjit
    def _rhs(x, y, s, t, delta, nu1, nu2):
        # x + i y = a1 exp(i Phi); s = log of the daughter-wave energy
        xdot = nu1 * x + delta * y + np.exp(s) - 2 * y**2
        ydot = nu1 * y - delta * x + 2 * x * y
        sdot = 2 * (nu2 - x)
        return xdot, ydot, sdot

    @staticjit
    def _jac(x, y, s, t, delta, nu1, nu2):
        row1 = [nu1, delta - 4*y, np.exp(s)]
        row2 = [-delta + 2*y, nu1 + 2*x, 0]
        row3 = [-2, 0, 0]
        return row1, row2, row3


class TwoWaveParticle(DynSys):
    @staticjit
    def _rhs(x, v, tau, t, b, eps, k):
        xdot = v
        vdot = -eps * (np.sin(x) + b * k * np.sin(k * (x - tau)))
        taudot = 1.0
        return xdot, vdot, taudot

    @staticjit
    def _postprocessing(x, v, tau):
        return np.sin(x), v, np.sin(tau)

    @staticjit
    def _jac(x, v, tau, t, b, eps, k):
        row1 = [0, 1, 0]
        row2 = [-eps*(b*k**2*np.cos(k*(tau - x)) + np.cos(x)), 0, b*eps*k**2*np.cos(k*(tau - x))]
        row3 = [0, 0, 0]
        return row1, row2, row3


class WattGovernor(DynSys):
    @staticjit
    def _rhs(x, y, z, t, c, f, k, r):
        xdot = y
        ydot = r * z**2 * np.sin(x) * np.cos(x) - np.sin(x) - c * y
        zdot = k * np.cos(x) - f
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, c, f, k, r):
        row1 = [0, 1, 0]
        row2 = [-2*r*z**2*np.sin(x)**2 + r*z**2 - np.cos(x), -c, r*z*np.sin(2*x)]
        row3 = [-k*np.sin(x), 0, 0]
        return row1, row2, row3


class WienBridge(DynSys):
    @staticjit
    def _rhs(x, y, z, t, al, be, c, k):
        g = -x + (k - 1) * y - z
        nz = al * z + (be - al) * np.maximum(z - 1, 0.0)
        xdot = g
        ydot = g - y
        zdot = c * (g - nz)
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, al, be, c, k):
        dnz = al + (be - al) * 0.5 * (1 + np.sign(z - 1))
        row1 = [-1, k - 1, -1]
        row2 = [-1, k - 2, -1]
        row3 = [-c, c * (k - 1), -c * (1 + dnz)]
        return row1, row2, row3


## Deprecated names, resolved lazily so they never appear in dir(dysts.flows) and
## are not counted twice by dysts.systems.get_attractor_list
_DEPRECATED_ALIASES = {"Aizawa": "Langford"}


def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"dysts.flows.{name} is deprecated; use dysts.flows.{new_name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    if name in _REMOVED_DELAY_SYSTEMS:
        raise _removed_delay_error(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
