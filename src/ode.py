# -*- coding: utf-8 -*-

"""
ODE solvers
"""

def euler(derivative):
    """
    Euler method
    """
    return lambda t, x, dt:  (t + dt, x + derivative(t, x) * dt)


def heun(derivative):
    """
    Heun method
    """
    def solve(t, x, dt):
        k1 = derivative(t, x) * dt
        k2 = derivative(t + dt, x + k1) * dt / 2
        return t + dt, x + (k1 + k2) * dt / 2
    return solve


def rk4(derivative):
    """
    RK4 method
    """
    def solve(t, x, dt):
        k1 = derivative(t, x) * dt
        k2 = derivative(t + dt / 2, x + k1 / 2) * dt / 2
        k3 = derivative(t + dt / 2, x + k2 / 2) * dt / 2
        k4 = derivative(t + dt, x + k3) * dt
        return t + dt, x + (k1 + k2 * 2 + k3 * 2 + k4) / 6
    return solve


def verlet(acceleration):
    """
    Verlet method
    """
    def solve(t, x, v, dt):
        xnew = x + v * dt + acceleration(t, x, v) * dt * dt / 2
        vnew = v + acceleration(t, x, v) * dt / 2 + acceleration(t, xnew, v) * dt / 2
        return t + dt, xnew, vnew
    return solve