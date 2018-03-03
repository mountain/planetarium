import numpy as np
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt


def ploter(opath, r=None, u=None, v=None):
    r = r * (r > 0.01)
    r = r / np.max(r)
    levels1 = np.linspace(0.0, 1.0, num=32)
    if len(r.shape) == 2:
        h, w = r.shape
    if len(r.shape) == 3:
        _, h, w = r.shape
    if len(r.shape) == 4:
        _, _, h, w = r.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    try:
        fig, ax = plt.subplots(sharex=True, sharey=True)

        if r is not None:
            ax.contourf(X, Y, r, levels=levels1)
        if u is not None and v is not None:
            s = np.sqrt(u * u + v * v + 0.00001)
            sx = np.max(s)
            s = s / sx
            u = u / sx
            v = v / sx
            ax.streamplot(X, Y, u, v, linewidth=1, color=s)

        plt.savefig(opath)
    except Exception as e:
        print(e)
    finally:
        plt.close()


def plotseq(seq):
    try:
        fig, axes = plt.subplots(6, 5, figsize=(12, 6),
                                 subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(hspace=0.3, wspace=0.05)

        for ax, data in zip(axes.flat, seq):
            A, B = data
            r, g, b = [A[0, 0], B[0, 0], A[0, 0] * A[0, 0] + B[0, 0] * B[0, 0]]
            c = np.dstack([r, g, b])
            ax.imshow(c, interpolation='nearest')
        plt.show()
    except Exception as e:
        print(e)
    finally:
        plt.close()
