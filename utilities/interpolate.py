import numpy as np
import scipy.interpolate as si
import scipy.signal as ss
from scipy.spatial import ConvexHull


def make_xz_mesh(nx, nz, zscale=.995):
    x, z = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,nz))
    for i in range(nx):
        z[:,i] *= (1-x[0,i]**2)**.5
    z *= zscale
    return x, z


def sph2cart(r, theta, phi):
    return r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)    


def interpolate(field, r, theta, phi, xi, method='cubic'):
    """Interpolate the field in the x-z plane (φ=0) at locations xi = [(x1,z1),(x2,z2),...]"""
    # Convert from spherical to cartesian coordinates
    points_0 = np.array([sph2cart(rr, tt, 0.)    for tt in theta.ravel() for rr in r.ravel()]) 
    points_1 = np.array([sph2cart(rr, tt, np.pi) for tt in theta.ravel() for rr in r.ravel()]) 
    points = np.append(points_0, points_1, axis=0)

    # Remove y coordinate
    points = np.array([[point[0], point[2]] for point in points])

    # Get field values at φ=0
    nphi = np.shape(field['g'][0])[0]
    values = np.append(field['g'][0][0].ravel(), field['g'][0][nphi//2].ravel())

    return si.griddata(points, values, xi, method='cubic')


def envelope_peak(x, y, first=None, final=None, usehull=False):
    if first is None: first = 0
    if final is None: final = len(x)-1

    if usehull:
        xvalues, yvalues = x[first:final+1], y[first:final+1]
        # Dummy points to ensure the convex hull only contains peaks
        if first > 0 and final < len(x)-1:
            xvalues = np.append(np.append(x[first-1], xvalues), x[final+1])
            yvalues = np.append(np.append(min(y), yvalues), min(y))
        points = np.array(list(zip(xvalues, yvalues)))
        hull = ConvexHull(points)
        indices = first + sorted(hull.vertices)

        # Remove adjacent indices and the extra bounds
        indices = indices[:-1][np.diff(indices) != 1]
        reject = [first-1, final+1]
        indices = [index for index in indices if index not in reject]
        indices = [first] + indices + [final]
    else:
        indices, _ = ss.find_peaks(y[first:final+1])
        indices += first
        # Force the first peak to get interpolated on its hip, not at the extremum
        # TODO: make this happen for all peaks?
        cur, next = indices[0], indices[1]
        while cur > first:
            # Draw a line between cur and next, then check if the value at cur-1
            # exceeds the line
            m = (y[next]-y[cur])/(x[next]-x[cur])
            value = y[next] + m*(x[cur-1] - x[next])
            if value < y[cur-1]:
                cur -= 1
            else:
                indices[0] = cur-1
                break
        indices = np.append(np.append(first, indices), final)
    points, values = x[indices], y[indices]
    return si.griddata(points, values, x, method='cubic')


def envelope_analytic(signal):
    signal_mean = np.mean(signal)
    return abs(ss.hilbert(signal - signal_mean))+signal_mean


def envelope(x, y, kind, first=None, final=None, usehull=False):
    if kind == 'peak':
        return envelope_peak(x, y, first=first, final=final, usehull=usehull)
    elif kind == 'analytic':
        return envelope_analytic(y)
    else:
        raise ValueError('Unsupported envelope kind!')


def lsqsolver(a, b):
    result = np.linalg.lstsq(a, b, rcond=None)
    return result[0]


def polyfit_peak(x, y, npeaks):
    peakindices, _ = ss.find_peaks(y)
    if len(peakindices) < 3:
        peakindices = [peakindices[0]] * 3

    if len(peakindices) == 3:
        solver = np.linalg.solve
    else:
        solver = lsqsolver
    npeaks = min(npeaks, len(peakindices))
    peakindices = peakindices[np.argsort(y[peakindices])[-npeaks:][::-1]]
    xx, yy = np.reshape(x[peakindices], (npeaks, 1)), y[peakindices]

    A = np.concatenate([xx ** 2, xx ** 1, xx ** 0], axis=1)
    p = solver(A, yy)
    return p, np.polyval(p, x)


def polyfit_lsq(x, y, first, final):
    if first is None: first = 0
    if final is None: final = len(x)-1

    solver = lsqsolver
    xx, yy = x[first:final + 1], y[first:final + 1]
    xx = np.reshape(xx, (len(xx), 1))

    A = np.concatenate([xx ** 2, xx ** 1, xx ** 0], axis=1)
    p = solver(A, yy)
    return p, np.polyval(p, x)


def polyfit(x, y, kind, npeaks=3, first=None, final=None):
    if kind == 'peak':
        return polyfit_peak(x, y, npeaks)
    elif kind == 'lsq':
        return polyfit_lsq(x, y, first, final)
    else:
        raise ValueError('Unsupported polyfit kind!')
