import numpy as np
import pandas as pd


def ray_tracing(owt, theta_0, svp):
    """ray tracing algorithm

    Args:
        None

    Returns:
        None

    """
    t = np.float64(0.0)
    c0 = svp['vitesse'].iloc[0]
    z = 0
    x = 0
    z = 0
    r = 0
    theta0 = theta_0
    st0 = np.sin(theta0)
    ct0 = np.cos(theta0)
    dz = svp['profondeur'].iloc[1] - svp['profondeur'].iloc[0]
    c1 = svp['vitesse'].iloc[1]
    g = (c1 - c0) / (dz)
    i = 0
    # iterate
    qContinue = 1
    while qContinue == 1:
        if np.absolute(g * st0) > 1e-10:  # cas d'une courbure "significative"
            R = -c0 / (g * st0)
            st1 = st0 - dz / R
            if np.absolute(st1) > 1:
                print("ERROR : le rayon fait demi-tour !!!")
                return -1
            theta1 = np.arcsin(st1)
            ct1 = np.cos(theta1)
            dt = (1 / g) * np.log((c1 / c0) * ((1 + ct0) / (1 + ct1)))
            if t + dt > owt:  # on est arrivé, calcul du dz à partir du dt
                qContinue = 0
                dt = owt - t
                theta1 = 2 * np.arctan(np.tan(theta0 / 2) * np.exp(g * dt))
                ct1 = np.cos(theta1)
                st1 = np.sin(theta1)
                dz = -R * (st1 - st0)
            t += dt
            x += R * (ct1 - ct0)
            z += dz
            r -= R * (theta1 - theta0)
        else:  # cas du rayon de courbure infini
            ct1 = ct0
            st1 = st0
            theta1 = theta0
            if np.absolute(g) > 1e-10:  # gradient significatif
                dt = (1 / g) * np.log(c1 / c0)
                if t + dt > owt:  # on est arrivé, calcul du dz à partir du dt
                    qContinue = 0
                    dt = owt - t
                    c1 = c0 * np.exp(g * dt)
                    dz = (c1 - c0) / g
                t += dt
                x += st1 / ct1 * dz
                z += dz
                r += dz / ct1
            else:  # célérité constante
                dt = (dz / ct1) / c0
                if t + dt > owt:  # on est arrivé, calcul du dz à partir du dt
                    qContinue = 0
                    dt = owt - t
                    dz = dt * ct1 * c0
                t += dt
                x += st0 / ct0 * dz
                z += dz
                r += dz / ct0
        if qContinue == 1:  # préparation à l'itération suivante
            i += 1
            ct0 = ct1
            st0 = st1
            c0 = c1
            theta0 = theta1
            if i + 1 >= svp.shape[0]:  # isocélère
                c1 = c0
                g = 0
                dz = 1000
            else:
                c1 = svp['vitesse'].iloc[i + 1]
                dz = svp['profondeur'].iloc[i + 1] - svp['profondeur'].iloc[i]
                g = (c1 - c0) / dz
    return x, z
