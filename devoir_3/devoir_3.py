import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sg


def computeMatrices(beta, alpha):
    """
    Retourne les matrices A, B, C et D du système discret.
    """
    A = np.array([-alpha])
    B = np.array([beta])
    C = np.array([-alpha])
    D = np.array([beta])

    return (A, B, C, D)


def systemeDiscret(A, B, C, D, ts, x=None):
    """
    Retourne la réponse du système discret à une entrée x (impulsion si pas spécifiée).
    """
    # ---- Version explicite ----
    # # Initialisation
    # ty = np.arange(0, ts * 100, ts)
    # q = np.zeros(len(ty))
    # y = np.zeros(len(ty))
    # # Si x = None, on considère une impulsion
    # if x is None:
    #     x = np.zeros(len(ty))
    #     x[0] = 1
    #
    # # Calcul de la réponse
    # for i in range(len(ty) - 1):  # -1 pour éviter l'indice out of range
    #     y[i] = C[0] * q[i] + D[0] * x[i]
    #     q[i + 1] = A[0] * q[i] + B[0] * x[i]
    #
    # # problème d'indice avec q[i+1] -> on calcule la dernière valeur de y à part
    # y[-1] = C * q[-1] + D * x[-1]
    # return (ty, y)

    # ---- Version avec scipy ----
    systeme = sg.StateSpace(A, B, C, D, dt=ts)

    if x is None:
        ty, y = sg.dimpulse(systeme)

    else:
        t, y, x = sg.dlsim(systeme, x)
        ty = np.squeeze(t)

    y = np.squeeze(y)

    return (ty, y)


def test_systemeDiscret():
    # Créer un signal sinusoidal discret
    ts = 0.2
    tx = np.arange(0, 20 + ts, ts)
    x = np.sin(0.5 * np.pi * tx)

    # Définir les paramètres du système
    beta = 2
    alpha = 0.8

    # Calculer la représentation d'état
    A, B, C, D = computeMatrices(beta, alpha)
    th, h = systemeDiscret(A, B, C, D, ts)
    ty, y = systemeDiscret(A, B, C, D, ts, x)
    # Affichage
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({"font.size": 18})

    mk_x, st_x, bl_x = plt.stem(tx, x, label="$x[n]$")
    mk_h, st_h, bl_h = plt.stem(th, h, label="$h[n]$")
    mk_y, st_y, bl_y = plt.stem(ty, y, label="$y[n]$")

    mks = (mk_x, mk_h, mk_y)
    sts = (st_x, st_h, st_y)
    bls = (bl_x, bl_h, bl_y)

    colors = ("b", "g", "r")
    sizes = (5, 6, 7)

    # Donner du style aux signals de façon compacte
    for mk, st, bl, c, s in zip(mks, sts, bls, colors, sizes):
        bl.set_color("k")
        bl.set_linewidth(1)
        mk.set_markersize(s)
        mk.set_color(c)
        st.set_color(c)

    plt.xlim([0.0, 15.0])

    plt.title(
        "Entrée $x[n]$, réponse impulsionnelle $h[n]$ et sortie $y[n]$ correspondant à un système donné "
    )
    plt.xlabel("$n\cdot t_s$ [s]")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_systemeDiscret()
