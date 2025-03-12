import numpy as np
import matplotlib.pyplot as plt

# Tailles de police utilisées avec la librairie matplotlib
legendsize = 14
titlesize = 14
labelsize = 13


# A SOUMETTRE
def serieFourier(x, N):
    """
    Calcule la série de Fourier d'un signal x[n] pour N harmoniques (signal de période N).


    Argument :
    ----------
    x : np.array contenant les valeurs du signal d'entrée.

    N : période du signal.

    Retourne :
    ----------
    (center_X, k) : tuple contenant les coefficients de Fourier centrés et les indices correspondants.
    """

    X = np.fft.fft(x, n=N)
    # Version explicite des indices
    k = np.arange(-np.ceil((N - 1) / 2), np.floor((N - 1) / 2) + 1)
    # Version condensée, même résultat
    k_2 = np.arange(-N // 2, N // 2)
    center_X = np.fft.fftshift(X)
    center_X[np.abs(center_X) < 1e-9] = 0
    return (center_X / N, k)


def plotSerieFourier(x, X, k, name):
    """
    Affiche le signal x[n], le module et l'argument des coefficients de Fourier de x[n].

    Argument :
    ----------
    x : np.array contenant les valeurs du signal d'entrée.

    X : np.array contenant les coefficients de Fourier de x[n].

    k : indices des coefficients de Fourier de x[n].

    name : chemin du fichier sauvegardé (sans l'extension)

    Retourne :
    ----------
    None (sauvegarde la figure)

    """

    # Création de la figure, de taille fixe.
    plt.figure(figsize=(12, 16))

    # 1er plot
    plt.subplot(3, 1, 1)
    plt.title(r"Signal $x[n]$", fontsize=titlesize)
    plt.stem(np.arange(0, len(x)), x)

    plt.xlabel(r"$n$ [-]", fontsize=labelsize)
    plt.ylabel(r"$x[n]$ [-]", fontsize=labelsize)

    # 2e plot
    plt.subplot(3, 1, 2)
    plt.title(
        r"Module des coefficients de Fourier $X[k]$ de $x[n]$", fontsize=titlesize
    )
    plt.stem(k, np.abs(X))

    plt.ylabel(r"|X[k]| [-]", fontsize=labelsize)
    plt.xlabel("k [-]", fontsize=labelsize)

    # 3e plot
    plt.subplot(3, 1, 3)
    plt.title(
        r"Argument des coefficients de Fourier $X[k]$ de $x[n]$", fontsize=titlesize
    )
    plt.stem(k, np.angle(X) * 180 / np.pi)

    plt.ylim(-180, 180)
    plt.yticks(np.arange(-180, 181, 90))
    plt.ylabel(r"Arg(X[k]) [deg]", fontsize=labelsize)
    plt.xlabel("k [-]", fontsize=labelsize)

    plt.subplots_adjust(
        hspace=0.85
    )  # Pour ajuster l'espace vertical entre sous-figures

    # Sauvegarde de la figure avec le bon nom.
    # Le second argument rétrécit les marges, par défaut relativement larges.
    plt.savefig(name + ".png", bbox_inches="tight")

    ####### A SUPPRIMER : affiche la figure
    plt.show()
    ####### A SUPPRIMER : affiche la figure
    return


# FIN A SOUMETTRE


def test_serieFourier():
    # Création du signal période x[n], sur une seule période
    N = 2
    p = 2  # à modifier
    n = np.arange(0, p * N)
    x = 2 + np.cos(n * np.pi)  # signal de l'exercice 4.6 b)

    (X, k) = serieFourier(x, N)

    print("k :  ", k)
    print("Module de X :", np.abs(X))


N = 24  # Période du signal
p = 4  # Nombre de périodes du signal


def signal(N, p):
    n = np.arange(0, p * N)
    return 2 + 2 * np.cos(n * np.pi * 2 / N) + 6 * np.sin(n * np.pi * 4 / N)


def plot_signal():
    n = np.arange(0, p * N)
    x = signal(N, p)

    plt.figure(figsize=(12, 6))
    markerline, stemlines, baseline = plt.stem(n, x)
    plt.setp(baseline, color="k")
    plt.setp(stemlines, color="b", linewidth=2, linestyle="--")
    plt.setp(markerline, color="b", marker="o", markersize=10)
    plt.title(r"Signal $x[n]$", fontsize=titlesize)
    plt.xlabel(r"$n$ [-]", fontsize=labelsize)
    plt.ylabel(r"$x[n]$ [-]", fontsize=labelsize)
    plt.grid("on", alpha=0.3)
    plt.ylim((-0.2, 3.5))
    plt.show()


def test_plotFourier():
    x = signal(N, p)

    (X, k) = serieFourier(x, N)

    plotSerieFourier(x, X, k, "devoir_4/fourier")


if __name__ == "__main__":
    test_plotFourier()
