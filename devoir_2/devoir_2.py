import numpy as np
import matplotlib.pyplot as plt

# Tailles de police et couleurs utilisées avec la librairie matplotlib
legendsize = 14
titlesize = 14
labelsize = 16
colors = {
    "orange": [0.894, 0.424, 0.039],
    "red": [0.753, 0.0, 0.0],
    "violet": [0.580, 0.0, 0.827],
    "green": [0.437, 0.576, 0.235],
    "lightgreen": [0.0, 0.9, 0.0],
    "darkgreen": [0.0, 0.5, 0.0],
    "blue": [0.0, 0.439, 0.753],
    "cyan": [0.0, 0.9, 1.0],
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "grey": [0.5, 0.5, 0.5],
    "lightgrey": [0.7, 0.7, 0.7],
    "maroon": [0.5, 0.0, 0.0],
    "brown": [0.545, 0.271, 0.075],
    "gold": [0.812, 0.710, 0.231],
    "pink": [1.0, 0.078, 0.576],
}

def convolution(x, nx, h, nh):
    """
    A COMPLETER
    """
    y = np.zeros(x.size + h.size - 1)
    h = h[::-1]
    ny = np.arange(nx[0] + nh[0], nx[-1] + nh[-1] + 1)
    # print(ny.shape, nh.shape, nx.shape)

    for i in range(len(y)):
        for j in range(len(h)):
            if 0 <= i - j < len(x):
                y[i] += x[i - j] * h[j]

    return (y, ny)


def plot_convolution(x, nx, y, ny, fig_name):
    """
    Dessine et sauvegarde la convolution entre deux signaux.

    Argument :
    ----------
    x : np.array contenant les valeurs du signal d'entrée.

    nx: np.array d'entiers consécutifs sur lesquels est itéré x.

    y : np.array contenant les valeurs du produit de la convolution de x et du deuxième signal

    ny: np.array d'entiers consécutifs sur lesquels est itéré y.

    fig_name : chemin du fichier sauvegardé (sans l'extension)

    Retourne :
    ----------
    plot en format png et return une figure du graphe
    """

    plt.figure(figsize=(15, 8))
    markerline, stemlines, baseline = plt.stem(ny, y)

    plt.setp(baseline, color=colors["black"])
    plt.setp(stemlines, color=colors["blue"], linewidth=3, linestyle="--")
    plt.setp(markerline, color=colors["blue"], marker="o", markersize=10)

    plt.title("Signal de sortie y[n] (Convolution)")
    plt.xlabel("n")
    plt.ylabel("y[n]")

    plt.savefig(fig_name + ".png", bbox_inches="tight")
    # plt.show()


def test_convolution():
    T = 4
    h = np.append(np.arange(1, T), np.arange(T, 0, -1))
    nh = np.arange(-(T - 1), T)
    # N.B.:
    # Nous vous fournissons ici des plots que nous considérons comme corrects.
    # Lorsque nous vous demandons des plots, faites bien attention à respecter les consignes spécifiques à chaque figure.
    # Si vous utilisez les variables définies au début du notebook (ex : labelsize), veillez à les re-définir sur INGInious.

    plt.figure(figsize=(12, 6))
    markerline, stemlines, baseline = plt.stem(nh, h)

    plt.setp(baseline, color=colors["black"])
    plt.setp(stemlines, color=colors["blue"], linewidth=3, linestyle="--")
    plt.setp(markerline, color=colors["blue"], marker="o", markersize=10)

    plt.title(f"Signal triangle h[n] (T = {T})", fontsize=titlesize)
    plt.xlabel(r"$n$ [-]", fontsize=labelsize)
    plt.ylabel(r"$h[n]$ [-]", fontsize=labelsize)
    plt.grid("on", alpha=0.3)
    plt.xlim((-4, 4))
    plt.ylim((-1, 5))
    plt.show()
    nx = np.arange(-10, 21)
    x = np.zeros(nx.size)
    deltas = [-7, 2, 13, 18]  # position des deltas
    x[deltas - nx[0]] = 1


    (y, ny) = convolution(x, nx, h, nh)
    plot_convolution(x, nx, y, ny, "devoir_2/plot_convolution")
    

def window(n, n0, n1, A):
    """
    cette fonction evalue la fonction h[n], c'est à dire vaut 1/M entre 0 et M et sinon 0

    Arguments :
    ----------
    n : tableaux des où évaluer la fonction
    n0 : indice de départ de la fenêtre
    n1 : dernier indice (non-compris) de la fenêtre
    A : amplitude (hauteur) fenêtre

    Retourne :
    ----------
    Une fonction prenant des valeurs binaires : 0 ou A en fonction
    """
    return A * ((n >= n0) & (n < n1))


def moving_average(x, M):
    """
    Arguments :
    ----------
    x : signal d'entrée
    M : taille de la fenêtre pour prendre la moyenne

    Retourne :
    ----------
    Un tableau de même taille que x contenant la moyenne des M dernières valeurs
    """
    n = np.arange(len(x))
    h = (1/M) * (window(n, 0, M, 1))

    return np.convolve(x, h)[:len(x)]
    

if __name__ == "__main__":
    test_convolution()
    # Production photovoltaïque journalière depuis le 27 mai 2023 de Mr et Mme Dupont (données gentiment transmises par Mr et Mme Dupont)
    x = np.load("devoir_2/prod_elec.npy")
    N = len(x)
    h = np.arange(N)  # jours
    y = moving_average(x, 14)  # Moyenne sur deux semaines

    plt.figure(figsize=(12, 6))
    plt.plot(
        h, x, label=r"Production quotidienne $x[n]$", linewidth=1, color=colors["blue"]
    )
    plt.plot(
        h,
        y,
        label=r"Production lissée par moyenne glissante $y[n]$ (après moyenne)",
        linewidth=3,
        color=colors["orange"],
    )
    plt.title("Production photovoltaïque de Mr et Mme Dupont", fontsize=titlesize)
    plt.xlabel("Jours, du 27 mai au 20 février", fontsize=labelsize)
    plt.ylabel("Rendement total [kWh]", fontsize=labelsize)
    plt.legend(fontsize=legendsize, framealpha=1, fancybox=False, edgecolor="k")
    plt.grid("on", alpha=0.3)
    plt.xlim((1, h[-1]))
    plt.ylim((0, 30))
    plt.xticks(h[::30])
    plt.show()

    import matplotlib

    matplotlib.__version__
