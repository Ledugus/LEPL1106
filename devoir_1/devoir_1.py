import numpy as np
import matplotlib.pyplot as plt


def echelon(n, n0, A):
    """
    Calcule la fonction échelon,
    définie comme w[n] = A si n >= n0, w[n] = 0 sinon.

    Arguments
    ---------
    n: numpy array contenant des indices (entiers) auxquels on applique la fonction delta de Kronecker.
    n0: entier, décalage de l'échelon.
    A: amplitude de l'échelon.

    Retourne
    --------
    result: numpy array de même taille que n contenant les valeurs w[n]
    """

    result = np.zeros(n.shape)

    # Dans numpy, la comparaison sur un array retourne l'ensemble des indices
    # pour lesquels la condition est vraie
    # On peut assigner la valeur A à tous ces indices en une seule opération
    result[n >= n0] = A

    return result


def plotEchelon(n, w, name):
    """Affiche le signal w(n) et sauvegarde la figure dans un fichier <name>.png."""

    # prendre la valeur maximale de w et son décalage
    A = max(w)
    n0 = n[np.argmax(w)]

    fs_text = 16
    fs_ticks = 14

    # Création de la figure, de taille fixe.
    plt.figure(figsize=(6, 3))
    plt.title(
        rf"Échelon discret décalé de ${n0}$ et d'amplitude ${A}$", fontsize=fs_text
    )
    markerline, stemlines, baseline = plt.stem(n, w)

    # Axes
    plt.xlabel("$n$", fontsize=fs_text)
    plt.ylabel("$w[n]$", fontsize=fs_text)
    plt.ylim((-0.1 * A, 1.25 * A))

    # Style du graphe
    baseline.set_color("k")
    markerline.set_markersize(6)
    baseline.set_linewidth(1)
    plt.xticks([0, n0], fontsize=fs_ticks, labels=[0, rf"$n_0={n0}$"])
    plt.yticks([0, A], fontsize=fs_ticks, labels=[0, rf"$A={A}$"])

    # Sauvegarde de la figure avec le bon nom.
    # Le second argument rétrécit les marges, par défaut relativement larges.
    plt.savefig(name + ".png", bbox_inches="tight")

    plt.show()  # De-commentez ceci pour tester "localement", ne pas inclure cette ligne sur Inginious !


n = np.arange(-10, 10)
plotEchelon(n, echelon(n, 1, 3), "devoir_1/echelon_0_1")
