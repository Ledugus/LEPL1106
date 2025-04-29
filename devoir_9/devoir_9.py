import numpy as np


def RouthTable(A, n):
    """Construit le tableau de Routh à partir du polynôme A
    Args:
        A (list): Coefficients du polynôme
        n (int): Degré du polynôme

    Returns:
        tab (np.ndarray) : Tableau de Routh du polynôme A
    """

    # Initialisation du tableau de Routh
    # Par défaut, tous les coefficients sont nuls
    tab = np.zeros((n + 1, (n + 2) // 2))  # Ligne à conserver dans votre code

    # Premières lignes du tableau
    tab[0, :] = A[-1::-2]  # Coefficients de la première ligne
    tab[1, : len(A[-2::-2])] = A[-2::-2]  # Coefficients de la deuxième ligne

    # Remplissage du tableau de Routh
    for i in range(2, n + 1):
        for j in range(tab.shape[1] - 1):
            tab[i, j] = (
                -1
                * (
                    tab[i - 2, 0] * tab[i - 1, j + 1]
                    - tab[i - 1, 0] * tab[i - 2, j + 1]
                )
                / tab[i - 1, 0]
            )

    return tab  # Retourne le tableau de Routh


def respectRHcriterion(RouthTab):
    """Vérifie si le critère de Routh-Hurwitz est respecté à partir du tableau de Routh
    Args:
        RouthTab (np.ndarray): Tableau de Routh

    Returns:
        bool: True si le critère est respecté, False sinon
    """

    first_col = RouthTab[:, 0]  # Première colonne du tableau de Routh
    signe = np.sign(first_col[0])  # Signe du premier élément de la première colonne
    for i in range(1, len(first_col)):
        if first_col[i] == 0:
            return False
        if np.sign(first_col[i]) != signe:
            return False

    return True  # A MODIFIER : Retourne True si le critère est respecté et False sinon


def test_stability_criterion():
    A1 = [1, 2, 3, 3, 1]
    RouthTab1 = RouthTable(A1, len(A1) - 1)
    print(RouthTab1)
    stable = respectRHcriterion(RouthTab1)

    print("Le système A1 est stable :", stable)

    # Système instable
    A2 = [1, 4, 2, 5, 1, 3, 2]
    RouthTab2 = RouthTable(A2, len(A2) - 1)
    print(RouthTab2)
    unstable = respectRHcriterion(RouthTab2)

    print("Le système A2 est stable :", unstable)


if __name__ == "__main__":
    test_stability_criterion()
