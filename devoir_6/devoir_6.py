import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy import signal

import scipy


def create_applyFilter(inputSignal, filterName, M, cutoff, fech, typeF):
    """
    Conçoit et applique un filtre numérique à un signal d'entrée.

    Paramètres :
    ------------
    inputSignal : array_like
        Le signal d'entrée à filtrer.
    filterName : str
        Nom du filtre à utiliser ('Butterworth', 'Chebyshev1' ou 'Chebyshev2').
        Butterworth est utilisé par défaut.
    M : int
        Ordre du filtre.
    cutoff : float ou array_like
        Fréquence(s) de coupure du filtre (en Hz).
    fech : float
        Fréquence d'échantillonnage du signal (en Hz).
    typeF : str
        Type de filtre ('lowpass', 'highpass', 'bandpass' ou 'bandstop').

    Retourne :
    ----------
    w : ndarray
        Fréquences pour la réponse en fréquence.
    h_f : ndarray
        Réponse en fréquence du filtre.
    output : ndarray
        Signal filtré.
    """

    if filterName == "Butterworth":
        b, a = signal.butter(M - 1, cutoff, btype=typeF, analog=False, fs=fech)
    elif filterName == "Chebyshev1":
        b, a = signal.cheby1(M - 1, 3, cutoff, btype=typeF, analog=False, fs=fech)
    elif filterName == "Chebyshev2":
        b, a = signal.cheby2(M - 1, 20, cutoff, btype=typeF, analog=False, fs=fech)
    else:
        b, a = signal.butter(M - 1, cutoff, btype=typeF, analog=False, fs=fech)

    systeme = signal.dlti(a, b)
    t, h = signal.dimpulse(systeme)
    h = np.squeeze(h)[: len(inputSignal)]
    output = np.convolve(inputSignal, h, mode="same")
    w, h_f = signal.freqz(b, a, fs=fech)

    return w, h_f, output


def plot_Bode(w, h, name):
    """
    Trace le diagramme de Bode (module et phase) d'un filtre.

    Paramètres :
    ------------
    w : array_like
        Fréquences en radians par seconde.
    h : array_like
        Réponse fréquentielle complexe du filtre.
    name : str
        Nom du fichier dans lequel sauvegarder le graphique.

    Retour :
    --------
    Aucun. La fonction affiche ou sauvegarde directement les graphiques.
    """
    magnitude_db = 20 * np.log10(np.abs(h))
    phase_deg = np.angle(h) * 180 / np.pi

    plt.figure(figsize=(12, 6))

    # Plot du module en [décibels]
    plt.subplot(211)
    plt.semilogx(w, magnitude_db)
    plt.title("Diagramme de Bode - Module")
    plt.xlabel("Fréquence [Hz]")
    plt.ylabel("Module [dB]")

    # Plot de la phase [degrés]
    plt.subplot(212)
    plt.semilogx(w, phase_deg)
    plt.title("Diagramme de Bode - Phase")
    plt.xlabel("Fréquence [Hz]")
    plt.ylabel("Phase [degrés]")

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(name + ".png")
    # NE PAS SOUMETTRE AVEC CETTE LIGNE
    plt.show()


def test_filtre():
    filterType = "Butterworth"  # Concevoir un filtre Butterworth
    # filterType = "Chebyshev1"  # Concevoir un filtre Butterworth
    # filterType = "Chebyshev2"  # Concevoir un filtre Chebyshev de type II
    filterOrder = 3  # Choisir un ordre de filtre approprié
    cutoffFrequency = 150  # Choisir une fréquence de coupure appropriée
    samplingFrequency = 1000  # Choisir une fréquence d'échantillonnage appropriée
    typeF = "highpass"  # Choisir un type de filtre approprié

    t = np.arange(0, 1, 1 / samplingFrequency)
    inputSignal = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)

    w, h, output = create_applyFilter(
        inputSignal,
        filterType,
        filterOrder,
        cutoffFrequency,
        samplingFrequency,
        typeF=typeF,
    )

    # Visualiser les résultats
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, inputSignal)
    plt.title("Signal d'entrée x(t)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 0.1])

    plt.subplot(2, 1, 2)
    plt.plot(t, np.real(output))
    plt.title("Sortie du filtre calculé par la convolution")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.xlim([0, 0.1])

    plt.tight_layout()
    plt.show()
    plot_Bode(w, h, "devoir_6/bode")


if __name__ == "__main__":
    test_filtre()
