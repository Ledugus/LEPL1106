import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io.wavfile import read, write
from scipy import signal
import time


def create_filter(filterName, M, cutoff, fech, typeF="lowpass"):
    # Code pour concevoir le filtre
    if filterName == "Butterworth":
        b, a = signal.butter(M - 1, cutoff, typeF, analog=False, fs=fech)
    elif filterName == "ChebyshevI":
        b, a = signal.cheby1(M - 1, 3, cutoff, typeF, analog=False, fs=fech)
    elif filterName == "ChebyshevII":
        b, a = signal.cheby2(M - 1, 20, cutoff, typeF, analog=False, fs=fech)
    filterSys = signal.dlti(b, a)
    th, he = signal.dimpulse(filterSys)

    return np.reshape(he, np.size(he)), b, a


filterType = "Butterworth"  # Concevoir un filtre Butterworth
filterOrder = 3  # Choisir un ordre de filtre approprié
cutoffFrequency = 150  # Choisir une fréquence de coupure appropriée
samplingFrequency = 1000  # Choisir une fréquence d'échantillonnage appropriée

he, b, a = create_filter(
    filterType, filterOrder, cutoffFrequency, samplingFrequency, "lowpass"
)


def applyFilter(inputSignal, he, segmentSize, overlap):
    """
    Applique un filtre à un signal d'entrée en utilisant une approche de filtrage par segments.
    Utilisation de la FFT et gestion des recouvrements et des effets de bord.

    Arguments:
    ---------
    - inputSignal : numpy array : signal d'entrée à filtrer
    - he : numpy array : réponse impulsionnelle du filtre
    - segmentSize : int : taille de chaque segment
    - overlap : int : taille de la zone de recouvrement entre les segments

    Retourne:
    --------
    - filtered_signal : numpy array : signal filtré
    """
    num_segments = (len(inputSignal) + segmentSize - overlap - 1) // (
        segmentSize - overlap
    )

    filtered_signal = np.zeros_like(inputSignal, dtype=np.float64)
    for i in range(num_segments):
        start = i * (segmentSize - overlap)
        end = min(start + segmentSize, len(inputSignal))
        segment = inputSignal[start:end]

        # Appliquer la FFT au segment
        segment_freq = np.fft.fft(segment, n=segmentSize)
        freq_resp = np.fft.fft(he, n=segmentSize)
        # Filtrer le segment dans le domaine fréquentiel
        filtered_segment_freq = segment_freq * freq_resp
        # Appliquer la FFT inverse pour obtenir le segment filtré
        filtered_segment = np.fft.ifft(filtered_segment_freq).real

        # Overlap-add method
        if i == 0:
            filtered_signal[:end] = filtered_segment
        else:
            # Overlap avant
            filtered_signal[start : start + overlap] += filtered_segment[:overlap]
            # Pas d'overlap au milieu
            filtered_signal[start : start + overlap] /= 2

            # Overlap après
            filtered_signal[start + overlap : end] = filtered_segment[
                overlap : end - start
            ]

    return filtered_signal


def plot_spectre(x, fech, x_f, name, bool_Ingi=True):
    """
    Trace le spectre d'un signal d'entrée et de son signal filtré.

    Arguments:
    ---------
    - x : numpy array : signal d'entrée
    - fech : int : fréquence d'échantillonnage
    - x_f : numpy array : signal filtré
    - name : str : nom du fichier de sortie

    Retourne:
    --------
    - None (le graphique est affiché ou enregistré)
    """
    t = 1 / fech * np.arange(0, len(x))

    X = np.fft.fft(x)
    freq_x = np.fft.fftfreq(len(x), 1 / fech)
    X_f = np.fft.fft(x_f)
    freq_x_f = np.fft.fftfreq(len(x_f), 1 / fech)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label="Signal d'entrée")
    plt.plot(t, x_f, label="Signal filtré")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 2)
    plt.plot(
        np.fft.fftshift(freq_x),
        np.fft.fftshift(np.abs(X)),
        label="Spectre du signal d'entrée",
    )
    plt.plot(
        np.fft.fftshift(freq_x_f),
        np.fft.fftshift(np.abs(X_f)),
        label="Spectre du signal filtré",
    )
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Module")
    plt.legend()

    # A COMPLETER

    plt.tight_layout()
    if bool_Ingi:
        plt.savefig(name + ".png", bbox_inches="tight")
    else:
        plt.show()


def applyFilter2(inputSignal, he, b, a):

    debut = time.perf_counter()
    output_oa = signal.oaconvolve(inputSignal, he, mode="same")
    fin = time.perf_counter()
    duree_execution = fin - debut
    print(
        f"La fonction signal.oaconvolve a pris {duree_execution} secondes pour s'exécuter."
    )

    debut = time.perf_counter()
    output_ifft = np.fft.ifft(
        np.fft.fft(inputSignal) * np.fft.fft(he, n=len(inputSignal))
    )
    fin = time.perf_counter()
    duree_execution = fin - debut
    print(
        f"La fonction np.ifft sans diviser le signal a pris {duree_execution} secondes pour s'exécuter."
    )

    debut = time.perf_counter()
    output_lfilter = signal.lfilter(b, a, inputSignal)
    fin = time.perf_counter()
    duree_execution = fin - debut
    print(
        f"La fonction signal.lfilter a pris {duree_execution} secondes pour s'exécuter."
    )

    # Calcul de la sortie par convolution classique
    debut = time.perf_counter()
    output_conv_directe = np.convolve(inputSignal, he, mode="same")
    fin = time.perf_counter()
    duree_execution = fin - debut
    print(f"La fonction np.convolve a pris {duree_execution} secondes pour s'exécuter.")

    return output_oa, output_ifft, output_conv_directe, output_lfilter


def test():
    filterType = "Butterworth"  # Concevoir un filtre Butterworth
    filterOrder = 3  # Choisir un ordre de filtre approprié
    cutoffFrequency = 150  # Choisir une fréquence de coupure appropriée
    samplingFrequency = 1000  # Choisir une fréquence d'échantillonnage appropriée
    t = np.arange(0, 1, 1 / samplingFrequency)
    inputSignal = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)

    he, b, a = create_filter(
        filterType, filterOrder, cutoffFrequency, samplingFrequency, "lowpass"
    )

    # Appliquer le filtre à l'entrée x
    debut = time.perf_counter()
    output = applyFilter(inputSignal, he, 50, 5)
    fin = time.perf_counter()
    duree_execution = fin - debut
    print(f"La fonction overlap-add a pris {duree_execution} secondes pour s'exécuter.")

    output_oa, output_ifft, output_conv, output_lfilter = applyFilter2(
        inputSignal, he, b, a
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
    plt.plot(t, np.real(output), label="overlap-and-add")
    plt.plot(t, np.real(output_conv), label="convolution")
    plt.plot(t, np.real(output_oa), label="oaconvolve", linestyle="-.")
    plt.plot(t, np.real(output_ifft), label="fft over the entire signal", linestyle=":")
    plt.plot(t, np.real(output_lfilter), label="lfilter", linestyle="--")
    plt.title("Sortie du filtre par la FFT et la convolution")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.xlim([0, 0.1])

    plt.tight_layout()
    plt.show()
    plot_spectre(inputSignal, samplingFrequency, output, "no", False)


def test_sound():
    fech, x = read("data/sound.wav")  # Music: https://www.purple-planet.com
    he, b, a = create_filter("Butterworth", 3, 500, fech, "lowpass")
    # Appliquer le filtre à l'entrée x
    # Les paramètres SegmentSize et overlap sont importants et doivent être choisis avec reflexion.
    # Vous pouvez les faire varier et en observer les conséquences si vous le désirer.
    debut = time.perf_counter()
    output = applyFilter(x, he, 50000, 0)
    fin = time.perf_counter()
    duree_execution = fin - debut
    print(f"La fonction overlap-add a pris {duree_execution} secondes pour s'exécuter.")
    output_oa, output_ifft, output_conv, output_lfilter = applyFilter2(x, he, b, a)

    plot_spectre(x, fech, output, "no", False)
    # Enregistrement du son pour vous permettre d'écouter les effets du filtrage passe-bas
    write("data/sortie.wav", fech, np.asarray(np.real(output), dtype=np.int16))


if __name__ == "__main__":
    test()
    # test_sound()
