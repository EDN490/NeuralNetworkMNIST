# Sidst opdateret: 2026-09-22 09:10 (Europe/Copenhagen)

"""
NeuralNetworkMNIST.py

Et neuralt netværk implementeret fra bunden med NumPy.

Formålet med programmet er at vise de grundlæggende principper
bag et neuralt netværk:

    - Data
    - Vektorer og matricer
    - Matrixmultiplikation
    - Neuroner
    - Vægte og bias
    - Activation functions
    - Forward propagation
    - Loss
    - Gradients
    - Backpropagation
    - Gradient descent
    - Batches
    - Epochs
    - Test accuracy

Netværkets struktur:

    784 -> 16 -> 16 -> 10

784 inputværdier
    ↓
16 neuroner
    ↓
16 neuroner
    ↓
10 outputværdier

MNIST:
    60.000 træningsbilleder
    10.000 testbilleder
    28 x 28 pixels = 784 pixels
    10 klasser: cifrene 0-9
"""

import csv
import os
import urllib.request

import numpy as np


# ============================================================
# FILER OG DOWNLOAD
# ============================================================

TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "mnist_test.csv"

TRAIN_URL = "https://pjreddie.com/media/files/mnist_train.csv"
TEST_URL = "https://pjreddie.com/media/files/mnist_test.csv"


def download_file(url, filename):
    """
    Downloader en fil, hvis den ikke allerede findes lokalt.

    Det betyder, at programmet kun downloader filen første gang.
    """

    # Hvis filen allerede findes, behøver vi ikke downloade den igen.
    if os.path.exists(filename):
        print(f"{filename} findes allerede - springer download over.")
        return

    print(f"Downloader {filename}...")

    # Nogle servere afviser requests, der ikke ligner en almindelig browser.
    # Derfor sender vi en User-Agent med requesten.
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        # Henter filen fra internettet.
        with urllib.request.urlopen(request) as response:

            # Åbner den lokale fil i "write binary"-tilstand.
            with open(filename, "wb") as file:
                file.write(response.read())

        print(f"{filename} er downloadet.")

    except Exception as error:
        # Hvis download fejler, fjerner vi en eventuel delvist downloadet fil.
        if os.path.exists(filename):
            os.remove(filename)

        print(f"Kunne ikke downloade {filename}.")
        print(f"Fejl: {error}")
        raise


def download_mnist():
    """
    Sørger for at både træningsdata og testdata findes lokalt.
    """

    download_file(TRAIN_URL, TRAIN_FILE)
    download_file(TEST_URL, TEST_FILE)


# ============================================================
# INDLÆSNING AF DATA
# ============================================================

def load_mnist(filename):
    """
    Læser en MNIST CSV-fil.

    CSV-format:

        label,pixel1,pixel2,...,pixel784

    Første kolonne:
        Det rigtige ciffer.

    De næste 784 kolonner:
        Pixelværdierne fra billedet.
    """

    images = []
    labels = []

    # Åbner CSV-filen.
    with open(filename, "r", newline="") as file:

        # csv.reader læser filen række for række.
        reader = csv.reader(file)

        for row in reader:

            # Springer tomme rækker over.
            if not row:
                continue

            # Hvis første værdi ikke kan konverteres til et heltal,
            # antager vi, at det er en header.
            try:
                label = int(row[0])
            except ValueError:
                continue

            # De resterende 784 værdier er pixels.
            pixels = [float(value) for value in row[1:]]

            # MNIST skal have præcis 784 pixels pr. billede.
            if len(pixels) != 784:
                raise ValueError(
                    f"Forventede 784 pixels, men fandt {len(pixels)}."
                )

            labels.append(label)
            images.append(pixels)

    # Konverterer listen med billeder til et NumPy-array.
    #
    # Resultatet bliver eksempelvis:
    #
    #     (60000, 784)
    #
    # 60000 billeder
    # 784 pixelværdier pr. billede
    images = np.array(images, dtype=np.float32)

    # Konverterer labels til et NumPy-array.
    #
    # Resultatet bliver eksempelvis:
    #
    #     (60000,)
    #
    labels = np.array(labels, dtype=np.int64)

    # Pixelværdierne går oprindeligt fra 0 til 255.
    #
    # Ved at dividere med 255 får vi værdier mellem:
    #
    #     0.0 og 1.0
    #
    # Det gør dataene nemmere for netværket at arbejde med.
    images = images / 255.0

    return images, labels


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def relu(x):
    """
    ReLU står for Rectified Linear Unit.

    ReLU er en activation function, som gør negative værdier
    til 0 og beholder positive værdier.
    """

    # ReLU:
    #
    #     max(0, x)
    #
    # Eksempel:
    #
    #     -2 -> 0
    #      0 -> 0
    #      3 -> 3

    return np.maximum(0, x)


def relu_derivative(x):
    """
    Den afledte af ReLU.

    Hvis x > 0:
        derivative = 1

    Hvis x <= 0:
        derivative = 0
    """

    return (x > 0).astype(np.float32)


def softmax(x):
    """
    Softmax omdanner outputværdierne til sandsynligheder.

    Hvis vi har 10 outputneuroner, får vi 10 tal.

    Summen af de 10 tal bliver 1.
    """

    # Vi trækker den største værdi fra hver række.
    #
    # Det ændrer ikke softmax-resultatet, men forhindrer
    # meget store e^x-værdier og gør beregningen stabil.
    #
    # axis=1 betyder, at vi finder den største værdi
    # for hvert billede.
    #
    # keepdims=True bevarer dimensionen, så resultatet
    # kan trækkes fra x med broadcasting.

    shifted = x - np.max(
        x,
        axis=1,
        keepdims=True
    )

    # Beregner e^x for alle værdier.
    exp_values = np.exp(shifted)

    # Dividerer hver værdi med summen af rækken.
    #
    # axis=1 betyder, at vi summerer de 10 outputværdier
    # for hvert billede.
    #
    # keepdims=True bevarer dimensionen.

    probabilities = exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )

    return probabilities


# ============================================================
# LOSS FUNCTION
# ============================================================

def cross_entropy_loss(predictions, labels):
    """
    Beregner Cross-Entropy Loss.

    Loss er et mål for, hvor forkert netværkets prediction er.

    Lav loss:
        Netværket giver høj sandsynlighed til det rigtige svar.

    Høj loss:
        Netværket giver lav sandsynlighed til det rigtige svar.
    """

    # Antal billeder i batchen.
    batch_size = len(labels)

    # Henter sandsynligheden for det korrekte label
    # for hvert billede.
    correct_probabilities = predictions[
        np.arange(batch_size),
        labels
    ]

    # Undgår log(0), som ikke er defineret.
    correct_probabilities = np.clip(
        correct_probabilities,
        1e-12,
        1.0
    )

    # Cross-Entropy:
    #
    #     -log(p)
    #
    # hvor p er sandsynligheden for det korrekte svar.

    loss = -np.mean(np.log(correct_probabilities))

    return loss


# ============================================================
# NEURAL NETWORK
# ============================================================

class NeuralNetwork:

    def __init__(self):

        # ----------------------------------------------------
        # LAG 1
        # ----------------------------------------------------

        # W betyder "weights".
        #
        # Weights bestemmer, hvor meget de forskellige input
        # skal påvirke neuronerne.
        #
        # Vi har:
        #
        #     784 inputværdier
        #
        # og:
        #
        #     16 neuroner
        #
        # Derfor har W1 dimensionerne:
        #
        #     (784, 16)

        self.W1 = (
            np.random.randn(784, 16)
            * np.sqrt(2 / 784)
        )

        # b betyder "bias".
        #
        # Bias giver hvert neuron en ekstra værdi,
        # som kan justeres under træningen.
        #
        # Dimension:
        #
        #     (1, 16)

        self.b1 = np.zeros((1, 16))

        # ----------------------------------------------------
        # LAG 2
        # ----------------------------------------------------

        # 16 neuroner -> 16 neuroner
        #
        # Derfor:
        #
        #     W2 = (16, 16)

        self.W2 = (
            np.random.randn(16, 16)
            * np.sqrt(2 / 16)
        )

        self.b2 = np.zeros((1, 16))

        # ----------------------------------------------------
        # LAG 3
        # ----------------------------------------------------

        # 16 neuroner -> 10 outputneuroner
        #
        # Derfor:
        #
        #     W3 = (16, 10)

        self.W3 = (
            np.random.randn(16, 10)
            * np.sqrt(2 / 16)
        )

        self.b3 = np.zeros((1, 10))

    # ========================================================
    # FORWARD PROPAGATION
    # ========================================================

    def forward(self, X):
        """
        Forward propagation betyder, at data bevæger sig
        fremad gennem netværket:

            input
              ↓
            lag 1
              ↓
            lag 2
              ↓
            output

        Her beregner netværket sin prediction.
        """

        # ----------------------------------------------------
        # LAG 1
        # ----------------------------------------------------

        # Først beregnes den vægtede sum:
        #
        #     z = X @ W + b
        #
        # z er altså resultatet før activation function.
        #
        # a bruges om resultatet efter activation function.

        # Matrixmultiplikation:
        #
        # X har dimensionerne:
        #
        #     (64, 784)
        #
        # Det betyder:
        #
        #     64 billeder
        #     784 inputværdier pr. billede
        #
        # W1 har dimensionerne:
        #
        #     (784, 16)
        #
        # Det betyder:
        #
        #     784 inputværdier
        #     16 neuroner
        #
        # @ betyder matrixmultiplikation.
        #
        # Ved matrixmultiplikation skal de indre dimensioner
        # være ens:
        #
        #     (64, 784) @ (784, 16)
        #            ↑       ↑
        #          784 = 784
        #
        # Resultatet får de ydre dimensioner:
        #
        #     (64, 16)
        #
        # Vi får altså 16 værdier for hvert af de 64 billeder.
        #
        # Den generelle regel er:
        #
        #     (m, n) @ (n, p) -> (m, p)

        self.z1 = X @ self.W1 + self.b1

        # Activation function anvendes på z1.
        #
        # a1 er derfor outputtet fra første lag
        # efter ReLU.

        self.a1 = relu(self.z1)

        # ----------------------------------------------------
        # LAG 2
        # ----------------------------------------------------

        # a1:
        #
        #     (64, 16)
        #
        # W2:
        #
        #     (16, 16)
        #
        # Derfor:
        #
        #     (64, 16) @ (16, 16)
        #
        # bliver:
        #
        #     (64, 16)

        self.z2 = self.a1 @ self.W2 + self.b2

        self.a2 = relu(self.z2)

        # ----------------------------------------------------
        # LAG 3
        # ----------------------------------------------------

        # a2:
        #
        #     (64, 16)
        #
        # W3:
        #
        #     (16, 10)
        #
        # Derfor:
        #
        #     (64, 16) @ (16, 10)
        #
        # bliver:
        #
        #     (64, 10)
        #
        # Vi ender altså med 10 outputværdier
        # for hvert billede.

        self.z3 = self.a2 @ self.W3 + self.b3

        # Softmax omdanner de 10 outputværdier
        # til sandsynligheder.

        self.output = softmax(self.z3)

        return self.output

    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    def backward(self, X, labels):
        """
        Backpropagation beregner, hvordan hver vægt og bias
        har bidraget til fejlen.

        Fejlen fra output sendes derfor baglæns gennem netværket,
        så vi kan beregne gradients for parametrene.

        Gradienterne fortæller, hvordan parametrene skal ændres
        for at reducere loss.
        """

        batch_size = len(labels)

        # ----------------------------------------------------
        # OUTPUT LAG
        # ----------------------------------------------------

        # Vi starter med outputtet fra softmax.
        dz3 = self.output.copy()

        # For hvert billede trækker vi 1 fra
        # sandsynligheden for det korrekte label.

        dz3[
            np.arange(batch_size),
            labels
        ] -= 1

        # Vi beregner gennemsnittet over batchen.
        dz3 /= batch_size

        # ----------------------------------------------------
        # GRADIENT FOR W3
        # ----------------------------------------------------

        # a2:
        #
        #     (64, 16)
        #
        # a2.T:
        #
        #     (16, 64)
        #
        # dz3:
        #
        #     (64, 10)
        #
        # Derfor:
        #
        #     (16, 64) @ (64, 10)
        #
        # bliver:
        #
        #     (16, 10)
        #
        # Det passer med W3.

        dW3 = self.a2.T @ dz3

        # Summerer gradienten for alle billeder.
        db3 = np.sum(
            dz3,
            axis=0,
            keepdims=True
        )

        # ----------------------------------------------------
        # SEND GRADIENTEN BAGLÆNS TIL LAG 2
        # ----------------------------------------------------

        # dz3:
        #
        #     (64, 10)
        #
        # W3.T:
        #
        #     (10, 16)
        #
        # Derfor:
        #
        #     (64, 10) @ (10, 16)
        #
        # bliver:
        #
        #     (64, 16)

        da2 = dz3 @ self.W3.T

        # ReLU's afledte bestemmer, hvor gradienten
        # skal sendes videre.

        dz2 = da2 * relu_derivative(self.z2)

        # ----------------------------------------------------
        # GRADIENT FOR W2
        # ----------------------------------------------------

        # a1.T:
        #
        #     (16, 64)
        #
        # dz2:
        #
        #     (64, 16)
        #
        # Derfor:
        #
        #     (16, 64) @ (64, 16)
        #
        # bliver:
        #
        #     (16, 16)

        dW2 = self.a1.T @ dz2

        db2 = np.sum(
            dz2,
            axis=0,
            keepdims=True
        )

        # ----------------------------------------------------
        # SEND GRADIENTEN BAGLÆNS TIL LAG 1
        # ----------------------------------------------------

        # dz2:
        #
        #     (64, 16)
        #
        # W2.T:
        #
        #     (16, 16)
        #
        # Derfor:
        #
        #     (64, 16) @ (16, 16)
        #
        # bliver:
        #
        #     (64, 16)

        da1 = dz2 @ self.W2.T

        dz1 = da1 * relu_derivative(self.z1)

        # ----------------------------------------------------
        # GRADIENT FOR W1
        # ----------------------------------------------------

        # X:
        #
        #     (64, 784)
        #
        # X.T:
        #
        #     (784, 64)
        #
        # dz1:
        #
        #     (64, 16)
        #
        # Derfor:
        #
        #     (784, 64) @ (64, 16)
        #
        # bliver:
        #
        #     (784, 16)
        #
        # Det er præcis samme dimension som W1.

        dW1 = X.T @ dz1

        db1 = np.sum(
            dz1,
            axis=0,
            keepdims=True
        )

        # Returnerer alle gradients.
        return dW1, db1, dW2, db2, dW3, db3

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, learning_rate, gradients):
        """
        Opdaterer vægte og bias med gradient descent.

        Learning rate bestemmer størrelsen på det skridt,
        vi tager, når parametrene opdateres.

        Grundideen er:

            parameter =
                parameter - learning_rate * gradient
        """

        dW1, db1, dW2, db2, dW3, db3 = gradients

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3


# ============================================================
# MAIN
# ============================================================

def main():

    print("Neural Network MNIST")
    print("====================")

    # Sørger for at MNIST CSV-filerne findes lokalt.
    #
    # Hvis de allerede findes, bliver de ikke downloadet igen.

    download_mnist()

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    X_train, y_train = load_mnist(TRAIN_FILE)

    # X_train:
    #
    #     (60000, 784)
    #
    # 60000 billeder
    # 784 inputværdier pr. billede
    #
    # y_train:
    #
    #     (60000,)
    #
    # Ét label pr. billede.

    print()
    print("Training data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    X_test, y_test = load_mnist(TEST_FILE)

    print()
    print("Test data:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # --------------------------------------------------------
    # CREATE NETWORK
    # --------------------------------------------------------

    network = NeuralNetwork()

    # Learning rate bestemmer hvor store skridt
    # vi tager, når vægtene opdateres.

    learning_rate = 0.01

    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------
    #
    # En epoch betyder:
    #
    #     Hele træningsdatasættet er blevet brugt én gang.
    #
    # Vi har:
    #
    #     60.000 træningsbilleder
    #
    # Med 10 epochs ser netværket derfor
    # de samme 60.000 billeder 10 gange.
    #
    # Netværket starter ikke forfra ved hver epoch.
    #
    # Efter første epoch er vægtene blevet ændret.
    #
    # Når anden epoch begynder, fortsætter netværket
    # derfor med de nye vægte fra første epoch.
    #
    # Det samme sker mellem alle efterfølgende epochs.
    #
    # Flere epochs giver netværket flere muligheder
    # for gradvist at forbedre sine vægte.

    epochs = 10

    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------
    #
    # Vi bruger ikke alle 60.000 billeder på én gang.
    #
    # I stedet deler vi træningsdata op i batches.
    #
    # Her bruger vi:
    #
    #     64 billeder pr. batch
    #
    # For hvert batch:
    #
    #     1. Forward propagation
    #     2. Beregn loss
    #     3. Backpropagation
    #     4. Opdater vægte
    #
    # Når alle batches er behandlet, er én epoch færdig.

    batch_size = 64

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(epochs):

        # Vi laver en tilfældig rækkefølge af træningsdata.
        #
        # Det betyder, at batches ikke altid indeholder
        # de samme billeder i samme rækkefølge.

        indices = np.random.permutation(len(X_train))

        total_loss = 0.0
        batch_count = 0

        # Går gennem hele træningsdatasættet
        # batch for batch.

        for start in range(
            0,
            len(X_train),
            batch_size
        ):

            # Finder slutningen på det aktuelle batch.
            end = start + batch_size

            # Henter indeksene til dette batch.
            batch_indices = indices[start:end]

            # Henter billederne.
            X_batch = X_train[batch_indices]

            # Henter de korrekte labels.
            y_batch = y_train[batch_indices]

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            predictions = network.forward(X_batch)

            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = cross_entropy_loss(
                predictions,
                y_batch
            )

            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            gradients = network.backward(
                X_batch,
                y_batch
            )

            # ------------------------------------------------
            # UPDATE
            # ------------------------------------------------

            network.update(
                learning_rate,
                gradients
            )

            # Gemmer loss for senere at kunne
            # beregne gennemsnittet.

            total_loss += loss
            batch_count += 1

        # Gennemsnitlig loss for hele epoch.

        average_loss = total_loss / batch_count

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    # ========================================================
    # TEST
    # ========================================================

    print()
    print("Testing...")

    # Sender alle testbilleder gennem netværket.
    test_predictions = network.forward(X_test)

    # Finder det output med den højeste sandsynlighed.
    #
    # np.argmax betyder:
    #
    #     Find positionen for den største værdi.
    #
    # axis=1 betyder:
    #
    #     Gør det for hvert billede separat.
    #
    # Eksempel:
    #
    #     [0.01, 0.02, 0.80, 0.03, ...]
    #
    # giver prediction:
    #
    #     2

    predicted_labels = np.argmax(
        test_predictions,
        axis=1
    )

    # Sammenligner vores prediction med det rigtige label.
    #
    # Resultatet bliver en række True/False-værdier:
    #
    #     True  = korrekt
    #     False = forkert

    correct = predicted_labels == y_test

    # np.mean() behandler True som 1 og False som 0.
    #
    # Derfor får vi andelen af korrekte predictions.

    accuracy = np.mean(correct)

    print(
        f"Test accuracy: {accuracy * 100:.2f}%"
    )

    # ========================================================
    # EKSEMPLER
    # ========================================================

    print()
    print("Eksempler:")

    # Viser de første 10 testeksempler.

    for i in range(10):

        prediction = predicted_labels[i]
        actual = y_test[i]

        # Sandsynligheden for den valgte prediction.
        confidence = test_predictions[i, prediction]

        print(
            f"Eksempel {i + 1}: "
            f"Prediction = {prediction}, "
            f"Actual = {actual}, "
            f"Confidence = {confidence:.4f}"
        )


# ============================================================
# START PROGRAMMET
# ============================================================

if __name__ == "__main__":
    main()