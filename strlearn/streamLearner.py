# -*- encoding: utf-8 -*-
import arff
from sklearn import preprocessing, base
import streamController as sc
import numpy as np
import time
import csv
from tqdm import tqdm

class StreamLearner:
    '''
    Inicjalizator przyjmuje:
    - `stream` - strumień danych jako binarny plik arff,
    - `clf` - obiekt klasyfikatora z sklearn, który musi obsługiwać uczenie inkrementacyjne (metoda `partial_fit`),
    - `chunk_size` - wielkość pojedynczej paczki z danymi,
    - `evaluate_interval` - liczba wzorców po których przetworzeniu przeprowadzamy ewaluację, jako zbiór testowy przyjmując wszystkie próbki przetworzone od poprzedniej ewaluacji
    - `controller` - delegat kontrolujący przetwarzanie, domyślnie pusty StreamController
    - `verbose` - flaga określająca czy learner ma raportować kolejne ewaluacje do stdout
    '''
    def __init__(self, stream, base_classifier, chunk_size=200, evaluate_interval=1000, controller=sc.StreamController(), verbose=True):
        # Assigning parameters
        self.base_classifier = base_classifier
        self.chunk_size = chunk_size
        self.evaluate_interval = evaluate_interval
        self.controller = controller
        self.controller.learner = self
        self.verbose = verbose

        # Loading dataset
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        self.classes = dataset['attributes'][-1][-1]
        self.X = data[:,:-1].astype(np.float)
        self.y = data[:,-1]

        # Data analysis
        self.number_of_samples = len(self.y)
        self.number_of_classes = len(self.classes)

        # Prepare to classification
        self.reset()


    def reset(self):
    '''
    Przygotowanie do procesu klasyfikacji. Zerowanie modelu klasyfikatora, liczników i list zbierających wyniki. Dodatkowo, przygotowanie kontrolera przetwarzania.
    '''
        self.clf = base.clone(self.base_classifier)
        self.evaluations = 0
        self.processed_chunks = 0
        self.processed_instances = 0

        self.scores = []
        self.score_points = []
        self.training_times = []
        self.evaluation_times = []
        self.controller_measures = []

        self.previous_chunk = None
        self.chunk = None

        self.controller.prepare()

    '''
    Uruchomienie procesu uczenia. Zainicjalizowanie licznika i wyzwalanie przetwarzania chunka aż do chwili, w której wyczerpie się strumień.
    '''
    def run(self):
        self.training_time = time.time()
        for i in tqdm(xrange(self.number_of_samples / self.chunk_size), desc='CHN'):
            self.process_chunk()

    '''
    Przetwarzanie pojedynczego chunka.
    '''
    def process_chunk(self):
        # Kopiujemy zużyty przy poprzednim powtórzeniu chunk do pamięci jako poprzedni i pobieramy nowy ze strumienia.
        self.previous_chunk = self.chunk
        startpoint = self.processed_chunks * self.chunk_size
        self.chunk = (self.X[startpoint:startpoint + self.chunk_size], self.y[startpoint:startpoint + self.chunk_size])

        # Poinformuj kontroler przetwarzania o przystąpieniu do analizy kolejnego chunka.
        self.controller.next_chunk()

        # Inicjalizujemy zbiór do douczania.
        X, y = [], []

        # Iterujemy próbki z chunka.
        for sid, x in enumerate(self.chunk[0]):
            # Sprawdzamy, czy według kontrolera wystąpił już warunek przerwania.
            if not self.controller.should_break_chunk(X):
                # Pobieramy pojedynczy wzorzec, wraz z etykietą.
                label = self.chunk[1][sid]

                # Sprawdzamy, czy według kontrolera, mamy włączyć aktualną próbkę do zbioru douczającego.
                if self.controller.should_include(X, x, label):
                    X.append(x)
                    y.append(label)

            # Weryfikujemy, czy należy rozpocząc ewaluację.
            self.processed_instances += 1
            if self.processed_instances % self.evaluate_interval == 0:
                self.evaluate()

        X = np.array(X)
        y = np.array(y)

        # Douczamy z aktualnym zbiorem douczającym.
        self.fit_with_chunk(X, y)
        self.processed_chunks += 1

    def fit_with_chunk(self, X, y):
        # Convert to numpy
        if X.ndim == 2:
            self.clf.partial_fit(X, y, self.classes)

    def evaluate(self):
        self.training_time = time.time() - self.training_time
        evaluation_time = time.time()

        # Prepare evaluation chunk
        startpoint = (self.evaluations - 1) * self.evaluate_interval

        if startpoint > 0:
            evaluation_chunk = (self.X[startpoint:startpoint + self.evaluate_interval], self.y[startpoint:startpoint + self.evaluate_interval])

            # Create empty training set
            X, y = evaluation_chunk

            score = self.clf.score(X, y)
            evaluation_time = time.time() - evaluation_time

            controller_measure = self.controller.get_measures()

            # Collecting results
            self.score_points.append(self.processed_instances)
            self.scores.append(score)
            self.evaluation_times.append(evaluation_time)
            self.training_times.append(self.training_time)
            self.controller_measures.append(controller_measure)

            # Presenting results
            if self.verbose:
                print '%i, %.3f, %.3f, %.3f, %s' % (
                    self.processed_instances, score, evaluation_time, self.training_time, controller_measure
                )

        self.evaluations += 1

        self.training_time = time.time()

    def serialize(self, filename):
        with open(filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            for idx, point in enumerate(self.score_points):
                spamwriter.writerow([
                    '%i' % self.score_points[idx],
                    '%.3f' % self.scores[idx],
                    '%.0f' % (self.evaluation_times[idx] * 1000.),
                    '%.0f' % (self.training_times[idx] * 1000.),
                    self.controller_measures[idx]
                ])
