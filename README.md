# Model ratingowy do predykcji defaultu kredytowego

Projekt na przedmiot Interpretowalność i Wyjaśnialność Uczenia Maszynowego

---

## Opis projektu

Celem projektu jest stworzenie modelu ratingowego do predykcji defaultu kredytowego na podstawie zbioru danych dotyczących klientów banku (firm). W ramach projektu stworzyliśmy dwa modele: model regresji logistycznej (interpretowalny) oraz model lasu losowego (black box). Zastosowaliśmy techniki kalibracji oraz metody wyjaśnialności, przede wszystkim SHAP, aby lepiej zrozumieć działanie modeli.

---

## Instalacja

### Opcja 1: Conda (zalecane)

```bash
# Utwórz środowisko z pliku environment.yml
conda env create -f environment.yml

# Aktywuj środowisko
conda activate xai_projekt
```

### Opcja 2: pip

```bash
# Zainstaluj zależności z requirements.txt
pip install -r requirements.txt
```

### Wymagania

- Python 3.11+
- Kluczowe biblioteki:
  - scikit-learn (modele ML)
  - optuna (tuning hiperparametrów)
  - shap (wyjaśnialność)
  - pandas, numpy (przetwarzanie danych)
  - matplotlib, seaborn (wizualizacje)

---

## Struktura repozytorium

```plaintext
XAI_Projekt-1/
│
├── data/                          # Dane wejściowe i przetworzone
│   ├── train.csv                  # Zbiór treningowy (po feature selection)
│   ├── test.csv                   # Zbiór testowy (po feature selection)
│   ├── zbiór_5.csv                # Oryginalny zbiór danych
│   └── zbiór_5_preprocessed.csv   # Zbiór po wstępnym preprocessingu
│
├── src/                           # Skrypty Python
│   ├── constants.py               # Stałe i konfiguracja projektu
│   ├── utils.py                   # Funkcje pomocnicze
│   ├── data_processing.py         # Pipeline'y preprocessingu danych
│   ├── manual_preprocessing.py    # Ręczny preprocessing zbioru
│   ├── split_data.py              # Podział danych train/test
│   ├── feature_selection.py       # Selekcja cech (RFE)
│   ├── baseline_models.py         # Trenowanie modeli bazowych
│   ├── lr_tuning.py               # Tuning regresji logistycznej
│   ├── rf_tuning.py               # Tuning lasu losowego
│   ├── calibration.py             # Kalibracja modeli
│   ├── threshold_selection.py     # Dobór optymalnego progu
│   ├── confusion_matrices.py      # Generowanie macierzy pomyłek
│   └── shap_analysis.py           # Analiza SHAP
│
├── notebooks/                     # Jupyter notebooks
│   ├── eda.ipynb                  # Eksploracyjna analiza danych
│   ├── logistic_regression.ipynb  # Wstępne eksperymenty z regresją logistyczną
│   ├── random_forest.ipynb        # Wstępne eksperymenty z lasem losowym
│   └── calibration.ipynb          # Wstępne eksperymenty z kalibracją
│
├── models/                        # Zapisane modele
│   ├── *_full.pkl                 # Modele trenowane na pełnym zbiorze cech
│   ├── *_reduced.pkl              # Modele po feature selection
│   ├── tuned_*.pkl                # Modele po tuningu hiperparametrów
│   └── calibrated_*.pkl           # Modele skalibrowane
│
├── plots/                         # Wykresy i wizualizacje
│   ├── shap/                      # Wykresy SHAP
│   ├── confusion_matrices.png     # Macierze pomyłek
│   ├── *_calibration_comparison.png  # Porównanie metod kalibracji
│   └── *_reliability.png          # Diagramy reliability nieskalibrowanych modeli
│
├── slownik_zmiennych_opisy.csv    # Słownik zmiennych
├── main.py                        # Główny skrypt uruchomieniowy
└── README.md                      # Dokumentacja projektu
```

### Workflow projektu

1. **Preprocessing**: `manual_preprocessing.py` → `split_data.py`
2. **Feature Selection**: `feature_selection.py`
3. **Baseline Models**: `baseline_models.py`
4. **Hyperparameter Tuning**: `lr_tuning.py`, `rf_tuning.py`
5. **Calibration**: `calibration.py`
6. **Evaluation**: `threshold_selection.py`, `confusion_matrices.py`
7. **Explainability**: `shap_analysis.py`

---

## Uruchomienie

Po instalacji należy uruchomić skrypt `main.py`, który przeprowadzi cały proces od wczytania i przetwarzania danych, treningu modeli, kalibracji, aż po generowanie wizualizacji i wyjaśnień:

```bash
python main.py
```

Plik `constants.py` zawiera wszystkie stałe używane w projekcie i umożliwia ewentualne zmiany niektórych parametrów (takich jak rozmiar zbioru testowego, seed) przed wywołaniem skryptu `main.py`.

---

## Dane

Dane zawierają informacje o klientach banku, w tym zmienną celu `default`, która wskazuje, czy klient spłacił kredyt (0) czy nie (1). Zbiór ma 3000 obserwacji, 220 kolumn i jest niezbalansowany (około 6% obserwacji to przypadki defaultu):

![Rozkład zmiennej celu](plots/target_distribution.png)

---

## EDA i preprocessing

Po sprawdzeniu rozkładu zmiennej celu, przeprowadziliśmy podstawową eksploracyjną analizę danych, w tym narysowanie macierzy korelacji, narysowanie kilku rozkładów cech.

![Macierz korelacji](plots/corr_matrix.png)

![Przykładowe rozkłady cech](plots/example_histograms.png)

Dzięki analizie danych zidentyfikowaliśmy kilka problemów, w tym występujące w danych wartości inf, outliery oraz wartości 0, które w niektórych kolumnach prawdopodobnie oznaczały brak danych. Ostateczny preprocessing danych obejmował:

- Usunięcie kolumn z dużą liczbą identycznych wartości ( > 95% = `UNIQUE_VALUE_THRESHOLD`)
- Zastąpienie wartości 0 na NaN w kolumnach, gdzie 0 występowało często ( > 75% = `ZERO_TO_NAN_THRESHOLD`)
- Podział na zbiór treningowy i testowy (0.3 = `TEST_SIZE`)
- Selekcję zmiennych za pomocą RFE (z liczbą cech ustawioną na 100 = MAX_FEATURES)
- Pipeline, w zależności od modelu:
  - Dla regresji logistycznej:
    - Zastąpienie wartości inf maksymalną wartością w kolumnie
    - SimpleImputer medianą lub najczęściej występującą wartością
    - One-Hot Encoding dla zmiennych kategorycznych, z `drop='first'` oraz `add_indicator` optymalizowanym podczas optymalizacji hiperparametrów
    - Clippowanie outlierów (`alpha`% najmniejszych i największych, gdzie `alpha` potraktowaliśmy jako hiperparametr do optymalizacji)
    - Skalowanie cech za pomocą StandardScaler
    - Wyeliminowanie parami skorelowanych cech od `corr_threshold` (także traktowany jako hiperparametr do optymalizacji)
  - Dla lasu losowego:
    - Zastąpienie wartości inf maksymalną wartością w kolumnie
    - SimpleImputer medianą lub najczęściej występującą wartością
    - One-Hot Encoding dla zmiennych kategorycznych

---

## Tuning / optymalizacja hiperparametrów

Po przetworzeniu danych, przeprowadziliśmy strojenie hiperparametrów dla obu modeli za pomocą optymalizacji bayesowskiej z wykorzystaniem pakietu AutoML "Optuna". Jako metrykę optymalizacji wybraliśmy ROC AUC.

## Kalibracja

## Wyjaśnialność modeli

## Dostosowanie progu decyzyjnego

## Mapowanie PD na ratingi

## Wnioski
