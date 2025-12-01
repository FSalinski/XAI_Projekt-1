# Model ratingowy do predykcji defaultu kredytowego

Projekt na przedmiot Interpretowalność i Wyjaśnialność Uczenia Maszynowego

---

## Opis projektu

Celem projektu jest stworzenie modelu ratingowego do predykcji defaultu kredytowego na podstawie zbioru danych dotyczących klientów banku (firm). W ramach projektu stworzyliśmy dwa modele: model regresji logistycznej (interpretowalny) oraz model lasu losowego (black box). Zastosowaliśmy techniki kalibracji oraz metody wyjaśnialności, przede wszystkim SHAP, aby lepiej zrozumieć działanie modeli.

---

## Wykorzystane technologie

- Python
- Biblioteki: pandas, numpy, scikit-learn, matplotlib, seaborn, shap, optuna

---

## Uruchomienie

Aby uruchomić projekt, należy posiadać środowisko Python z zainstalowanymi wymaganymi bibliotekami. Można to zrobić za pomocą pliku `requirements.txt`:

```bash
pip install -r requirements.txt
```

Następnie należy uruchomić skrypt `main.py`, który przeprowadzi cały proces od wczytania i przetwarzania danych, treningu modeli, kalibracji, aż po generowanie wizualizacji i wyjaśnień:

```bash
python main.py
```

Plik `constants.py` zawiera wszystkie stałe używane w projekcie i umożliwia łatwą konfigurację przed wywołaniem skryptu `main.py`.

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

- Usunięcie kolumn z dużą liczbą identycznych wartości ( > 95% = UNIQUE_VALUE_THRESHOLD)
- Zastąpienie wartości 0 na NaN w kolumnach, gdzie 0 występowało często ( > 75% = ZERO_TO_NAN_THRESHOLD)
- Podział na zbiór treningowy i testowy (0.3 = TEST_SIZE)
- Selekcję zmiennych za pomocą RFE (z liczbą cech ustawioną na 100 = MAX_FEATURES)
- Pipeline, który w zależności od modelu stosował:
  - Dla regresji logistycznej:
    - Zastępował wartości inf na maksymalną wartość w kolumnie
    - SimpleImputer medianą lub najczęściej występującą wartością
    - One-Hot Encoding dla zmiennych kategorycznych
    - Zastępowanie outlierów ($\alpha$% najmniejszych i największych, gdzie $\alpha$ potraktowaliśmy jako hiperparametr) na wartości graniczne
    - Skalowanie cech za pomocą StandardScaler
  - Dla lasu losowego:
    - Zastępował wartości inf na maksymalną wartość w kolumnie
    - SimpleImputer medianą lub najczęściej występującą wartością
    - One-Hot Encoding dla zmiennych kategorycznych

---

## Tuning hiperparametrów

Po przetworzeniu danych, przeprowadziliśmy strojenie hiperparametrów dla obu modeli za pomocą optymalizacji bayesowskiej z wykorzystaniem Optuny. Jako metrykę optymalizacji wybraliśmy ROC AUC.

...

## Kalibracja

W ramach projektu naszym zadaniem było również przeprowadzenie kalibracji modeli, do średniej PD równej 4%. Wykorzystaliśmy do tego kalibrację izotoniczną oraz sigmoid, dla obu modeli.

![Kalibracja LR](plots/lr_calibration_comparison.png)
![Kalibracja RF](plots/rf_calibration_comparison.png)

## Wyjaśnialność modeli

W celu lepszego wyjaśnienia działania modeli, narysowaliśmy wykresy SHAP dla obu modeli, zarówno globalne (mean absolute SHAP i beeswarm), jak i lokalne (waterfall dla kilku przykładowych obserwacji).

### Wyjaśnienia globalne dla regresji logistycznej

![Shap LR values](plots/shap/lr_shap_bar.png)
![Shap LR beeswarm](plots/shap/lr_shap_beeswarm.png)

### Wyjaśnienia globalne dla lasu losowego

![Shap RF values](plots/shap/rf_shap_bar.png)
![Shap RF beeswarm](plots/shap/rf_shap_beeswarm.png)

### Przykładowe wyjaśnienia lokalne

...

## Dostosowanie progu decyzyjnego

Ostatnim etapem tworzenia modelu było dostosowanie

## Mapowanie PD na ratingi

## Wnioski
