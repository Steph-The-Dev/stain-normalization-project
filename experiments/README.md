# Experimente: Domain Shift auf Camelyon17-WILDS

Diese Notebooks messen, was ein fremdes Spital ein Modell kostet und was Stain Normalization (Macenko) und Farbaugmentation (HED) davon zurückholen. Sie sind die Grundlage des Vortrags «Domain Shift & Datenqualität: Wie Bildstandardisierung KI-Modelle in der Praxis robuster macht» (tfz Medical Cluster, 24. November 2026).

Es ist eine Nachmessung im Kleinen: ein öffentlicher Datensatz, ein kleines Standardnetz (ResNet-18), drei Durchläufe. Die Ergebnisse zeigen Trends, keine allgemeingültigen Effektgrössen. Notebooks und Kommentare sind auf Deutsch.

## Aufbau

| Notebook | Frage | Braucht |
| --- | --- | --- |
| `01_leave_one_center_out.ipynb` | Wie viel Genauigkeit kostet ein fremdes Spital? Was bringen Macenko, HED-Augmentation und beides? | Download (rund 10 GB), GPU empfohlen |
| `02_robustness_curve.ipynb` | Ab welcher künstlichen Farbverschiebung kippt das Modell? Reagiert es stärker auf Hämatoxylin oder Eosin? | Modelle aus Notebook 1 |
| `03_measure_shift.ipynb` | Wie verschieden sind die Spitäler in Farbe, Farbvektoren und Merkmalsraum, und verringert Normalisierung das? Ohne Diagnose-Labels. | Stain-Parameter aus Notebook 1 |

- Training auf Spital 0, 3 und 4, Test auf Spital 1 und 2 (fremd) sowie auf ungesehenen Slides der Trainingsspitäler (bekannt).
- Macenko-Parameter werden pro Slide aus 32 Ausschnitten geschätzt und brauchen keine Labels.
- `stainlib.py`: Macenko (Fit pro Slide, Anwendung pro Ausschnitt, Plausibilitätsprüfung), HED-Augmentation, kontrollierte Verschiebungen, Lab-Statistik, MMD, Spital-Erkennbarkeit. Nur NumPy, scikit-image, scikit-learn.
- `trainlib.py`: Datensatz, Split nach Spital und Slide, Training, Evaluation, Merkmalsextraktion (PyTorch).
- `extras/`: Diagnose-Skripte zu den verfälschten Farbvektoren (siehe unten).

## Einrichtung und Reihenfolge

```bash
cd experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

1. Notebook 1 ausführen. Es lädt den Datensatz nach `data/`, speichert Split und Stain-Parameter in `results/` und Modelle in `models/` (beides nicht im Repository).
2. Notebook 2 und 3 in beliebiger Reihenfolge.

Läufe werden in Notebook 1 über `LAUF` gewählt: `"standard"` (vortrainiert), `"scratch"` (von Grund auf), `"dark60"` und `"scratch60"` (dieselben mit Filter für dunkle Pixel, nur Macenko-Varianten) sowie `"tellez"` (von Grund auf, 10 Epochen, HED σ = 0,2). Jeder Lauf bekommt einen eigenen Namenszusatz; nichts wird überschrieben. Notebook 2 und 3 nutzen mit `MACENKO_DARK = 60` den korrigierten Lauf, mit `None` den ursprünglichen.

## Ergebnisse

Genauigkeit in Prozent, Mittel über drei Durchläufe. Macenko mit Filter für dunkle Pixel (Schwelle 60).

| Training | Variante | bekannte Spitäler | Spital 1 | Spital 2 | fremde Spitäler |
| --- | --- | --- | --- | --- | --- |
| von Grund auf, 5 Epochen | ohne Schutz | 95,0 | 80,8 | 67,4 | 74,1 |
| | HED-Augmentation | 92,2 | 84,4 | 80,4 | 82,4 |
| | Macenko | 95,1 | 81,6 | 91,7 | 86,6 |
| | Macenko + HED | 94,3 | 83,6 | 91,7 | 87,6 |
| ImageNet-vortrainiert, 3 Epochen | ohne Schutz | 85,7 | 88,3 | 89,0 | 88,7 |
| | HED-Augmentation | 93,2 | 90,4 | 91,1 | 90,8 |
| | Macenko | 91,9 | 87,4 | 95,3 | 91,4 |
| | Macenko + HED | 93,7 | 88,3 | 95,2 | 91,8 |

- Ohne Vortraining kostet ein fremdes Spital 21 Punkte. Macenko holt 12,5 davon zurück, fast alles in Spital 2, dem Farbausreisser.
- Mit Vortraining ist die Lücke weg. Macenko bringt im Mittel 2,7 Punkte: Spital 2 gewinnt 6, Spital 1 verliert knapp einen. In Spital 1 ist in beiden Bedingungen die Augmentation besser.
- Stresstest (Notebook 2, vortrainiert, ein Durchlauf): Bei künstlicher Farbverschiebung σ = 0,5 kippen 17,1 % der Entscheide ohne Schutz, 7,6 % mit HED-Augmentation und 1,9 % mit Macenko. Ab σ ≈ 0,3 liegt Macenko vor der Augmentation.
- Shift ohne Labels (Notebook 3): Der mittlere Farbabstand zweier Spitäler sinkt von ΔE 16,5 auf 5,9; im Farbton allein von 12,7 auf 2,9, der Rest ist Helligkeit. Das Spital bleibt aus Bildmerkmalen zu 79 % erkennbar (roh 85 %, Zufall 20 %).

Die vollständigen Zahlen stehen in `results/` (`nb1_all_conditions.csv` enthält alle Läufe, auch die ursprünglichen ohne Filter).

## Verfälschte Farbvektoren und Filter für dunkle Pixel

Im ursprünglichen Lauf waren die Macenko-Vektoren in 16 von 50 Slides verfälscht: Der Hämatoxylin-Vektor lag fast auf der Grauachse (Winkel unter 5°) oder die H-Intensität betrug genau 9,49, den Wert eines schwarzen Pixels. Ursache sind sehr dunkle, kaum bunte Pixel:

| Spital | Anteil Pixel mit hellstem Kanal < 60 | mittleres RGB | Buntheit |
| --- | --- | --- | --- |
| 0 | 0,5 % | 6 / 3 / 7 | 4,7 |
| 1 | 1,1 % | 37 / 23 / 35 | 15,2 |
| 2 | 0,08 % | 19 / 8 / 50 | 42,0 |
| 3 | 0,30 % | 45 / 16 / 52 | 36,2 |
| 4 | 0 % | – | – |

Dichte Zellkerne (Spital 2, 3) bleiben auch dunkel kräftig blauviolett und stören nicht. Fast graue Pixel liegen im Dichteraum nahe der Grauachse und ziehen den Hämatoxylin-Vektor dorthin. Was diese Pixel im Gewebe sind, wurde nicht angesehen.

Korrektur: `macenko_fit(..., dark_thresh=60)` ignoriert Pixel, deren hellster Kanal unter 60 liegt. Danach ist keine Slide mehr auffällig. Ein Filter bei 30 reichte nicht: Er erfasste Spital 1 nicht und senkte dessen Genauigkeit auf 79 %. Derselbe Filter steckt jetzt im `MacenkoNormalizer` der App (`src/normalizers/macenko.py`, Test in `tests/test_macenko_dark_pixels.py`).

## Grenzen

- Ein Datensatz aus fünf niederländischen Zentren mit ähnlichen Protokollen.
- Drei Durchläufe, zwei Testspitäler: Unterschiede von ein bis drei Punkten sind nicht belastbar.
- Angelehnt an Camelyon17-WILDS (Koh et al. 2021), keine Reproduktion: WILDS nutzt ein DenseNet-121 und testet nur auf Spital 2.
- Der Stresstest verschiebt Farbe künstlich; ob σ = 0,5 einem realen Laborunterschied entspricht, wurde nicht gemessen.
