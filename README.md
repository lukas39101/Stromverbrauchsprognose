# Stromverbrauchsprognose-Projekt

Dieses Projekt befasst sich mit der Prognose des Stromverbrauchs und der Strompreise in verschiedenen europäischen Ländern mithilfe von maschinellem Lernen und Zeitreihenanalyse. Es umfasst die Datenvorverarbeitung, Modelltraining, Vorhersage und Visualisierung der Ergebnisse in einer interaktiven Webanwendung.

## Inhaltsverzeichnis

1. [Projektbeschreibung](#projektbeschreibung)
2. [Daten](#daten)
3. [Verwendete Technologien](#verwendete-technologien)
4. [Installation](#installation)
5. [Verwendung](#verwendung)
## Projektbeschreibung

Das Ziel dieses Projekts ist es, den Stromverbrauch und die Strompreise in verschiedenen europäischen Ländern vorherzusagen. Dies ist wichtig für die Planung und Optimierung der Stromerzeugung und -verteilung. Wir verwenden historische Daten, die von Open Power System Data bereitgestellt werden, um maschinelle Lernmodelle zu trainieren und zukünftige Werte vorherzusagen.

## Daten

Wir verwenden Stromverbrauchs- und Strompreisdaten von Open Power System Data für 32 europäische Länder. Diese Daten umfassen stündliche Messungen, die zu täglichen Durchschnittswerten aggregiert werden. Aufgrund der Größe der Daten werden sie nicht direkt in diesem Repository bereitgestellt. Sie können die Daten jedoch von [Open Power System Data](https://github.com/Open-Power-System-Data/time_series) herunterladen.

## Verwendete Technologien

- **Python**: Die Hauptprogrammiersprache für Datenverarbeitung und Modellierung.
- **Pandas**: Für die Datenmanipulation und -analyse.
- **Prophet**: Ein von Facebook entwickeltes Modell für Zeitreihenprognosen.
- **Scikit-learn**: Für die Berechnung von Evaluierungsmetriken.
- **Streamlit**: Für die Erstellung der interaktiven Webanwendung.
- **Plotly**: Für die interaktive Visualisierung der Daten.

## Installation

1. Klonen Sie das Repository:
   ```bash
   git clone https://github.com/lukas39101/Stromverbrauchsprognose.git
   cd Stromverbrauchsprognose
2. Erstellen und aktivieren Sie eine virtuelle Umgebung
   python -m venv venv
   source venv/bin/activate  # Auf Windows: venv\Scripts\activate
3. Installieren Sie die erforderlichen Pakete:
   pip install -r requirements.txt
4. Laden Sie die Datensätze von Open Power System Data herunter und legen Sie sie im Projektverzeichnis ab.

## Verwendung

1. Starten Sie die Streamlit-Anwendung:
   streamlit run app.py
2. Öffnen Sie Ihren Browser und gehen Sie zu http://localhost:8501, um die Webanwendung zu verwenden.

