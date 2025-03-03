# Bremen Big Data Challenge 2025

## Datendownload

Die Daten stehen zum Download bereit: (Dateigröße ca. 132 MB; entpackt ca. 369 MB).

## Aufgabenbeschreibung

Die Bremen Big Data Challenge (BBDC) 2025 konzentriert sich auf das Thema Geldwäschebekämpfung. Konkret geht es bei der Aufgabe darum, vorherzusagen, ob eine Person finanziellen Betrug begangen hat, basierend auf ihren Transaktions-Bankdaten für einen Monat. Aus Datenschutz- und Sicherheitsgründen wurde der Datensatz synthetisch erstellt, basierend auf vordefinierten Verteilungen von betrügerischen und nicht-betrügerischen Nutzern.

Teilnehmer der Wettbewerb erhalten transaktionsbezogene Daten für 11 Tausend einzigartige Konten, wobei etwa 15% der Nutzer Betrug (insbesondere Geldwäsche) begangen haben. Die Aufgabe besteht darin, ein Modell zu trainieren, das basierend auf den gelabelten Trainingsdaten vorhersagen kann, welche Nutzer im ungelabelten Testdatensatz Betrug begangen haben.

## Datendetails

Es gibt insgesamt 6 Dateien, deren Details im Folgenden beschrieben werden:

1. x_train.csv
2. y_train.csv
3. x_val.csv
4. y_val.csv
5. x_test.csv
6. professional_skeleton.csv

##

1. “x_train.csv”: Enthält alle für die BBDC 2025-Aufgabe erforderlichen Daten. Jede Zeile stellt eine einzelne Transaktion dar. Die Details der Daten sind unten beschrieben:
   
| Merkmal               | Beschreibung                                                                                                                                                                                                                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | 
| AccountID             | Eindeutige ID des Kontoinhabers. Vorhersagen müssen auf dieser Ebene erfolgen.                                                                                                                                                                                                                                         |
| Hour                  | Relativer Zeitbezug der Transaktion in Stunden, seit Beginn des Probemonats. Alle 24 Stunden repräsentieren einen neuen Tag.                                                                                                                                                                                           |
| Action                | Der Typ der durchgeführten Transaktion. Beschreibungen der einzelnen Transaktionstypen sind unten angegeben.                                                                                                                                                                                                           |
| External              | Eindeutige Kennung der Gegenpartei der Transaktion. Der erste Buchstabe beschreibt den Kontotyp. C-Codes zeigen an, dass die Transaktion mit einem anderen Kundenkonto interagiert hat, B-Codes bedeuten eine Interaktion mit der Bank selbst, und M-Codes zeigen eine Interaktion mit einem Händler oder Unternehmen. |
| Amount                | Der numerische Wert der Transaktion.                                                                                                                                                                                                                                                          |
| OldBalance            | Der Kontostand des Kontoinhabers vor der Transaktion.                                                                                                                                                                                                                                                                   |
| NewBalance            | Der Kontostand des Kontoinhabers nach der Transaktion.                                                                                                                                                                                                                                                                  |
| UnauthorizedOverdraft | Wenn „1“, wurde die Transaktion nicht verarbeitet, da nicht genügend Guthaben auf dem Konto vorhanden war, um die Transaktion durchzuführen.                                                                                                                                                                        |

| Action   | Beschreibung                                                                                  |
| -------- | --------------------------------------------------------------------------------------------- |
| CASH_IN   | Der Kontoinhaber zahlt Bargeld auf sein Konto ein.                                            |
| CASH_OUT  | Der Kontoinhaber hebt Bargeld von seinem Konto ab.                                            |
| DEBIT    | Geld wird über einen automatisierten Lastschriftauftrag vom Konto abgezogen. 
Diese Aktion wird über eine Bank (B) durchgeführt.                |
| PAYMENT  | Das Konto führt eine digitale Zahlung über die physische oder virtuelle Karte aus. Dies entspricht einem Kauf bei einem Händler (M), der den vom Kontoinhaber gezahlten Betrag erhält.         |
| TRANSFER | Der Kontoinhaber überweist Geld von seinem Konto auf ein anderes (C) mittels seiner Banking-App. |

2. “y_train.csv”: Diese Datei zeigt, welche Konten aus der Trainingsstichprobe finanziellen Betrug begangen haben oder nicht. Dieses binäre Flag ist das Zielfeld für das Training des Modells:
   
| Merkmal   | Beschreibung                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------ | --------- 
| AccountID | Eindeutige ID des Subjekts.                                                                                                    |
| Fraudster | Binärer Klassifikator. „1“ zeigt an, dass der Kontoinhaber Betrug begangen hat, „0“ zeigt an, dass er es nicht getan hat. |

3-4. *”Validierungssätze”: Zur Vereinfachung haben wir separate Trainings- und Validierungssätze erstellt, da die Trennung eines Teils der Daten komplex werden kann, wenn Interaktionsnetzwerke zwischen Transaktionspartnern und Drittparteien berücksichtigt werden. Für jeden der oben genannten Trainingssätze (1) und (2) gibt es einen entsprechenden Validierungssatz. Diese sollten zur Validierung und Feinabstimmung der trainierten Modelle verwendet werden.

5. “x_test.csv”: Die Testsätze liefern die Eingaben, die in die Pipeline des trainierten Modells für die Einreichung eingespeist werden sollen. Die bereitgestellten Felder und Beschreibungen entsprechen denen, die in (1) angegeben sind.

## Einreichung

Die Datei “professional_skeleton.csv” muss mit den vorhergesagten Betrugskennzeichen ausgefüllt werden. Diese Datei kann dann im BBDC 2025 Submission-Portal (https://bbdc.csl.uni-bremen.de/submission/) hochgeladen werden. Anschließend wird die Punktzahl automatisch berechnet und angezeigt, und die Rangliste wird aktualisiert. Die Anzahl der Zeilen und die Reihenfolge der `AccountID` dürfen nicht geändert werden.

## Bewertung

Die endgültige Punktzahl wird basierend auf f1-score berechnet. Je höher die Punktzahl, desto besser. Die Mindestpunktzahl beträgt 0,0 (0 %), und die Höchstpunktzahl beträgt 1,0 (100 %).
