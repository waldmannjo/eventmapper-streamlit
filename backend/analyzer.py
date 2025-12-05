# Schritt 1: Strukturanalyse
# Analyse des Dokuments auf Statuscodes und Reason Codes.

# Konfiguration
# LLM_MODEL = "gpt-4.1-2025-04-14" # "gpt-4o"  # Stärkeres Modell für Analyse empfohlen

import json
from backend.synonyms import (
    STATUS_SYNONYMS,
    REASON_SYNONYMS_CLASSIC,
    REASON_SYNONYMS_CONTEXT,
    REASON_SYNONYMS_COLUMNS
)

def analyze_structure_step1(client, text: str, model_name: str = "gpt-4o"):
    system_prompt = "Du bist ein Experte für Datenanalyse. Antworte ausschließlich mit validem JSON."
    
    # Synonym-Listen für den Prompt aufbereiten
    status_synonyms_str = ", ".join(STATUS_SYNONYMS)
    reason_classic_str = ", ".join(REASON_SYNONYMS_CLASSIC)
    reason_context_str = ", ".join(REASON_SYNONYMS_CONTEXT)
    reason_columns_str = ", ".join(REASON_SYNONYMS_COLUMNS)

    user_prompt = f"""
    # Aufgabe
    Analysiere das Dokument (PDF/XLSX) und identifiziere ALLE potenziellen Quellen für Statuscodes und Reason Codes.
    Es kann mehrere Tabellen, Sets, Code-Listen, Feldbeschreibungen oder Spalten geben, die Codes enthalten.

    # Ziel
    Erzeuge eine vollständige Kandidatenliste:
    - status_candidates: alle Stellen, an denen Statuscodes/Ereigniscodes/Scanarten/Events etc. definiert oder als Spalte geführt werden
    - reason_candidates: alle Stellen, an denen Zusatz-/Reason-/Qualifier-/Fehler-/Substatuscodes oder Zusatzinformationen geführt werden

    # Begriffe & Synonyme (WICHTIG)
    ## Statuscodes (Status)
    - {status_synonyms_str}
    ## Reason Codes (Reason/Zusatz)
    1) Klassisch:
    - {reason_classic_str}
    2) Im Kontext "Zusatzinformationen" (auch wenn nicht numerisch!):
    - {reason_context_str}
    3) Feld-/Spaltennamen, die oft Reason bedeuten:
    - {reason_columns_str}

    # Vorgehen (streng)
    1) Dokumentstruktur erkennen:
    - PDF: Kapitel/Abschnitte, Tabellen ("Tabelle 8"), Listen, "Set X", Überschriften, Feldbeschreibungen, Beispiele.
    - XLSX: Tabellenblätter, Tabellenbereiche, Spaltenüberschriften, ggf. Filter/Listen.
    2) Kandidaten sammeln (breit, lieber zu viel als zu wenig):
    - Status-Kandidaten, wenn irgendwo eine Liste/Mapping von Ereignissen/Scanarten/Statuscodes existiert ODER eine Spalte solche Codes enthält.
    - Reason-Kandidaten, wenn irgendwo Zusatz-/Reason-/Qualifier-Codes oder Zusatzinfos existieren ODER eine Spalte/Struktur Zusatz/Detail/Info als Qualifier zu einem Status enthält.
    3) Bei zusammengesetzten Codes:
    - Wenn ein Set/Statuscode + Zusatz/Qualifier gemeinsam eine Bedeutung trägt (z.B. "SE" + "Zusatz"), dann:
        - SE/Status -> status_candidates
        - Zusatz/Qualifier -> reason_candidates
    4) Beispiele & Felddefinitionen NICHT übersehen:
    - Wenn das Dokument Feldbeschreibungen enthält wie:
        - "Scan-Art", "Event", "Status", "Shipment Event", "LSP Statuscode" => Status
        - "Codes", "Zusatzcodes", "Zusatzcode", "Code", "codelist/code", "Zusatz", "Info", "Detail" => Reason
    - Auch wenn es keine eigene Code-Tabelle gibt: eine Spalte/Feld kann trotzdem Quelle sein.
    5) Ergebnis deduplizieren:
    - Mehrfachnennungen derselben Quelle (z.B. gleiche Tabelle einmal im Inhaltsverzeichnis, einmal im Kapitel) nur einmal listen.
    6) Wenn es wirklich keine Reason Codes gibt:
    - reason_candidates muss ein leeres Array [] sein. (Kein Text wie "keine".)

    # Output JSON Format (genau dieses Schema)
    {{
    "status_candidates": [
        {{ "id": "1", "name": "...", "description": "...", "context": "..." }}
    ],
    "reason_candidates": [
        {{ "id": "1", "name": "...", "description": "...", "context": "..." }}
    ]
    }}

    # Anforderungen an die Felder
    - id: fortlaufend als String ("1","2",...)
    - name:
    - PDF: "Tabelle 8 – DPD Scanarten", "Set 1: Auslieferung", "Kapitel 6.2 Datensatzaufbau" etc.
    - XLSX: "Sheet: Status Codes ASL / Spalte: EDIFACT Shipment Event"
    - context: präzise Fundstelle
    - PDF: "Seite X, Kapitel Y.Z, Überschrift ...", ggf. "Tabelle N"
    - XLSX: "Tabellenblatt <Name>, Spalte <Name>", optional Bereich falls erkennbar
    - description: 1–2 Sätze, warum dies Status- oder Reason-Quelle ist (z.B. "enthält Scanart-Codes 01..27 mit Beschreibung" / "enthält Zusatzcode-Liste (Qualifier) zur Scannung")

    # Qualitäts-Checks (vor Ausgabe kurz intern prüfen)
    - Habe ich sowohl tabellarische Code-Listen als auch Feld-/Spaltenquellen erfasst?
    - Habe ich Set-Strukturen (SE + Zusatz) korrekt auf Status vs. Reason aufgeteilt?
    - Sind reason_candidates wirklich [] wenn nichts gefunden wurde?

    Dokument:
    {text}
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)