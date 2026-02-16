# Schritt 2: Extraktion
# Extrahiere Statuscodes und Reasoncodes aus dem Dokument.

# Konfiguration
# LLM_MODEL = "gpt-4o"  # Stärkeres Modell für Analyse empfohlen


import json
import io
import pandas as pd
from openai import OpenAI

def extract_data_step2(client, text: str, status_scope: list, reason_scope: list, model_name: str = "gpt-4o"):
    """
    Extrahiert Daten basierend auf dem Scope (User-Auswahl).
    Priorisiert existierende Kombinationen im Text.
    """
    
    # 1. Scopes für den Prompt formatieren
    # Wenn Listen leer sind, setzen wir einen Platzhalter, damit der Prompt nicht verwirrt ist.
    scope_text_status = ", ".join(status_scope) if status_scope else "Keine spezifische Auswahl (suche allgemein)"
    scope_text_reason = ", ".join(reason_scope) if reason_scope else "Keine (Code 'nicht vorhanden')"

    system_prompt = "Du bist ein Datenextraktions-Assistent. Antworte ausschließlich mit validem JSON."

    user_prompt = f"""
    # Aufgabe
    Extrahiere Statuscodes und Reasoncodes aus dem Dokument. Halte dich dabei STRENG an die vom User getroffene Auswahl der Quellen.

    # User-Auswahl (Scope)
    - Nutze für Statuscodes diese Quellen: {scope_text_status}
    - Nutze für Reasoncodes diese Quellen: {scope_text_reason}

    # ENTSCHEIDUNGSLOGIK (WICHTIG):
    Analysiere die Struktur der gewählten Quellen und entscheide den Modus:

    FALL A: KOMBINATION GEFUNDEN (Mode: "combined")
    - Wenn in den gewählten Quellen Statuscodes und Reasoncodes bereits fest verknüpft sind (z.B. eine Tabelle mit Spalten "Status" und "Reason", oder Codes wie "10-01" wobei 10 Status und 01 Reason ist).
    - Oder wenn der User für Status und Reason dieselbe Tabelle gewählt hat und diese beide Informationen enthält.
    -> DANN: Extrahiere diese exakte Kombination in 'combined_csv'.
    
    FALL B: GETRENNTE LISTEN (Mode: "separate")
    - Wenn die gewählten Quellen für Status und Reason strukturell voneinander getrennt sind (z.B. "Tabelle 8" nur Status, "Tabelle 12" nur Reasons).
    - Und keine logische Verknüpfung im Text besteht.
    -> DANN: Extrahiere Statuscodes nach 'status_csv' und Reasoncodes nach 'reasons_csv'.

    # Output JSON Format
    {{
      "mode": "combined" ODER "separate",
      "combined_csv": "Statuscode;Reasoncode;Beschreibung",  // Nur füllen wenn mode=combined
      "status_csv": "Statuscode;Beschreibung",               // Nur füllen wenn mode=separate
      "reasons_csv": "Reasoncode;Beschreibung"               // Nur füllen wenn mode=separate
    }}
    
    # Formatierungsregeln
    - Separator: Semikolon (;)
    - Header in den CSV-Strings inkludieren.
    - Wenn Spaltennamen im Text fehlen, benenne sie generisch (Code;Beschreibung).
    
    Dokument:
    {text}
    """

    response = client.responses.create(
        model=model_name,
        instructions=system_prompt,
        input=user_prompt,
        text={"format": {"type": "json_object"}}
    )
    return json.loads(response.output_text)

def preview_csv_string(csv_str):
    """Hilfsfunktion: Wandelt CSV-String in DataFrame für Preview um."""
    if not csv_str or len(csv_str) < 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(csv_str), sep=None, engine='python', on_bad_lines='skip')
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"CSV Preview Error: {e}")
        return pd.DataFrame()