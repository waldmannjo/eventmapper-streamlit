# Schritt 2: Extraktion
# Extrahiere Statuscodes und Reasoncodes aus dem Dokument.

import json
import io
import pandas as pd

# Konfiguration
LLM_MODEL = "gpt-4o"  # Stärkeres Modell für Analyse empfohlen

def extract_data_step2(client, text: str, status_term: str, reason_term: str):
    system_prompt = "Du bist ein Datenextraktions-Assistent. Antworte ausschließlich mit validem JSON."

    user_prompt = f"""
    # Aufgabe
    Extrahiere Statuscodes ("{status_term}") und Reasoncodes ("{reason_term}").
    
    # Logik
    1. Suche "{status_term}" und "{reason_term}".
    2. Sind sie kombiniert? -> Mode "combined", erstelle CSV (Statuscode;Reasoncode;Beschreibung).
    3. Sind sie getrennt? -> Mode "separate", erstelle zwei CSVs.
    
    # Output JSON
    {{
      "mode": "combined" ODER "separate",
      "combined_csv": "Statuscode;Reasoncode;Beschreibung",
      "status_csv": "Statuscode;Beschreibung",
      "reasons_csv": "Reasoncode;Beschreibung"
    }}
    
    Separator: Semikolon (;).
    
    Dokument:
    {text}
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def preview_csv_string(csv_str):
    """Hilfsfunktion: Wandelt CSV-String in DataFrame für Preview um."""
    if not csv_str or len(csv_str) < 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(csv_str), sep=";", on_bad_lines='skip')
    except:
        return pd.DataFrame()