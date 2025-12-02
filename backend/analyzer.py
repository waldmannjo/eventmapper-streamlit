# Schritt 1: Strukturanalyse
# Analyse des Dokuments auf Statuscodes und Reason Codes.

import json


# Konfiguration
LLM_MODEL = "gpt-4o"  # Stärkeres Modell für Analyse empfohlen

def analyze_structure_step1(client, text: str):
    system_prompt = "Du bist ein Experte für Datenanalyse. Antworte ausschließlich mit validem JSON."
    
    user_prompt = f"""
    # Aufgabe:
    Analysiere das Dokument auf Statuscodes und Reason Codes.

    # Begriffe
    - Statuscodes: Carrier Event code, Event, Scanart, Status, Main code.
    - Reason Codes: additional code, Zusatzcode, reason, error code.

    # Vorgehen
    1. Identifiziere Tabellen/Felder für diese Codes.
    2. Beschreibe Bezeichnung, Stelle, Format.
    3. Wenn keine Reason Codes da sind: "nicht vorhanden".

    # Output JSON Format
    {{
      "Statuscode": {{ "Bezeichnung_im_Dokument": "...", "Stelle": "...", "Beispiel": "..." }},
      "Reasoncode": {{ ... }} oder "nicht vorhanden"
    }}

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