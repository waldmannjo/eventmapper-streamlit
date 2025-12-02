# Schritt 3: Merge
# Merge der extrahierten Daten.

import pandas as pd
from .extractor import preview_csv_string

def merge_data_step3(extraction_result):
    mode = extraction_result.get("mode")
    
    # FALL A: Bereits kombinierte Liste (Combined Mode)
    if mode == "combined":
        csv_data = extraction_result.get("combined_csv", "")
        df = preview_csv_string(csv_data)
        if not df.empty:
            # Cleanup Spaltennamen
            df.columns = [c.strip() for c in df.columns]
            # Sicherheitscheck: Falls "Beschreibung" fehlt, letzte Spalte nehmen
            if "Beschreibung" not in df.columns:
                df["Beschreibung"] = df.iloc[:, -1]
        return df

    # FALL B: Getrennte Listen (Separate Mode)
    else:
        status_csv = extraction_result.get("status_csv", "")
        reasons_csv = extraction_result.get("reasons_csv", "")
        
        df_status = preview_csv_string(status_csv)
        df_reasons = preview_csv_string(reasons_csv)
        
        # --- INTELLIGENTER CHECK: Sind echte Reasoncodes vorhanden? ---
        has_real_reasons = False
        
        if not df_reasons.empty:
            # Wir prüfen die erste Zeile auf typische "Leer"-Phrasen der KI
            first_code = str(df_reasons.iloc[0, 0]).strip().lower()
            
            # Falls mehr als eine Spalte da ist, prüfen wir auch die Beschreibung
            first_desc = ""
            if df_reasons.shape[1] > 1:
                first_desc = str(df_reasons.iloc[0, 1]).strip().lower()
            
            invalid_keywords = ["nicht vorhanden", "keine", "none", "n/a", "not available", "no codes"]
            
            # Nur wenn KEIN invalid keyword gefunden wird, gelten die Reasons als echt
            is_dummy_code = any(k == first_code for k in invalid_keywords) # Exakter Match oder sehr nah
            is_dummy_desc = any(k in first_desc for k in invalid_keywords) # Substring Match
            
            if not is_dummy_code and not is_dummy_desc:
                has_real_reasons = True

        # --- MERGE LOGIK ---
        if has_real_reasons and not df_status.empty:
            # Cross Join (Jeder Status mit jedem Reason)
            df_status['key'] = 1
            df_reasons['key'] = 1
            df_combined = pd.merge(df_status, df_reasons, on='key').drop("key", axis=1)
            
            # Spalten normalisieren (Wir erwarten 4 relevante Spalten nach Merge)
            # Struktur ist jetzt: [StatusCol1, StatusCol2, ReasonCol1, ReasonCol2]
            cols = df_combined.columns.tolist()
            
            # Annahme: Spalte 0=Status, Spalte 1=StatusDesc, Spalte 2=Reason, Spalte 3=ReasonDesc
            # Wir benennen sie generisch um, um Fehler zu vermeiden
            if len(cols) >= 4:
                df_combined.columns = ["Statuscode", "StatusDesc", "Reasoncode", "ReasonDesc"] + cols[4:]
                
                # Beschreibung kombinieren: "StatusText - ReasonText"
                df_combined["Beschreibung"] = df_combined["StatusDesc"].astype(str) + " - " + df_combined["ReasonDesc"].astype(str)
                
                return df_combined[["Statuscode", "Reasoncode", "Beschreibung"]]
            
            # Fallback falls Spaltenstruktur unerwartet
            return df_combined

        # --- FALLBACK: NUR STATUS (Wenn keine echten Reasons da sind) ---
        elif not df_status.empty:
            # Wir erwarten: Spalte 0 = Code, Spalte 1 = Beschreibung
            # Falls nur 1 Spalte da ist, duplizieren wir sie
            if df_status.shape[1] == 1:
                df_status.columns = ["Statuscode"]
                df_status["Beschreibung"] = df_status["Statuscode"]
            else:
                # Wir nehmen die ersten zwei Spalten
                df_status = df_status.iloc[:, :2]
                df_status.columns = ["Statuscode", "Beschreibung"]
            
            # Reasoncode explizit leer lassen (NICHT "nicht vorhanden")
            df_status["Reasoncode"] = ""
            
            return df_status[["Statuscode", "Reasoncode", "Beschreibung"]]
            
    return pd.DataFrame()