# Schritt 3: Merge
# Merge der extrahierten Daten.

import pandas as pd
from .extractor import preview_csv_string

def merge_data_step3(extraction_result):
    mode = extraction_result.get("mode")
    
    if mode == "combined":
        csv_data = extraction_result.get("combined_csv", "")
        df = preview_csv_string(csv_data)
        if not df.empty:
            df.columns = [c.strip() for c in df.columns]
        return df

    else:
        status_csv = extraction_result.get("status_csv", "")
        reasons_csv = extraction_result.get("reasons_csv", "")
        
        df_status = preview_csv_string(status_csv)
        
        # Check ob Reason CSV existiert und valide ist
        has_reasons = reasons_csv and len(reasons_csv.strip()) > 10
        
        if has_reasons:
            df_reasons = preview_csv_string(reasons_csv)
            
            if not df_status.empty and not df_reasons.empty:
                # Cross Join
                df_status['key'] = 1
                df_reasons['key'] = 1
                df_combined = pd.merge(df_status, df_reasons, on='key').drop("key", axis=1)
                
                # Spalten umbenennen (Annahme: Code=1. Spalte, Desc=2. Spalte)
                cols = df_combined.columns.tolist()
                if len(cols) >= 4:
                    df_combined.columns = ["Statuscode", "StatusDesc", "Reasoncode", "ReasonDesc"]
                    df_combined["Beschreibung"] = df_combined["StatusDesc"].astype(str) + " - " + df_combined["ReasonDesc"].astype(str)
                    return df_combined[["Statuscode", "Reasoncode", "Beschreibung"]]
        
        # Fallback: Nur Status
        if not df_status.empty:
            # Wir nehmen an Spalte 0 ist Code, Spalte 1 ist Desc
            df_status.columns = ["Statuscode", "Beschreibung"]
            df_status["Reasoncode"] = ""
            return df_status
            
    return pd.DataFrame()