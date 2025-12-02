import io
import pandas as pd

def extract_text_from_file(uploaded_file):
    """Liest Text aus PDF, XLSX, CSV, TXT. Liest bei Excel ALLE Sheets."""
    filename = uploaded_file.name
    text = ""
    
    try:
        if filename.endswith('.pdf'):
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
        elif filename.endswith('.xlsx'):
            # sheet_name=None liest ALLE Blätter in ein Dictionary
            # Key = Sheet-Name, Value = DataFrame
            dfs = pd.read_excel(uploaded_file, sheet_name=None)
            
            for sheet_name, df in dfs.items():
                # Wir fügen eine klare Markierung hinzu, damit die KI das Sheet erkennt
                text += f"\n--- TABELLENBLATT: {sheet_name} ---\n"
                text += df.to_string() + "\n"
                
        elif filename.endswith('.csv') or filename.endswith('.txt'):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
            
    except Exception as e:
        return f"Fehler beim Lesen: {e}"
        
    # Optional: Limit erhöhen, falls viele Sheets da sind
    return text[:100000]