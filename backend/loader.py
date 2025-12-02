import io
import pandas as pd

def extract_text_from_file(uploaded_file):
    """Liest Text aus PDF, XLSX, CSV, TXT."""
    filename = uploaded_file.name
    text = ""
    try:
        if filename.endswith('.pdf'):
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            text = df.to_string()
        elif filename.endswith('.csv') or filename.endswith('.txt'):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
    except Exception as e:
        return f"Fehler beim Lesen: {e}"
        
    return text[:60000] # Token Limit beachten