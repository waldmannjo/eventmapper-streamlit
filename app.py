import streamlit as st
import pandas as pd
from openai import OpenAI
import backend as logic  # <-- Das ist unser neues Modul

st.set_page_config(page_title="Eventmapper", layout="wide")
st.title("Eventmapper")

# --- Sidebar ---
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        st.warning("Bitte API Key eingeben.")
        st.stop()
        
    if st.button("🔄 Prozess zurücksetzen", help="Löscht alle gespeicherten Daten und setzt den Workflow auf Schritt 0 zurück."):
        st.session_state.clear()
        st.rerun()

# --- State Initialisierung ---
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "raw_text" not in st.session_state: st.session_state.raw_text = ""
if "analysis_res" not in st.session_state: st.session_state.analysis_res = {}
if "extraction_res" not in st.session_state: st.session_state.extraction_res = {}
if "df_merged" not in st.session_state: st.session_state.df_merged = pd.DataFrame()
if "df_final" not in st.session_state: st.session_state.df_final = pd.DataFrame()

# =========================================================
# SCHRITT 0: UPLOAD
# =========================================================
st.header("Schritt 0: Dokument Upload")
uploaded_file = st.file_uploader("Datei hochladen", type=["pdf", "xlsx", "csv", "txt"])

if uploaded_file and not st.session_state.raw_text:
    with st.spinner("Lese Datei ein..."):
        text = logic.extract_text_from_file(uploaded_file)
        st.session_state.raw_text = text
        st.success(f"Text extrahiert ({len(text)} Zeichen).")
        st.session_state.current_step = 0

if st.session_state.raw_text:
    if st.session_state.current_step == 0:
        if st.button("Weiter zu Schritt 1: Strukturanalyse starten"):
            with st.spinner("Analysiere Struktur..."):
                res = logic.analyze_structure_step1(client, st.session_state.raw_text)
                st.session_state.analysis_res = res
                st.session_state.current_step = 1
                st.rerun()

# =========================================================
# SCHRITT 1: ANALYSE ERGEBNIS
# =========================================================
if st.session_state.current_step >= 1:
    st.divider()
    st.header("Schritt 1: Ergebnis der Strukturanalyse")
    st.json(st.session_state.analysis_res, expanded=False)
    
    # Werte für Vorschlag extrahieren
    status_data = st.session_state.analysis_res.get("Statuscode", {})
    reason_data = st.session_state.analysis_res.get("Reasoncode", {})
    
    def_stat = status_data.get("Bezeichnung_im_Dokument", "Status") if isinstance(status_data, dict) else "Status"
    def_reas = reason_data.get("Bezeichnung_im_Dokument", "Reason") if isinstance(reason_data, dict) else "Reason"

    st.subheader("Konfiguration für Extraktion")
    col1, col2 = st.columns(2)
    with col1:
        st_term = st.text_input("Suchbegriff Statuscode", value=def_stat, key="term_stat")
    with col2:
        re_term = st.text_input("Suchbegriff Reasoncode", value=def_reas, key="term_reas")

    if st.session_state.current_step == 1:
        if st.button("Weiter zu Schritt 2: Daten extrahieren"):
            with st.spinner("Extrahiere Daten..."):
                ext_res = logic.extract_data_step2(client, st.session_state.raw_text, st_term, re_term)
                st.session_state.extraction_res = ext_res
                st.session_state.current_step = 2
                st.rerun()

# =========================================================
# SCHRITT 2: EXTRAKTION ZWISCHENERGEBNIS
# =========================================================
if st.session_state.current_step >= 2:
    st.divider()
    st.header("Schritt 2: Extrahierte Rohdaten")
    
    mode = st.session_state.extraction_res.get("mode", "unknown")
    st.info(f"Erkannter Modus: **{mode}**")
    
    if mode == "separate":
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Statuscodes (Vorschau)")
            df_s = logic.preview_csv_string(st.session_state.extraction_res.get("status_csv"))
            st.dataframe(df_s, height=200)
        with col_b:
            st.caption("Reasoncodes (Vorschau)")
            df_r = logic.preview_csv_string(st.session_state.extraction_res.get("reasons_csv"))
            st.dataframe(df_r, height=200)
    else:
        st.caption("Kombinierte Liste (Vorschau)")
        df_c = logic.preview_csv_string(st.session_state.extraction_res.get("combined_csv"))
        st.dataframe(df_c, height=200)

    if st.session_state.current_step == 2:
        if st.button("Weiter zu Schritt 3: Merge & Formatierung"):
            with st.spinner("Führe Merge durch..."):
                df_m = logic.merge_data_step3(st.session_state.extraction_res)
                st.session_state.df_merged = df_m
                st.session_state.current_step = 3
                st.rerun()

# =========================================================
# SCHRITT 3: MERGE ERGEBNIS
# =========================================================
if st.session_state.current_step >= 3:
    st.divider()
    st.header("Schritt 3: Merge Ergebnis")
    
    if st.session_state.df_merged.empty:
        st.error("Keine Daten nach Merge vorhanden. Bitte Prozess prüfen.")
    else:
        st.write(f"Anzahl Zeilen: {len(st.session_state.df_merged)}")
        st.dataframe(st.session_state.df_merged.head(), width="stretch")
        
        if st.session_state.current_step == 3:
            if st.button("Weiter zu Schritt 4: KI Mapping starten"):
                with st.spinner("Mappe Codes (Embedding + LLM)..."):
                    df_fin = logic.run_mapping_step4(client, st.session_state.df_merged)
                    st.session_state.df_final = df_fin
                    st.session_state.current_step = 4
                    st.rerun()

# =========================================================
# SCHRITT 4: FINALERGEBNIS
# =========================================================
if st.session_state.current_step >= 4:
    st.divider()
    st.header("✅ Schritt 4: Finales Mapping")
    
    st.dataframe(st.session_state.df_final, width="stretch")
    
    csv_data = st.session_state.df_final.to_csv(index=False, sep=";").encode('utf-8')
    st.download_button("💾 Mapping herunterladen", csv_data, "final_mapping.csv", "text/csv")