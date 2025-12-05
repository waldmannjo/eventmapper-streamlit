# Schritt 4: Mapping
# Mapping der extrahierten Daten auf Standard-Codes.

import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import streamlit as st # Für Caching des Modells

from codes import CODES  # Importiert codes.py aus dem Hauptverzeichnis

# Konfiguration
EMB_MODEL = "text-embedding-3-large" # Konsistent bleiben
# LOW_CONF_THRESHOLD = 0.60
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HISTORY_FILE = "examples/CES_Umschlüsselungseinträge_all.xlsx"

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)

def embed_texts(client, texts, batch_size=500):
    """Erzeugt Embeddings für eine Liste von Texten in Batches."""
    if not texts:
        return np.array([])
    
    all_embeddings = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            # API call
            resp = client.embeddings.create(model=EMB_MODEL, input=batch)
            
            # Extract embeddings (preserve order)
            # resp.data is a list of embedding objects
            batch_embeddings = [e.embedding for e in resp.data]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"Embedding Error in batch {i}-{i+len(batch)}: {e}")
            # Wir werfen den Fehler weiter, da unvollständige Embeddings das Mapping zerschießen
            raise e
            
    return np.array(all_embeddings)

@st.cache_resource
def load_history_examples(_client):
    """Lädt historische Mappings und berechnet deren Embeddings einmalig."""
    if not os.path.exists(HISTORY_FILE):
        return None, None
    
    try:
        df_hist = pd.read_excel(HISTORY_FILE)
        # Benötigte Spalten prüfen
        req_cols = ['Description', 'AEB Event Code']
        if not all(c in df_hist.columns for c in req_cols):
            return None, None
            
        # Bereinigen
        df_hist = df_hist.dropna(subset=['Description', 'AEB Event Code'])
        
        if df_hist.empty:
             return None, None

        # Embeddings bauen
        # Wir nehmen 'Description' als Basis für die Ähnlichkeit
        hist_texts = df_hist['Description'].astype(str).tolist()
        
        # Batch-Embedding nutzen
        hist_vecs = embed_texts(_client, hist_texts, batch_size=500)
        
        return df_hist, hist_vecs
    except Exception as e:
        print(f"History Loading Error: {e}")
        return None, None

def get_similar_historical_entries(query_vec, df_hist, hist_vecs, top_k=3):
    """
    Findet die ähnlichsten historischen Beispiele für einen Query-Vektor.
    Gibt eine Liste von Dicts zurück: [{'input': str, 'mapped_code': str, 'score': float}, ...]
    """
    if df_hist is None or hist_vecs is None or len(hist_vecs) == 0:
        return []
    
    # Cosine Similarity berechnen
    # query_vec ist (dim,), hist_vecs ist (N, dim)
    sims = cosine_similarity(query_vec.reshape(1, -1), hist_vecs).ravel()
    
    # Top K Indizes
    top_indices = np.argsort(sims)[-top_k:][::-1]
    
    examples = []
    for idx in top_indices:
        row = df_hist.iloc[idx]
        examples.append({
            "input": row['Description'],
            "mapped_code": row['AEB Event Code'],
            "score": sims[idx]
        })
        
    return examples

def run_mapping_step4(client, df, model_name, threshold: float = 0.60, progress_callback=None):
    if df.empty: return df
    
    if progress_callback: progress_callback(0.05, "Lade Ressourcen & Embeddings...")

    # Ressourcen laden
    ce_model = load_cross_encoder()
    df_hist, hist_vecs = load_history_examples(client)

    # Textspalte finden
    if "Beschreibung" not in df.columns:
        df["Beschreibung"] = df.iloc[:, -1]

    # 1. Embed Standard Codes
    # Aufgabenspezifisches Präfix für Kontext
    code_texts = [f"Definition eines internen Sendungsstatus: {c[1]}. Details: {c[2]}" for c in CODES]
    code_vecs = embed_texts(client, code_texts)

    # 2. Embed Input (Enhanced Context)
    input_texts = []
    raw_input_texts_for_ce = [] # Texte ohne Präfix für Cross-Encoder
    
    for _, row in df.iterrows():
        parts = []
        if "Statuscode" in df.columns: parts.append(str(row["Statuscode"]))
        if "Reasoncode" in df.columns: parts.append(str(row["Reasoncode"]))
        parts.append(str(row["Beschreibung"]))
        
        combined_text = " ".join(parts)
        
        # Für Embedding (Bi-Encoder) mit Präfix
        input_texts.append(f"Beschreibung eines Sendungsstatus vom Transportdienstleister: {combined_text}")
        # Für Cross-Encoder nutzen wir den reinen Text + Kontext des Codes
        raw_input_texts_for_ce.append(combined_text)

    q_vecs = embed_texts(client, input_texts)
    
    pred_codes = []
    conf_scores = []
    sources = []
    top_candidates_list = [] # Speichert die Top-Kandidaten für den LLM Fallback
    
    # KNN THRESHOLD for direct match
    KNN_DIRECT_MATCH_THRESHOLD = 0.93

    # 3. Match (Priority: k-NN -> Bi-Encoder -> Cross-Encoder)
    total_items = len(q_vecs)
    for i, v in enumerate(q_vecs):
        if progress_callback:
            prog = 0.1 + (0.5 * (i / total_items))
            progress_callback(prog, f"Mapping Zeile {i+1}/{total_items}")
        
        # --- A. k-NN Check (History) ---
        knn_match_found = False
        if df_hist is not None and hist_vecs is not None:
            # Suche nur den Top-1 Nachbarn für direkten Match
            hist_matches = get_similar_historical_entries(v, df_hist, hist_vecs, top_k=1)
            if hist_matches:
                best_hist = hist_matches[0]
                if best_hist['score'] >= KNN_DIRECT_MATCH_THRESHOLD:
                    pred_codes.append(best_hist['mapped_code'])
                    conf_scores.append(float(best_hist['score']))
                    sources.append("history-knn")
                    top_candidates_list.append([]) # Keine Kandidaten für LLM nötig
                    knn_match_found = True
        
        if knn_match_found:
            continue

        # --- B. Standard Pipeline (Bi-Encoder + Cross-Encoder) ---
        
        # Bi-Encoder: Cosine Similarity gegen CODES
        sims = cosine_similarity(v.reshape(1,-1), code_vecs).ravel()
        
        # Vorfilterung: Top K
        top_k_prefilter = 10 
        top_k_idx = np.argsort(sims)[-top_k_prefilter:][::-1]
        
        # Cross-Encoder: Re-Ranking
        ce_pairs = []
        for idx in top_k_idx:
            code_desc = f"{CODES[idx][1]}. {CODES[idx][2]}"
            ce_pairs.append([raw_input_texts_for_ce[i], code_desc])
            
        ce_scores = ce_model.predict(ce_pairs)
        
        sorted_indices_in_top_k = np.argsort(ce_scores)[::-1] # Höchster Score zuerst
        
        reranked_indices = [top_k_idx[j] for j in sorted_indices_in_top_k]
        reranked_scores = [ce_scores[j] for j in sorted_indices_in_top_k]
        
        best_idx = reranked_indices[0]
        best_score = reranked_scores[0]
        
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        conf = sigmoid(best_score)
        
        pred_codes.append(CODES[best_idx][0])
        conf_scores.append(conf)
        sources.append("emb+ce") 
        
        # Kandidaten für LLM speichern (Top 3 nach Re-Ranking)
        candidates = []
        for j in range(min(3, len(reranked_indices))):
            r_idx = reranked_indices[j]
            candidates.append({
                "code": CODES[r_idx][0],
                "desc": CODES[r_idx][1],
                "score": float(sigmoid(reranked_scores[j]))
            })
        top_candidates_list.append(candidates)

    df["final_code"] = pred_codes
    df["confidence"] = conf_scores
    df["source"] = sources
    
import asyncio
from openai import AsyncOpenAI

# ... (imports remain the same, ensure asyncio and AsyncOpenAI are added if not present)

async def classify_single_row(async_client, row_text, candidates, hist_str, model_name, semaphore):
    async with semaphore:
        cand_str = "\n".join([f"- {c['code']} ({c['desc']})" for c in candidates])
        
        system_prompt = "Du bist ein Mapping-Experte. Wähle den passendsten Code aus den Vorschlägen. Nutze die historischen Beispiele als Orientierung für den Stil der Zuordnung."
        
        user_prompt = f"""
        Mappe diesen Input auf einen Standard-Code.
        
        Input: "{row_text}"
        
        Vorschläge (basierend auf Ähnlichkeit):
        {cand_str}
        {hist_str}
        
        Antworte im JSON-Format: {{ "code": "CODE", "reasoning": "Kurze Begründung" }}
        """
        
        try:
            resp = await async_client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            res = json.loads(resp.choices[0].message.content)
            return res.get("code")
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

async def run_llm_batch_async(api_key, tasks_data, model_name):
    """
    tasks_data: List of dicts { 'index': int, 'text': str, 'candidates': list, 'hist_str': str }
    """
    async_client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(15) # Max 15 concurrent requests
    
    tasks = []
    for item in tasks_data:
        task = classify_single_row(
            async_client, 
            item['text'], 
            item['candidates'], 
            item['hist_str'], 
            model_name, 
            semaphore
        )
        tasks.append(task)
        
    results = await asyncio.gather(*tasks)
    return results

def run_mapping_step4(client, df, model_name, threshold: float = 0.60, progress_callback=None):
    if df.empty: return df
    
    if progress_callback: progress_callback(0.05, "Lade Ressourcen & Embeddings...")

    # Ressourcen laden
    ce_model = load_cross_encoder()
    df_hist, hist_vecs = load_history_examples(client)

    # Textspalte finden
    if "Beschreibung" not in df.columns:
        df["Beschreibung"] = df.iloc[:, -1]

    # 1. Embed Standard Codes
    # Aufgabenspezifisches Präfix für Kontext
    code_texts = [f"Definition eines internen Sendungsstatus: {c[1]}. Details: {c[2]}" for c in CODES]
    code_vecs = embed_texts(client, code_texts)

    # 2. Embed Input (Enhanced Context)
    input_texts = []
    raw_input_texts_for_ce = [] # Texte ohne Präfix für Cross-Encoder
    
    for _, row in df.iterrows():
        parts = []
        if "Statuscode" in df.columns: parts.append(str(row["Statuscode"]))
        if "Reasoncode" in df.columns: parts.append(str(row["Reasoncode"]))
        parts.append(str(row["Beschreibung"]))
        
        combined_text = " ".join(parts)
        
        # Für Embedding (Bi-Encoder) mit Präfix
        input_texts.append(f"Beschreibung eines Sendungsstatus vom Transportdienstleister: {combined_text}")
        # Für Cross-Encoder nutzen wir den reinen Text + Kontext des Codes
        raw_input_texts_for_ce.append(combined_text)

    q_vecs = embed_texts(client, input_texts)
    
    pred_codes = []
    conf_scores = []
    sources = []
    top_candidates_list = [] # Speichert die Top-Kandidaten für den LLM Fallback
    
    # KNN THRESHOLD for direct match
    KNN_DIRECT_MATCH_THRESHOLD = 0.93

    # 3. Match (Priority: k-NN -> Bi-Encoder -> Cross-Encoder)
    total_items = len(q_vecs)
    for i, v in enumerate(q_vecs):
        if progress_callback:
            prog = 0.1 + (0.5 * (i / total_items))
            progress_callback(prog, f"Mapping Zeile {i+1}/{total_items}")
        
        # --- A. k-NN Check (History) ---
        knn_match_found = False
        if df_hist is not None and hist_vecs is not None:
            # Suche nur den Top-1 Nachbarn für direkten Match
            hist_matches = get_similar_historical_entries(v, df_hist, hist_vecs, top_k=1)
            if hist_matches:
                best_hist = hist_matches[0]
                if best_hist['score'] >= KNN_DIRECT_MATCH_THRESHOLD:
                    pred_codes.append(best_hist['mapped_code'])
                    conf_scores.append(float(best_hist['score']))
                    sources.append("history-knn")
                    top_candidates_list.append([]) # Keine Kandidaten für LLM nötig
                    knn_match_found = True
        
        if knn_match_found:
            continue

        # --- B. Standard Pipeline (Bi-Encoder + Cross-Encoder) ---
        
        # Bi-Encoder: Cosine Similarity gegen CODES
        sims = cosine_similarity(v.reshape(1,-1), code_vecs).ravel()
        
        # Vorfilterung: Top K
        top_k_prefilter = 10 
        top_k_idx = np.argsort(sims)[-top_k_prefilter:][::-1]
        
        # Cross-Encoder: Re-Ranking
        ce_pairs = []
        for idx in top_k_idx:
            code_desc = f"{CODES[idx][1]}. {CODES[idx][2]}"
            ce_pairs.append([raw_input_texts_for_ce[i], code_desc])
            
        ce_scores = ce_model.predict(ce_pairs)
        
        sorted_indices_in_top_k = np.argsort(ce_scores)[::-1] # Höchster Score zuerst
        
        reranked_indices = [top_k_idx[j] for j in sorted_indices_in_top_k]
        reranked_scores = [ce_scores[j] for j in sorted_indices_in_top_k]
        
        best_idx = reranked_indices[0]
        best_score = reranked_scores[0]
        
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        conf = sigmoid(best_score)
        
        pred_codes.append(CODES[best_idx][0])
        conf_scores.append(conf)
        sources.append("emb+ce") 
        
        # Kandidaten für LLM speichern (Top 3 nach Re-Ranking)
        candidates = []
        for j in range(min(3, len(reranked_indices))):
            r_idx = reranked_indices[j]
            candidates.append({
                "code": CODES[r_idx][0],
                "desc": CODES[r_idx][1],
                "score": float(sigmoid(reranked_scores[j]))
            })
        top_candidates_list.append(candidates)

    df["final_code"] = pred_codes
    df["confidence"] = conf_scores
    df["source"] = sources
    
    # 4. LLM Fallback (Async Batch Processing)
    unsure_mask = (df["confidence"] < threshold) & (df["source"] != "history-knn")
    unsure_indices = df[unsure_mask].index
    total_unsure = len(unsure_indices)

    if total_unsure > 0:
        if progress_callback: progress_callback(0.7, f"Starte LLM Batch-Verarbeitung für {total_unsure} Zeilen...")
        
        # Prepare data for batch processing
        tasks_data = []
        for i, idx in enumerate(unsure_indices):
            row_text = df.at[idx, 'Beschreibung']
            candidates = top_candidates_list[idx]
            
            # Few-Shot Context
            pos_idx = df.index.get_loc(idx)
            current_vec = q_vecs[pos_idx]
            hist_examples = get_similar_historical_entries(current_vec, df_hist, hist_vecs, top_k=3)
            
            hist_str = ""
            if hist_examples:
                hist_lines = [f"- Input '{ex['input']}' wurde gemappt auf '{ex['mapped_code']}'" for ex in hist_examples]
                hist_str = "\nHistorische Beispiele (zur Orientierung):\n" + "\n".join(hist_lines)
            
            tasks_data.append({
                'index': idx, # Store original DF index
                'text': row_text,
                'candidates': candidates,
                'hist_str': hist_str
            })
            
        # Run Async Batch
        try:
            # Extract API Key safely
            api_key = client.api_key
            results = asyncio.run(run_llm_batch_async(api_key, tasks_data, model_name))
            
            # Process Results
            for i, new_code in enumerate(results):
                idx = tasks_data[i]['index']
                if new_code and any(c[0] == new_code for c in CODES):
                    df.at[idx, "final_code"] = new_code
                    df.at[idx, "source"] = "llm-batch"
                    
        except Exception as e:
            print(f"Async Batch Error: {e}")
            # Fallback to nothing or log error
            pass
            
    
    if progress_callback: progress_callback(1.0, "Fertig!")
    return df