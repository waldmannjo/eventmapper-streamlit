# Schritt 4: Mapping
# Mapping der extrahierten Daten auf Standard-Codes.

import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from codes import CODES  # Importiert codes.py aus dem Hauptverzeichnis

# Konfiguration
# LLM_MODEL = "gpt-4o-mini"   # Stärkeres Modell für Analyse empfohlen    
EMB_MODEL = "text-embedding-3-small"
LOW_CONF_THRESHOLD = 0.60

def embed_texts(client, texts):
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return np.array([e.embedding for e in resp.data])

def run_mapping_step4(client, df, model_name: str = "gpt-4o-mini"):
    if df.empty: return df
    
    # Textspalte finden
    if "Beschreibung" not in df.columns:
        df["Beschreibung"] = df.iloc[:, -1]

    # 1. Embed Standard Codes
    code_texts = [f"{c[0]}: {c[1]}. {c[2]}" for c in CODES]
    code_vecs = embed_texts(client, code_texts)

    # 2. Embed Input
    input_texts = df["Beschreibung"].astype(str).tolist()
    q_vecs = embed_texts(client, input_texts)
    
    pred_codes = []
    conf_scores = []
    sources = []
    
    # 3. Match
    for v in q_vecs:
        sims = cosine_similarity(v.reshape(1,-1), code_vecs).ravel()
        top_idx = int(np.argmax(sims))
        top_val = sims[top_idx]
        second = np.partition(sims, -2)[-2] if len(sims) > 1 else 0.0
        conf = (top_val + (top_val - second)) / 2.0
        
        pred_codes.append(CODES[top_idx][0])
        conf_scores.append(conf)
        sources.append("emb")

    df["final_code"] = pred_codes
    df["confidence"] = conf_scores
    df["source"] = sources
    
    # 4. LLM Fallback
    unsure_mask = df["confidence"] < LOW_CONF_THRESHOLD
    if unsure_mask.any():
        code_summary = "\n".join([f"{c[0]} ({c[1]})" for c in CODES])
        for idx in df[unsure_mask].index:
            try:
                prompt = f"Mappe Text auf Code. JSON: {{'code': 'CODE'}}\nText: {df.at[idx, 'Beschreibung']}\nCodes:\n{code_summary}"
                resp = client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                res = json.loads(resp.choices[0].message.content)
                df.at[idx, "final_code"] = res.get("code")
                df.at[idx, "source"] = "llm"
            except: pass

    return df