# Schritt 4: Mapping
# Mapping der extrahierten Daten auf Standard-Codes.

import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from codes import CODES  # Importiert codes.py aus dem Hauptverzeichnis

# Konfiguration
EMB_MODEL = "text-embedding-3-large"
# LOW_CONF_THRESHOLD = 0.60

def embed_texts(client, texts):
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return np.array([e.embedding for e in resp.data])

def run_mapping_step4(client, df, model_name, threshold: float = 0.60):
    if df.empty: return df
    
    # Textspalte finden
    if "Beschreibung" not in df.columns:
        df["Beschreibung"] = df.iloc[:, -1]

    # 1. Embed Standard Codes
    code_texts = [f"Definition eines internen Sendungsstatus: {c[1]}. Details: {c[2]}" for c in CODES]
    code_vecs = embed_texts(client, code_texts)

    # 2. Embed Input (Enhanced Context)
    # Wir bauen einen reichhaltigeren String für das Embedding
    input_texts = []
    for _, row in df.iterrows():
        parts = []
        if "Statuscode" in df.columns: parts.append(str(row["Statuscode"]))
        if "Reasoncode" in df.columns: parts.append(str(row["Reasoncode"]))
        parts.append(str(row["Beschreibung"]))
        input_texts.append(f"Beschreibung eines Sendungsstatus vom Transportdienstleister: {' '.join(parts)}")

    q_vecs = embed_texts(client, input_texts)
    
    pred_codes = []
    conf_scores = []
    sources = []
    top_candidates_list = [] # Speichert die Top-3 Kandidaten für den LLM Fallback
    
    # 3. Match
    for v in q_vecs:
        sims = cosine_similarity(v.reshape(1,-1), code_vecs).ravel()
        
        # Top 3 Indizes holen
        top_3_idx = np.argsort(sims)[-3:][::-1]
        
        top_idx = top_3_idx[0]
        top_val = sims[top_idx]
        second_val = sims[top_3_idx[1]] if len(sims) > 1 else 0.0
        
        # Confidence Berechnung
        conf = (top_val + (top_val - second_val)) / 2.0
        
        pred_codes.append(CODES[top_idx][0])
        conf_scores.append(conf)
        sources.append("emb")
        
        # Kandidaten für LLM speichern (Code, ShortDesc, Score)
        candidates = []
        for idx in top_3_idx:
            candidates.append({
                "code": CODES[idx][0],
                "desc": CODES[idx][1],
                "score": float(sims[idx])
            })
        top_candidates_list.append(candidates)

    df["final_code"] = pred_codes
    df["confidence"] = conf_scores
    df["source"] = sources
    
    # 4. LLM Fallback
    unsure_mask = df["confidence"] < threshold
    if unsure_mask.any():
        # Wir geben dem LLM nur die Top-Kandidaten + eine Option "Other"
        # Das spart Tokens und fokussiert das Modell.
        
        for idx in df[unsure_mask].index:
            row_text = df.at[idx, 'Beschreibung']
            candidates = top_candidates_list[idx] # Liste von Dicts
            
            # Kandidaten-String bauen
            cand_str = "\n".join([f"- {c['code']} ({c['desc']})" for c in candidates])
            
            system_prompt = "Du bist ein Mapping-Experte. Wähle den passendsten Code aus den Vorschlägen oder entscheide dich für einen anderen, wenn keiner passt."
            
            user_prompt = f"""
            Mappe diesen Input auf einen Standard-Code.
            
            Input: "{row_text}"
            
            Vorschläge (basierend auf Ähnlichkeit):
            {cand_str}
            
            Antworte im JSON-Format: {{ "code": "CODE", "reasoning": "Kurze Begründung" }}
            """
            
            try:
                resp = client.chat.completions.create(
                    model=model_name, 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                res = json.loads(resp.choices[0].message.content)
                
                new_code = res.get("code")
                # Validierung: Ist der Code gültig?
                if any(c[0] == new_code for c in CODES):
                    df.at[idx, "final_code"] = new_code
                    df.at[idx, "source"] = "llm"
            except Exception as e:
                print(f"LLM Error: {e}")
                pass

    return df