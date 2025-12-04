# Schritt 4: Mapping
# Mapping der extrahierten Daten auf Standard-Codes.

import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import streamlit as st # Für Caching des Modells

from codes import CODES  # Importiert codes.py aus dem Hauptverzeichnis

# Konfiguration
EMB_MODEL = "text-embedding-3-large"
# LOW_CONF_THRESHOLD = 0.60
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)

def embed_texts(client, texts):
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return np.array([e.embedding for e in resp.data])

def run_mapping_step4(client, df, model_name, threshold: float = 0.60):
    if df.empty: return df
    
    # Cross-Encoder laden (wird gecacht)
    ce_model = load_cross_encoder()

    # Textspalte finden
    if "Beschreibung" not in df.columns:
        df["Beschreibung"] = df.iloc[:, -1]

    # 1. Embed Standard Codes
    # Aufgabenspezifisches Präfix für Kontext
    code_texts = [f"Definition eines internen Sendungsstatus: {c[1]}. Details: {c[2]}" for c in CODES]
    code_vecs = embed_texts(client, code_texts)

    # 2. Embed Input (Enhanced Context)
    input_texts = []
    raw_input_texts_for_ce = [] # Texte ohne Präfix für Cross-Encoder (oder mit, je nach Modell-Präferenz, aber meist roh besser lesbar)
    
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
    
    # 3. Match (Bi-Encoder Vorfilterung + Cross-Encoder Re-Ranking)
    for i, v in enumerate(q_vecs):
        # A. Bi-Encoder: Cosine Similarity
        sims = cosine_similarity(v.reshape(1,-1), code_vecs).ravel()
        
        # Hole mehr Kandidaten für das Re-Ranking (z.B. Top 10 statt 3)
        # Wir wollen sichergehen, dass der "wahre" Match dabei ist
        top_k_prefilter = 10 
        top_k_idx = np.argsort(sims)[-top_k_prefilter:][::-1]
        
        # B. Cross-Encoder: Re-Ranking
        # Erstelle Paare: (Input-Text, Code-Beschreibung)
        # Wir nutzen hier eine Kombination aus Titel und Beschreibung für den Code
        ce_pairs = []
        for idx in top_k_idx:
            code_desc = f"{CODES[idx][1]}. {CODES[idx][2]}"
            ce_pairs.append([raw_input_texts_for_ce[i], code_desc])
            
        # Vorhersage (liefert Logits oder Scores, meist unbegrenzt, aber vergleichbar)
        ce_scores = ce_model.predict(ce_pairs)
        
        # Sortiere die top_k Indizes basierend auf den Cross-Encoder Scores neu
        # ce_scores ist parallel zu top_k_idx
        sorted_indices_in_top_k = np.argsort(ce_scores)[::-1] # Höchster Score zuerst
        
        # Die neuen Top-Indizes (gemappt zurück auf CODES)
        reranked_indices = [top_k_idx[j] for j in sorted_indices_in_top_k]
        reranked_scores = [ce_scores[j] for j in sorted_indices_in_top_k]
        
        # Bester Match nach Re-Ranking
        best_idx = reranked_indices[0]
        best_score = reranked_scores[0]
        
        # Sigmoid-ähnliche Normalisierung für Confidence (optional, da CE Scores logits sein können)
        # MS-Marco liefert oft Werte zwischen -10 und 10. Wir nutzen eine einfache Sigmoid für 0-1
        def sigmoid(x):
             return 1 / (1 + np.exp(-x))
        
        conf = sigmoid(best_score)
        
        pred_codes.append(CODES[best_idx][0])
        conf_scores.append(conf)
        sources.append("emb+ce") # Markierung als Embedding + Cross-Encoder
        
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