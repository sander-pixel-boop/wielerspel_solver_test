import streamlit as st
import pandas as pd
import json
import os
import hashlib
from datetime import datetime
from thefuzz import process, fuzz
from supabase import create_client

# 1. Paginaconfiguratie
st.set_page_config(page_title="Custom Klassiekers Spel", layout="wide", page_icon="🎮")

# 2. Check Inlog
if "ingelogde_speler" not in st.session_state:
    st.warning("⚠️ Log eerst in op de Home pagina.")
    st.stop()

speler_naam = st.session_state["ingelogde_speler"]

# 3. Database Connectie
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()
tabel_naam = st.secrets["TABEL_NAAM"]

# --- HULPFUNCTIES ---
def normalize_name_logic(text):
    if not isinstance(text, str): return ""
    import unicodedata
    nfkd_form = unicodedata.normalize('NFKD', text.lower().strip())
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def generate_signature(data_dict):
    data_str = json.dumps(data_dict, sort_keys=True)
    salt = "GeheimeKlassiekerSleutel2026"
    return hashlib.sha256((data_str + salt).encode('utf-8')).hexdigest()

# --- DATA LADEN ---
@st.cache_data
def load_game_data():
    try:
        df_p = pd.read_csv("sporza_prijzen_startlijst.csv", sep=None, engine='python')
        df_s = pd.read_csv("renners_stats.csv", sep=None, engine='python')
        if 'Naam' in df_p.columns: df_p = df_p.rename(columns={'Naam': 'Renner'})
        if 'Naam' in df_s.columns: df_s = df_s.rename(columns={'Naam': 'Renner'})
        
        races = ["STR", "NOK", "BKC", "MSR", "RVB", "E3", "IFF", "DDV", "RVV", "SP", "PR", "RVL", "BRP", "AGT", "WAP", "LBL"]
        available = [r for r in races if r in df_p.columns]
        
        koers_map = {"NOK":"SPR","BKC":"SPR","MSR":"AVG","RVB":"SPR","E3":"COB","IFF":"SPR","DDV":"COB","RVV":"COB","SP":"SPR","PR":"COB","RVL":"SPR","BRP":"HLL","AGT":"HLL","WAP":"HLL","LBL":"HLL"}
        
        df = pd.merge(df_p, df_s[['Renner', 'COB', 'HLL', 'SPR', 'AVG', 'Team']], on='Renner', how='left')
        return df, available, koers_map
    except:
        return pd.DataFrame({'Renner': ['Wout van Aert', 'Mathieu van der Poel', 'Tadej Pogačar']}), ["OML", "KBK", "STR"], {}

df, races, k_map = load_game_data()
alle_renners = sorted(df['Renner'].dropna().unique()) if not df.empty else []

# --- STATE ---
if "game_base_team" not in st.session_state: st.session_state.game_base_team = []
if "game_transfers" not in st.session_state: 
    st.session_state.game_transfers = [{"uit": None, "in": None, "moment": None} for _ in range(5)]
    
# Zorg dat de lijst lokaal ALTIJD lengte 5 heeft om IndexErrors te voorkomen
while len(st.session_state.game_transfers) < 5:
    st.session_state.game_transfers.append({"uit": None, "in": None, "moment": None})

if "game_picks" not in st.session_state: st.session_state.game_picks = {r: {"extras": [], "joker": None} for r in races}

# --- UI ---
st.title(f"🎮 Custom Spel: {speler_naam.capitalize()}")

with st.sidebar:
    st.header("Opslag")
    if st.button("💾 Opslaan in Cloud", type="primary", use_container_width=True):
        data = {"base": st.session_state.game_base_team, "transfers": st.session_state.game_transfers, "picks": st.session_state.game_picks}
        payload = {"username": speler_naam, "custom_team": {"data": data, "signature": generate_signature(data)}}
        supabase.table(tabel_naam).upsert(payload, on_conflict="username").execute()
        st.success("Opgeslagen!")

    if st.button("🔄 Laden uit Cloud", use_container_width=True):
        res = supabase.table(tabel_naam).select("custom_team").eq("username", speler_naam).execute()
        if res.data:
            d = res.data[0]["custom_team"]["data"]
            st.session_state.game_base_team = d.get("base", [])
            
            # Voorkom IndexError bij inladen van lege/te korte lijst uit cloud
            geladen_transfers = d.get("transfers", [])
            while len(geladen_transfers) < 5:
                geladen_transfers.append({"uit": None, "in": None, "moment": None})
            st.session_state.game_transfers = geladen_transfers
            
            st.session_state.game_picks = d.get("picks", {r: {"extras": [], "joker": None} for r in races})
            st.rerun()

st.write("Stel hieronder je team samen, plan je transfers in en kies je kopmannen!")
st.divider()

# --- INTERFACE COMPONENTEN ---
tab1, tab2, tab3 = st.tabs(["🚴 Basis Team", "🔄 Transfers", "🏁 Per Koers"])

# Tab 1: Basis Team Selectie
with tab1:
    st.subheader("Selecteer je Basis Team (Max 20)")
    
    geselecteerd = st.multiselect(
        "Kies je renners:", 
        options=alle_renners, 
        default=st.session_state.game_base_team,
        max_selections=20
    )
    
    st.session_state.game_base_team = geselecteerd
    st.progress(len(geselecteerd) / 20)
    st.write(f"**{len(geselecteerd)} / 20** geselecteerd")

# Tab 2: Transfers
with tab2:
    st.subheader("Plan je Transfers (Max 5)")
    st.info("Kies welke renner eruit gaat, wie erin komt, en NA welke koers dit gebeurt.")
    
    for i in range(5):
        col1, col2, col3 = st.columns(3)
        t = st.session_state.game_transfers[i]
        
        with col1:
            uit_opties = [""] + st.session_state.game_base_team
            idx_uit = uit_opties.index(t["uit"]) if t["uit"] in uit_opties else 0
            gekozen_uit = st.selectbox(f"Wissel {i+1} UIT", options=uit_opties, index=idx_uit, key=f"uit_{i}")
            st.session_state.game_transfers[i]["uit"] = gekozen_uit if gekozen_uit else None
            
        with col2:
            in_opties = [""] + alle_renners
            idx_in = in_opties.index(t["in"]) if t["in"] in in_opties else 0
            gekozen_in = st.selectbox(f"Wissel {i+1} IN", options=in_opties, index=idx_in, key=f"in_{i}")
            st.session_state.game_transfers[i]["in"] = gekozen_in if gekozen_in else None
            
        with col3:
            moment_opties = [""] + races
            idx_moment = moment_opties.index(t["moment"]) if t["moment"] in moment_opties else 0
            gekozen_moment = st.selectbox(f"Na koers", options=moment_opties, index=idx_moment, key=f"mom_{i}")
            st.session_state.game_transfers[i]["moment"] = gekozen_moment if gekozen_moment else None

# Tab 3: Selecties per koers (Kopmannen)
with tab3:
    st.subheader("Kopman Selectie")
    koers_keuze = st.selectbox("Kies de koers om je selectie te bekijken:", races)
    
    if koers_keuze:
        actieve_team = list(st.session_state.game_base_team) 
        
        if actieve_team:
            huidige_kopman = st.session_state.game_picks[koers_keuze].get("joker")
            opties_kopman = [""] + actieve_team
            idx_kopman = opties_kopman.index(huidige_kopman) if huidige_kopman in opties_kopman else 0
            
            st.write(f"### Selectie voor {koers_keuze}")
            gekozen_kopman = st.selectbox("Kies je Kopman (dubbele punten):", options=opties_kopman, index=idx_kopman, key=f"kopman_{koers_keuze}")
            st.session_state.game_picks[koers_keuze]["joker"] = gekozen_kopman if gekozen_kopman else None
        else:
            st.warning("Kies eerst je basis team in het eerste tabblad!")
