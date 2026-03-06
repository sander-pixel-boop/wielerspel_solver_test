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
        
        # Simpele merge voor test
        df = pd.merge(df_p, df_s[['Renner', 'COB', 'HLL', 'SPR', 'AVG', 'Team']], on='Renner', how='left')
        return df, available, koers_map
    except:
        return pd.DataFrame(), [], {}

df, races, k_map = load_game_data()

# --- STATE ---
if "game_base_team" not in st.session_state: st.session_state.game_base_team = []
if "game_transfers" not in st.session_state: st.session_state.game_transfers = []
if "game_picks" not in st.session_state: st.session_state.game_picks = {r: {"extras": [], "joker": None} for r in races}

# --- UI ---
st.title(f"🎮 Custom Spel: {speler_naam.capitalize()}")

with st.sidebar:
    if st.button("💾 Opslaan in Cloud", type="primary", use_container_width=True):
        data = {"base": st.session_state.game_base_team, "transfers": st.session_state.game_transfers, "picks": st.session_state.game_picks}
        payload = {"username": speler_naam, "custom_team": {"data": data, "signature": generate_signature(data)}}
        supabase.table(tabel_naam).upsert(payload, on_conflict="username").execute()
        st.success("Opgeslagen!")

    if st.button("🔄 Laden uit Cloud", use_container_width=True):
        res = supabase.table(tabel_naam).select("custom_team").eq("username", speler_naam).execute()
        if res.data:
            d = res.data[0]["custom_team"]["data"]
            st.session_state.game_base_team = d["base"]
            st.session_state.game_transfers = d["transfers"]
            st.session_state.game_picks = d["picks"]
            st.rerun()

st.write("Succes met je team!")
