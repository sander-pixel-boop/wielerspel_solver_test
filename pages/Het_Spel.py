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
        
        # Zorg dat stats numeriek zijn voor de top 50 berekening
        for col in ['COB', 'HLL', 'SPR', 'AVG']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        return df, available, koers_map
    except:
        return pd.DataFrame({'Renner': ['Wout van Aert', 'Mathieu van der Poel', 'Tadej Pogačar']}), ["OML", "KBK", "STR"], {}

df, races, k_map = load_game_data()
alle_renners = sorted(df['Renner'].dropna().unique()) if not df.empty else []

# --- STATE ---
if "game_base_team" not in st.session_state: 
    st.session_state.game_base_team = []
if "game_picks" not in st.session_state: 
    st.session_state.game_picks = {r: {"extras": [], "dark_horse": None, "kopman": None} for r in races}

# --- UI ---
st.title(f"🎮 Custom Spel: {speler_naam.capitalize()}")

with st.sidebar:
    st.header("Opslag")
    if st.button("💾 Opslaan in Cloud", type="primary", use_container_width=True):
        data = {"base": st.session_state.game_base_team, "picks": st.session_state.game_picks}
        payload = {"username": speler_naam, "custom_team": {"data": data, "signature": generate_signature(data)}}
        supabase.table(tabel_naam).upsert(payload, on_conflict="username").execute()
        st.success("Opgeslagen!")

    if st.button("🔄 Laden uit Cloud", use_container_width=True):
        res = supabase.table(tabel_naam).select("custom_team").eq("username", speler_naam).execute()
        if res.data:
            d = res.data[0]["custom_team"]["data"]
            st.session_state.game_base_team = d.get("base", [])
            st.session_state.game_picks = d.get("picks", {r: {"extras": [], "dark_horse": None, "kopman": None} for r in races})
            st.rerun()

st.write("Kies je 10 vaste renners en vul per koers je 3 extra renners, je dark horse en je kopman aan.")
st.divider()

# --- INTERFACE COMPONENTEN ---
tab1, tab2 = st.tabs(["🚴 Basis Team (10)", "🏁 Selecties per Koers"])

# Tab 1: Basis Team Selectie
with tab1:
    st.subheader("Selecteer je Basis Team (Max 10)")
    
    geselecteerd = st.multiselect(
        "Kies je 10 vaste renners:", 
        options=alle_renners, 
        default=st.session_state.game_base_team,
        max_selections=10
    )
    
    st.session_state.game_base_team = geselecteerd
    st.progress(len(geselecteerd) / 10 if len(geselecteerd) <= 10 else 1.0)
    st.write(f"**{len(geselecteerd)} / 10** geselecteerd")

# Tab 2: Selecties per koers
with tab2:
    st.subheader("Kopman, Extra's & Dark Horse")
    koers_keuze = st.selectbox("Kies een koers:", races)
    
    if koers_keuze:
        if koers_keuze not in st.session_state.game_picks:
            st.session_state.game_picks[koers_keuze] = {"extras": [], "dark_horse": None, "kopman": None}
            
        huidige_picks = st.session_state.game_picks[koers_keuze]
        
        # 1. Drie extra renners
        st.markdown(f"### 1. Drie Extra Renners voor {koers_keuze}")
        beschikbare_extras = [r for r in alle_renners if r not in st.session_state.game_base_team]
        
        gekozen_extras = st.multiselect(
            "Kies maximaal 3 extra renners (buiten je basisteam):",
            options=beschikbare_extras,
            default=[x for x in huidige_picks.get("extras", []) if x in beschikbare_extras],
            max_selections=3,
            key=f"extras_{koers_keuze}"
        )
        st.session_state.game_picks[koers_keuze]["extras"] = gekozen_extras

        # 2. Dark Horse (buiten top 50)
        st.markdown("### 2. Dark Horse")
        stat_voor_koers = k_map.get(koers_keuze, "AVG")
        
        if stat_voor_koers in df.columns:
            top_50_renners = df.sort_values(by=stat_voor_koers, ascending=False).head(50)['Renner'].tolist()
        else:
            top_50_renners = df.head(50)['Renner'].tolist()
            
        buiten_top_50 = [r for r in alle_renners if r not in top_50_renners]
        opties_dark_horse = [""] + buiten_top_50
        
        huidige_dark_horse = huidige_picks.get("dark_horse")
        idx_dh = opties_dark_horse.index(huidige_dark_horse) if huidige_dark_horse in opties_dark_horse else 0
        
        gekozen_dark_horse = st.selectbox(
            "Kies je renner buiten de top 50 (150 bonuspunten bij een top-10 klassering):",
            options=opties_dark_horse,
            index=idx_dh,
            key=f"dh_{koers_keuze}",
            help=f"Het systeem gebruikt de '{stat_voor_koers}' statistiek om de top 50 voor deze koers te bepalen."
        )
        st.session_state.game_picks[koers_keuze]["dark_horse"] = gekozen_dark_horse if gekozen_dark_horse else None
        
        # 3. Kopman
        st.markdown("### 3. Kopman")
        actieve_ploeg = st.session_state.game_base_team + gekozen_extras
        opties_kopman = [""] + actieve_ploeg
        
        huidige_kopman = huidige_picks.get("kopman")
        idx_kopman = opties_kopman.index(huidige_kopman) if huidige_kopman in opties_kopman else 0
        
        gekozen_kopman = st.selectbox(
            "Kies je Kopman (dubbele punten) uit je 13 actieve renners:",
            options=opties_kopman,
            index=idx_kopman,
            key=f"kopman_{koers_keuze}"
        )
        st.session_state.game_picks[koers_keuze]["kopman"] = gekozen_kopman if gekozen_kopman else None

        # Overzicht weergeven
        st.info("💡 **Jouw selectie voor deze koers:**\n" + 
                f"\n**Basis (10):** {', '.join(st.session_state.game_base_team) if st.session_state.game_base_team else 'Geen'}" +
                f"\n**Extra (3):** {', '.join(gekozen_extras) if gekozen_extras else 'Geen'}" +
                f"\n**Dark Horse:** {gekozen_dark_horse if gekozen_dark_horse else 'Geen'}" +
                f"\n**Kopman:** {gekozen_kopman if gekozen_kopman else 'Geen'}")
