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

def is_team_locked():
    if os.path.exists("uitslagen.csv"):
        try:
            df_u = pd.read_csv("uitslagen.csv", sep=None, engine='python')
            df_u.columns = [str(c).strip().title() for c in df_u.columns]
            if 'Race' in df_u.columns:
                verreden = [str(x).strip().upper() for x in df_u['Race'].unique()]
                if "NOK" in verreden:
                    return True
        except:
            return False
    return False

# --- DATA LADEN ---
@st.cache_data
def load_game_data():
    try:
        df_p = pd.read_csv("sporza_prijzen_startlijst.csv", sep=None, engine='python')
        df_s = pd.read_csv("renners_stats.csv", sep=None, engine='python')
        if 'Naam' in df_p.columns: df_p = df_p.rename(columns={'Naam': 'Renner'})
        if 'Naam' in df_s.columns: df_s = df_s.rename(columns={'Naam': 'Renner'})
        
        # Sporza koersen VANAF de koers NA Strade Bianche (STR = Strade, NOK = Nokere Koerse)
        races = ["NOK", "BKC", "MSR", "RVB", "E3", "IFF", "DDV", "RVV", "SP", "PR", "RVL", "BRP", "AGT", "WAP", "LBL"]
        available = [r for r in races if r in df_p.columns]
        
        koers_map = {"NOK":"SPR","BKC":"SPR","MSR":"AVG","RVB":"SPR","E3":"COB","IFF":"SPR","DDV":"COB","RVV":"COB","SP":"SPR","PR":"COB","RVL":"SPR","BRP":"HLL","AGT":"HLL","WAP":"HLL","LBL":"HLL"}
        
        df = pd.merge(df_p, df_s[['Renner', 'COB', 'HLL', 'SPR', 'AVG', 'Team']], on='Renner', how='left')
        
        # Zorg dat stats numeriek zijn voor de top 50 berekening
        for col in ['COB', 'HLL', 'SPR', 'AVG']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        return df, available, koers_map
    except:
        return pd.DataFrame({'Renner': ['Wout van Aert', 'Mathieu van der Poel', 'Tadej Pogačar']}), ["NOK", "MSR", "RVV"], {}

df, races, k_map = load_game_data()
alle_renners = sorted(df['Renner'].dropna().unique()) if not df.empty else []
team_locked = is_team_locked()

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

st.divider()

# --- INTERFACE COMPONENTEN ---
tab0, tab1, tab2 = st.tabs(["📖 Spelregels & Uitleg", "🚴 Basis Team (10)", "🏁 Selecties per Koers"])

# Tab 0: Spelregels
with tab0:
    st.header("📖 Spelregels")
    st.markdown("""
    Dit is een tactisch wielerspel waarbij je strategisch moet plannen voor de resterende voorjaarsklassiekers (vanaf Nokere Koerse).

    ### 1. Je Selectie
    * **Basis Team:** Je kiest éénmalig **10 vaste renners**. Deze renners zitten standaard in je team voor álle resterende koersen.
    * **Extra Renners:** Per koers mag je **3 extra renners** toevoegen aan je dagselectie. Je actieve team per koers bestaat dus altijd uit 13 renners.

    ### 2. Kopman & Dark Horse
    * **Kopman:** Per koers kies je **1 kopman** uit je 13 actieve renners. De punten van deze renner worden verdubbeld (x2).
    * **Dark Horse:** Het systeem bepaalt per koers een top 50 van favorieten (op basis van de benodigde specialiteit: sprint, kasseien of heuvels). Je kiest per koers **1 renner van buiten deze top 50**.
        * Eindigt jouw Dark Horse in de top 10 van de uitslag? Dan scoor je **150 bonuspunten**.

    ### 3. Puntentelling (Top 20)
    Voor elke renner in je dagselectie van 13 (inclusief je kopman) krijg je punten als ze in de top 20 eindigen:
    
    | Positie | Punten | Positie | Punten | Positie | Punten | Positie | Punten |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | **1e** | 100 | **6e** | 40 | **11e** | 20 | **16e** | 10 |
    | **2e** | 80 | **7e** | 36 | **12e** | 18 | **17e** | 8 |
    | **3e** | 70 | **8e** | 32 | **13e** | 16 | **18e** | 6 |
    | **4e** | 60 | **9e** | 28 | **14e** | 14 | **19e** | 4 |
    | **5e** | 50 | **10e**| 24 | **15e** | 12 | **20e** | 2 |
    
    *Let op: Bovenop deze basispunten komen eventuele vermenigvuldigers (kopman x2) en de Dark Horse bonus (+150 bij top 10).*
    
    ### 4. Kalender
    Het spel loopt over de volgende 15 koersen:
    Nokere Koerse (NOK), Bredene Koksijde (BKC), Milaan-Sanremo (MSR), Classic Brugge-De Panne (RVB), E3 Saxo Classic (E3), Gent-Wevelgem (IFF), Dwars door Vlaanderen (DDV), Ronde van Vlaanderen (RVV), Scheldeprijs (SP), Parijs-Roubaix (PR), Brabantse Pijl (RVL), Amstel Gold Race (BRP), Waalse Pijl (AGT), Luik-Bastenaken-Luik (LBL). *(Opmerking: WAP/AGT mapping kan variëren afhankelijk van je dataset).*
    """)

# Tab 1: Basis Team Selectie
with tab1:
    st.subheader("Selecteer je Basis Team (Max 10)")
    
    if team_locked:
        st.warning("🔒 De uitslag van Nokere Koerse (NOK) is verwerkt. Je basisteam is definitief en kan niet meer worden aangepast.")
        st.write("### Jouw vaste team:")
        for renner in sorted(st.session_state.game_base_team):
            st.write(f"- {renner}")
    else:
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
