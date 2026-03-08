import streamlit as st
import pandas as pd
import pulp
import json
import unicodedata
import os
from thefuzz import process, fuzz
from supabase import create_client
from datetime import datetime

# --- CONFIGURATIE ---
st.set_page_config(page_title="Sporza Giro AI", layout="wide", page_icon="🇮🇹")

if "ingelogde_speler" not in st.session_state:
    st.warning("⚠️ Je bent niet ingelogd. Ga terug naar de Home pagina om in te loggen.")
    st.stop()

speler_naam = st.session_state["ingelogde_speler"]

@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()
TABEL_NAAM = "gebruikers_data_test"
DB_KOLOM = "sporza_giro_team26"

# --- ETAPPE DATA ---
GIRO_ETAPPES = [
    {"id": 1, "date": "08/05", "route": "Nessebar - Burgas", "km": 156, "type": "Vlak ➖"},
    {"id": 2, "date": "09/05", "route": "Burgas - Valiko Tarnovo", "km": 220, "type": "Berg ⛰️"},
    {"id": 3, "date": "10/05", "route": "Plovdiv - Sofia", "km": 174, "type": "Heuvel ↗️"},
    {"id": 4, "date": "12/05", "route": "Catanzaro - Cosenza", "km": 144, "type": "Berg ⛰️"},
    {"id": 5, "date": "13/05", "route": "Praia a Mare - Potenza", "km": 204, "type": "Heuvel ↗️"},
    {"id": 6, "date": "14/05", "route": "Paestum - Naples", "km": 161, "type": "Heuvel ↗️"},
    {"id": 7, "date": "15/05", "route": "Formia - Blockhaus", "km": 246, "type": "Berg ⛰️"},
    {"id": 8, "date": "16/05", "route": "Chieti - Fermo", "km": 159, "type": "Heuvel ↗️"},
    {"id": 9, "date": "17/05", "route": "Cervia - Corno alle Scale", "km": 184, "type": "Berg ⛰️"},
    {"id": 10, "date": "19/05", "route": "Viareggio - Massa", "km": 40.2, "type": "Tijdrit ⏱️"},
    {"id": 11, "date": "20/05", "route": "Porcari - Chiavari", "km": 178, "type": "Heuvel ↗️"},
    {"id": 12, "date": "21/05", "route": "Imperia - Novi Ligure", "km": 177, "type": "Heuvel ↗️"},
    {"id": 13, "date": "22/05", "route": "Alessandria - Verbania", "km": 186, "type": "Heuvel ↗️"},
    {"id": 14, "date": "23/05", "route": "Aosta - Pila", "km": 133, "type": "Berg ⛰️"},
    {"id": 15, "date": "24/05", "route": "Voghera - Milan", "km": 136, "type": "Vlak ➖"},
    {"id": 16, "date": "26/05", "route": "Bellinzona - Carì", "km": 113, "type": "Berg ⛰️"},
    {"id": 17, "date": "27/05", "route": "Cassano d'Adda - Andalo", "km": 200, "type": "Heuvel ↗️"},
    {"id": 18, "date": "28/05", "route": "Fai della Paganella - Pieve di Soligo", "km": 167, "type": "Heuvel ↗️"},
    {"id": 19, "date": "29/05", "route": "Feltre - Alleghe", "km": 151, "type": "Berg ⛰️"},
    {"id": 20, "date": "30/05", "route": "Gemona del Friuli - Piancavallo", "km": 199, "type": "Berg ⛰️"},
    {"id": 21, "date": "31/05", "route": "Rome - Rome", "km": 131, "type": "Vlak ➖"},
]

# --- HULPFUNCTIES ---
def normalize_name_logic(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def match_naam_slim(naam, dict_met_namen):
    naam_norm = normalize_name_logic(naam)
    lijst_met_namen = list(dict_met_namen.keys())
    
    bekende_gevallen = {
        "philipsen": "jasper philipsen", "j. philipsen": "jasper philipsen", "j philipsen": "jasper philipsen",
        "pedersen": "mads pedersen", "m. pedersen": "mads pedersen", "m pedersen": "mads pedersen",
        "pidcock": "thomas pidcock", "t. pidcock": "thomas pidcock", "tom pidcock": "thomas pidcock",
        "van aert": "wout van aert", "w. van aert": "wout van aert", 
        "van der poel": "mathieu van der poel", "m. van der poel": "mathieu van der poel",
        "pogacar": "tadej pogacar", "t. pogacar": "tadej pogacar",
        "de lie": "arnaud de lie", "a. de lie": "arnaud de lie",
        "ganna": "filippo ganna", "thomas": "geraint thomas",
        "merlier": "tim merlier", "milan": "jonathan milan"
    }
    
    if naam_norm in bekende_gevallen:
        correct = bekende_gevallen[naam_norm]
        for target in lijst_met_namen:
            if correct in target:
                return dict_met_namen[target]
                    
    if naam_norm in lijst_met_namen:
        return dict_met_namen[naam_norm]
        
    bests = process.extractBests(naam_norm, lijst_met_namen, scorer=fuzz.token_set_ratio, limit=5)
    if bests and bests[0][1] >= 75:
        top_score = bests[0][1]
        candidates = [b[0] for b in bests if b[1] >= top_score - 3]
        candidates.sort(key=lambda x: (abs(len(x) - len(naam_norm)), -fuzz.ratio(naam_norm, x)))
        return dict_met_namen[candidates[0]]
        
    return naam

# --- DATA LADEN ---
@st.cache_data
def load_giro_data():
    prijzen_file = "sporza_prijzen_startlijst.csv"
    stats_file = "renners_stats.csv"
    
    if not os.path.exists(prijzen_file) or not os.path.exists(stats_file):
        st.warning(f"Bestanden '{prijzen_file}' of '{stats_file}' ontbreken. Zorg dat deze in de map staan.")
        return pd.DataFrame()

    try:
        df_prog = pd.read_csv(prijzen_file, sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
        df_stats = pd.read_csv(stats_file, sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip') 
        
        df_prog.columns = df_prog.columns.str.strip()
        df_stats.columns = df_stats.columns.str.strip()
        
        if 'Naam' in df_prog.columns: df_prog = df_prog.rename(columns={'Naam': 'Renner'})
        if 'Naam' in df_stats.columns: df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        if 'Ploeg' in df_stats.columns and 'Team' not in df_stats.columns: df_stats = df_stats.rename(columns={'Ploeg': 'Team'})
        
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        stats_names = df_stats['Renner'].unique()
        norm_to_stats = {normalize_name_logic(n): n for n in stats_names}
        
        df_prog['Renner_Stats'] = df_prog['Renner'].apply(lambda x: match_naam_slim(x, norm_to_stats))
        
        merged_df = pd.merge(df_prog, df_stats, left_on='Renner_Stats', right_on='Renner', how='left', suffixes=('', '_drop'))
        merged_df = merged_df.drop(columns=[c for c in merged_df.columns if '_drop' in c or c == 'Renner_Stats'])
        
        merged_df['Prijs'] = pd.to_numeric(merged_df['Prijs'], errors='coerce').fillna(0)
        merged_df.loc[merged_df['Prijs'] > 1000, 'Prijs'] = merged_df['Prijs'] / 1000000
        merged_df.loc[merged_df['Prijs'] == 0.8, 'Prijs'] = 0.75
        
        merged_df = merged_df[merged_df['Prijs'] > 0].sort_values(by='Prijs', ascending=False).drop_duplicates(subset=['Renner'])
        
        for col in ['GC', 'SPR', 'ITT', 'MTN']:
            if col not in merged_df.columns: merged_df[col] = 0
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
        
        merged_df['Team'] = merged_df.get('Team', pd.Series(['Onbekend']*len(merged_df))).fillna('Onbekend')
        
        return merged_df
    except Exception as e:
        st.error(f"Fout in dataverwerking: {e}")
        return pd.DataFrame()

def calculate_giro_ev(df):
    df = df.copy()
    df['EV_GC'] = (df['GC'] / 100)**4 * 400  
    df['EV_SPR'] = (df['SPR'] / 100)**4 * 250 
    df['EV_ITT'] = (df['ITT'] / 100)**4 * 80  
    df['EV_MTN'] = (df['MTN'] / 100)**4 * 100 

    df['Giro_EV'] = (df['EV_GC'] + df['EV_SPR'] + df['EV_ITT'] + df['EV_MTN']).fillna(0).round(0).astype(int)
    df['Waarde (EV/M)'] = (df['Giro_EV'] / df['Prijs']).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)
    
    def bepaal_rol(row):
        if row['GC'] >= 85: return 'Klassementsrenner'
        if row['SPR'] >= 85: return 'Sprinter'
        if row['ITT'] >= 85 and row['GC'] < 75: return 'Tijdrijder'
        if row['MTN'] >= 80 and row['GC'] < 80: return 'Aanvaller / Klimmer'
        return 'Knecht / Vrijbuiter'
        
    df['Type'] = df.apply(bepaal_rol, axis=1)
    return df

def calculate_prediction_ev(df, predictions, top_x):
    # Sporza etappepunten top 10
    pts_map = [50, 40, 30, 25, 20, 16, 14, 12, 10, 8]
    pred_series = pd.Series(0, index=df.index)
    
    for stage_id, preds in predictions.items():
        for pos in range(min(top_x, len(preds))):
            renner = preds[pos]
            if renner and renner != "-":
                idx = df[df['Renner'] == renner].index
                if not idx.empty:
                    pred_series.loc[idx[0]] += pts_map[pos]
    return pred_series

# --- SOLVER ---
def solve_giro_team(df, max_bud, max_ren, max_per_team, force_base, ban_base, ev_column):
    prob = pulp.LpProblem("Sporza_Giro_Solver", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    
    prob += pulp.lpSum([df.loc[i, ev_column] * x[i] for i in df.index])
    prob += pulp.lpSum([x[i] for i in df.index]) == max_ren
    prob += pulp.lpSum([df.loc[i, 'Prijs'] * x[i] for i in df.index]) <= max_bud
    
    teams = df['Team'].unique()
    for team in teams:
        team_indices = df[df['Team'] == team].index
        prob += pulp.lpSum([x[i] for i in team_indices]) <= max_per_team

    for i in df.index:
        renner = df.loc[i, 'Renner']
        if renner in force_base: prob += x[i] == 1
        if renner in ban_base: prob += x[i] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=15))
    if pulp.LpStatus[prob.status] == 'Optimal':
        return [df.loc[i, 'Renner'] for i in df.index if x[i].varValue > 0.5]
    return []

# --- HOOFDCODE ---
df_raw = load_giro_data()

if df_raw.empty:
    st.stop()

if "giro_selected_riders" not in st.session_state: 
    st.session_state.giro_selected_riders = []
if "giro_stage_predictions" not in st.session_state:
    st.session_state.giro_stage_predictions = {str(stage["id"]): [None]*10 for stage in GIRO_ETAPPES}

# Zorg ervoor dat oude opgeslagen lengtes naar 10 worden geüpdatet
for k in st.session_state.giro_stage_predictions:
    while len(st.session_state.giro_stage_predictions[k]) < 10:
        st.session_state.giro_stage_predictions[k].append(None)

with st.sidebar:
    st.header(f"👤 Profiel: {speler_naam.capitalize()}")
    
    st.write("☁️ **Cloud Database**")
    if speler_naam != "gast":
        c_cloud1, c_cloud2 = st.columns(2)
        with c_cloud1:
            if st.button("💾 Opslaan", type="primary", use_container_width=True):
                try:
                    data = {
                        "selected_riders": st.session_state.giro_selected_riders, 
                        "predictions": st.session_state.giro_stage_predictions,
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    supabase.table(TABEL_NAAM).update({DB_KOLOM: data}).eq("username", speler_naam).execute()
                    st.success("Opgeslagen!")
                except Exception as e: st.error(f"Fout: {e}")
        with c_cloud2:
            if st.button("🔄 Inladen", use_container_width=True):
                try:
                    res = supabase.table(TABEL_NAAM).select(DB_KOLOM).eq("username", speler_naam).execute()
                    if res.data and res.data[0].get(DB_KOLOM):
                        db_data = res.data[0][DB_KOLOM]
                        st.session_state.giro_selected_riders = db_data.get("selected_riders", [])
                        preds = db_data.get("predictions", {str(stage["id"]): [None]*10 for stage in GIRO_ETAPPES})
                        for k in preds:
                            while len(preds[k]) < 10: preds[k].append(None)
                        st.session_state.giro_stage_predictions = preds
                        st.success("Geladen!")
                        st.rerun()
                    else: st.warning("Geen team gevonden.")
                except Exception as e: st.error(f"Fout: {e}")
    else:
        st.info("Log in met een account om cloud-opslag te gebruiken.")
        
    st.divider()
    st.write("📁 **Lokale Backup (.json)**")
    save_data = {
        "selected_riders": st.session_state.giro_selected_riders,
        "predictions": st.session_state.giro_stage_predictions
    }
    st.download_button("📥 Download als .JSON", data=json.dumps(save_data), file_name=f"{speler_naam}_giro_team.json", mime="application/json", use_container_width=True)
    
    uploaded_file = st.file_uploader("📂 Upload Team (.json)", type="json")
    if uploaded_file is not None and st.button("Laad .json in", use_container_width=True):
        try:
            ld = json.load(uploaded_file)
            oude_selectie = ld.get("selected_riders", [])
            huidige_renners = df_raw['Renner'].tolist()
            
            def update_naam(naam):
                bests = process.extractBests(naam, huidige_renners, scorer=fuzz.token_set_ratio, limit=4)
                if bests and bests[0][1] > 75:
                    top_score = bests[0][1]
                    cands = [b[0] for b in bests if b[1] >= top_score - 2]
                    cands.sort(key=lambda x: (abs(len(x) - len(naam)), -fuzz.ratio(naam, x)))
                    return cands[0]
                return naam

            st.session_state.giro_selected_riders = [update_naam(r) for r in oude_selectie if update_naam(r) in huidige_renners]
            preds = ld.get("predictions", {str(stage["id"]): [None]*10 for stage in GIRO_ETAPPES})
            for k in preds:
                while len(preds[k]) < 10: preds[k].append(None)
            st.session_state.giro_stage_predictions = preds
            st.success("Lokaal bestand geladen!")
            st.rerun()
        except Exception as e:
            st.error("Fout bij inladen.")

    st.divider()
    st.markdown("### 🧠 Berekeningsmethode")
    bouw_methode = st.radio(
        "Hoe moet de AI het team samenstellen?",
        ["1. Volledig AI (Op basis van Stats)", "2. Mijn Voorspellingen (+ AI opvulling)"],
        help="Bij optie 2 forceert de solver de renners die jij in de 'Etappe Voorspellingen' tab hebt gezet. De overgebleven plekken/budget worden logisch aangevuld door de AI."
    )
    top_x_voorspellingen = st.number_input("Top X Voorspellen per etappe", min_value=1, max_value=10, value=5)
    
    st.divider()
    st.markdown("### ⚙️ Spelregels & Limieten")
    max_budget = st.number_input("Budget (Miljoen)", value=100.0, step=1.0)
    max_renners = st.number_input("Aantal Renners", value=16, step=1)
    max_per_ploeg = st.number_input("Max per ploeg", value=3, min_value=1)
    
    df = calculate_giro_ev(df_raw)
    df['Prediction_EV'] = calculate_prediction_ev(df, st.session_state.giro_stage_predictions, top_x_voorspellingen)
    
    # HYBRIDE SCORE
    df['Combined_EV'] = (df['Prediction_EV'] * 1000) + df['Giro_EV']

    with st.expander("🔒 Forceren / Uitsluiten", expanded=False):
        force_base = st.multiselect("🟢 Moet in team:", options=df['Renner'].tolist())
        ban_base = st.multiselect("🔴 Niet in team:", options=[r for r in df['Renner'].tolist() if r not in force_base])

    st.write("")
    if st.button("🚀 BEREKEN GIRO TEAM", type="primary", use_container_width=True):
        actieve_ev_col = "Giro_EV"
        if "2." in bouw_methode:
            actieve_ev_col = "Combined_EV"
                
        res = solve_giro_team(df, max_budget, max_renners, max_per_ploeg, force_base, ban_base, actieve_ev_col)
        
        if res:
            st.session_state.giro_selected_riders = res
            st.rerun()
        else:
            st.error("Geen oplossing mogelijk binnen dit budget en deze restricties.")

st.title("🇮🇹 Grote Ronde: Sporza Giromanager")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Jouw Selectie", "📅 Etappe Voorspellingen", "📋 Database (Giro)", "ℹ️ Uitleg Giromanager"])

with tab1:
    if not st.session_state.giro_selected_riders:
        st.info("👈 Kies je parameters in de zijbalk en klik op **Bereken Giro Team**.")
    else:
        st.subheader("📊 Dashboard")
        start_team_df = df[df['Renner'].isin(st.session_state.giro_selected_riders)].copy()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 Budget over", f"€ {max_budget - start_team_df['Prijs'].sum():.2f}M")
        m2.metric("🚴 Renners", f"{len(start_team_df)} / {max_renners}")
        m3.metric("🎯 EV (AI model)", f"{start_team_df['Giro_EV'].sum()}")
        m4.metric("🏆 EV (Jouw Voorspellingen)", f"{start_team_df['Prediction_EV'].sum()} pt")
        st.divider()
        
        toon_kolommen = ['Renner', 'Team', 'Type', 'Prijs', 'GC', 'SPR', 'ITT', 'MTN', 'Giro_EV']
        if "2." in bouw_methode: toon_kolommen.append('Prediction_EV')
        
        st.dataframe(start_team_df[toon_kolommen].sort_values(by='Prijs', ascending=False), hide_index=True, use_container_width=True)

with tab2:
    st.subheader(f"🏆 Voorspel de Top {top_x_voorspellingen} per Etappe")
    st.write("De AI gebruikt deze voorspellingen (indien je Methode 2 kiest in de zijbalk) als absolute prioriteit bij het bouwen van je team.")
    st.markdown("**Punten top 10:** 50, 40, 30, 25, 20, 16, 14, 12, 10, 8")
    
    renners_opties = ["-"] + sorted(df['Renner'].tolist())
    
    for etappe in GIRO_ETAPPES:
        stage_id_str = str(etappe["id"])
        
        with st.expander(f"Etappe {etappe['id']}: {etappe['route']} ({etappe['type']} | {etappe['km']} km) - {etappe['date']}", expanded=False):
            # Maak blokken van maximaal 5 kolommen per rij voor netheid
            for i in range(0, top_x_voorspellingen, 5):
                chunk_size = min(5, top_x_voorspellingen - i)
                cols = st.columns(chunk_size)
                
                for j in range(chunk_size):
                    pos = i + j
                    huidige_keuze = st.session_state.giro_stage_predictions[stage_id_str][pos]
                    index_keuze = renners_opties.index(huidige_keuze) if huidige_keuze in renners_opties else 0
                    
                    with cols[j]:
                        nieuwe_keuze = st.selectbox(
                            f"Positie {pos+1}", 
                            options=renners_opties, 
                            index=index_keuze, 
                            key=f"stage_{stage_id_str}_pos_{pos}"
                        )
                        st.session_state.giro_stage_predictions[stage_id_str][pos] = nieuwe_keuze if nieuwe_keuze != "-" else None

with tab3:
    st.subheader("Alle Renners")
    col_f1, col_f2 = st.columns(2)
    with col_f1: search_name = st.text_input("🔍 Zoek op naam of Ploeg:")
    with col_f2: type_filter = st.multiselect("Rol:", options=df['Type'].unique())
    
    d_df = df.copy()
    if search_name: d_df = d_df[d_df['Renner'].str.contains(search_name, case=False, na=False) | d_df['Team'].str.contains(search_name, case=False, na=False)]
    if type_filter: d_df = d_df[d_df['Type'].isin(type_filter)]
    
    st.dataframe(d_df[['Renner', 'Team', 'Type', 'Prijs', 'GC', 'SPR', 'ITT', 'MTN', 'Giro_EV', 'Prediction_EV']].sort_values('Giro_EV', ascending=False), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("""
    ### Twee manieren om een team te bouwen
    Deze solver geeft je de ultieme flexibiliteit door het algoritme op twee manieren in te zetten:
    
    **1. Volledig AI (Automatisch)**
    De wiskundige solver berekent de verwachte waarde (EV) van elke renner aan de hand van zijn statistieken in relatie tot het totale parcours (21 etappes). Hij maximaliseert deze waarde zonder over je budget (100M) of limieten (max 3 per ploeg) te gaan.
    
    **2. Mijn Voorspellingen (+ AI opvulling)**
    Jij vult in de tab 'Etappe Voorspellingen' jouw klassement in per rit. De tool berekent exact hoeveel Sporza-etappepunten dit elke renner oplevert. De wiskundige solver geeft absolute prioriteit aan het kopen van jouw voorspelde renners. Het overgebleven budget en de overgebleven plekken worden vervolgens optimaal opgevuld door de basis-AI.
    """)
