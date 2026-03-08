import streamlit as st
import pandas as pd
import pulp
import json
import unicodedata
import os
import base64
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

# --- ETAPPE DATA (Standaard wegingen obv profielzwaarte) ---
GIRO_ETAPPES = [
    {"id": 1, "date": "08/05", "route": "Nessebar - Burgas", "km": 156, "type": "Vlak ➖", "w": {"SPR": 1.0, "GC": 0.0, "ITT": 0.0, "MTN": 0.0}},
    {"id": 2, "date": "09/05", "route": "Burgas - Valiko Tarnovo", "km": 220, "type": "Heuvel ↗️", "w": {"SPR": 0.3, "GC": 0.3, "ITT": 0.0, "MTN": 0.4}},
    {"id": 3, "date": "10/05", "route": "Plovdiv - Sofia", "km": 174, "type": "Vlak/Heuvel", "w": {"SPR": 0.9, "GC": 0.0, "ITT": 0.0, "MTN": 0.1}},
    {"id": 4, "date": "12/05", "route": "Catanzaro - Cosenza", "km": 144, "type": "Vlak/Heuvel", "w": {"SPR": 0.6, "GC": 0.0, "ITT": 0.0, "MTN": 0.4}},
    {"id": 5, "date": "13/05", "route": "Praia a Mare - Potenza", "km": 204, "type": "Heuvel ↗️", "w": {"SPR": 0.1, "GC": 0.6, "ITT": 0.0, "MTN": 0.3}},
    {"id": 6, "date": "14/05", "route": "Paestum - Naples", "km": 161, "type": "Heuvel ↗️", "w": {"SPR": 0.8, "GC": 0.0, "ITT": 0.0, "MTN": 0.2}},
    {"id": 7, "date": "15/05", "route": "Formia - Blockhaus", "km": 246, "type": "Berg ⛰️", "w": {"SPR": 0.0, "GC": 0.9, "ITT": 0.0, "MTN": 0.1}},
    {"id": 8, "date": "16/05", "route": "Chieti - Fermo", "km": 159, "type": "Heuvel ↗️", "w": {"SPR": 0.2, "GC": 0.4, "ITT": 0.0, "MTN": 0.4}},
    {"id": 9, "date": "17/05", "route": "Cervia - Corno alle Scale", "km": 184, "type": "Berg ⛰️", "w": {"SPR": 0.0, "GC": 0.8, "ITT": 0.0, "MTN": 0.2}},
    {"id": 10, "date": "19/05", "route": "Viareggio - Massa", "km": 40.2, "type": "Tijdrit ⏱️", "w": {"SPR": 0.0, "GC": 0.0, "ITT": 1.0, "MTN": 0.0}},
    {"id": 11, "date": "20/05", "route": "Porcari - Chiavari", "km": 178, "type": "Heuvel ↗️", "w": {"SPR": 0.2, "GC": 0.4, "ITT": 0.0, "MTN": 0.4}},
    {"id": 12, "date": "21/05", "route": "Imperia - Novi Ligure", "km": 177, "type": "Vlak ➖", "w": {"SPR": 0.6, "GC": 0.0, "ITT": 0.0, "MTN": 0.4}},
    {"id": 13, "date": "22/05", "route": "Alessandria - Verbania", "km": 186, "type": "Heuvel ↗️", "w": {"SPR": 0.6, "GC": 0.0, "ITT": 0.0, "MTN": 0.4}},
    {"id": 14, "date": "23/05", "route": "Aosta - Pila", "km": 133, "type": "Berg ⛰️", "w": {"SPR": 0.0, "GC": 0.9, "ITT": 0.0, "MTN": 0.1}},
    {"id": 15, "date": "24/05", "route": "Voghera - Milan", "km": 136, "type": "Vlak ➖", "w": {"SPR": 1.0, "GC": 0.0, "ITT": 0.0, "MTN": 0.0}},
    {"id": 16, "date": "26/05", "route": "Bellinzona - Carì", "km": 113, "type": "Berg ⛰️", "w": {"SPR": 0.0, "GC": 0.9, "ITT": 0.0, "MTN": 0.1}},
    {"id": 17, "date": "27/05", "route": "Cassano d'Adda - Andalo", "km": 200, "type": "Heuvel ↗️", "w": {"SPR": 0.1, "GC": 0.5, "ITT": 0.0, "MTN": 0.4}},
    {"id": 18, "date": "28/05", "route": "Fai della Paganella - Pieve di Soligo", "km": 167, "type": "Heuvel ↗️", "w": {"SPR": 0.3, "GC": 0.2, "ITT": 0.0, "MTN": 0.5}},
    {"id": 19, "date": "29/05", "route": "Feltre - Alleghe", "km": 151, "type": "Berg ⛰️", "w": {"SPR": 0.0, "GC": 0.9, "ITT": 0.0, "MTN": 0.1}},
    {"id": 20, "date": "30/05", "route": "Gemona del Friuli - Piancavallo", "km": 199, "type": "Berg ⛰️", "w": {"SPR": 0.0, "GC": 0.9, "ITT": 0.0, "MTN": 0.1}},
    {"id": 21, "date": "31/05", "route": "Rome - Rome", "km": 131, "type": "Vlak ➖", "w": {"SPR": 1.0, "GC": 0.0, "ITT": 0.0, "MTN": 0.0}},
]

def laad_profiel_scores():
    bestand = "giro262/profile_score.csv"
    if os.path.exists(bestand):
        try:
            df_scores = pd.read_csv(bestand, sep=None, engine='python')
            df_scores.columns = df_scores.columns.str.strip()
            for _, row in df_scores.iterrows():
                try:
                    s_id = int(row['id'])
                    for e in GIRO_ETAPPES:
                        if e['id'] == s_id:
                            if 'SPR' in df_scores.columns: e['w']['SPR'] = float(row['SPR'])
                            if 'GC' in df_scores.columns: e['w']['GC'] = float(row['GC'])
                            if 'ITT' in df_scores.columns: e['w']['ITT'] = float(row['ITT'])
                            if 'MTN' in df_scores.columns: e['w']['MTN'] = float(row['MTN'])
                except:
                    continue
        except Exception as err:
            st.warning(f"Fout bij inladen profile_score.csv: {err}")

laad_profiel_scores()

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
        "de lie": "arnaud de lie", "a. de lie": "arnaud de lie"
    }
    if naam_norm in bekende_gevallen:
        correct = bekende_gevallen[naam_norm]
        for target in lijst_met_namen:
            if correct in target: return dict_met_namen[target]
    if naam_norm in lijst_met_namen: return dict_met_namen[naam_norm]
    bests = process.extractBests(naam_norm, lijst_met_namen, scorer=fuzz.token_set_ratio, limit=5)
    if bests and bests[0][1] >= 75:
        top_score = bests[0][1]
        candidates = [b[0] for b in bests if b[1] >= top_score - 3]
        candidates.sort(key=lambda x: (abs(len(x) - len(naam_norm)), -fuzz.ratio(naam_norm, x)))
        return dict_met_namen[candidates[0]]
    return naam

def get_clickable_image_html(image_path, fallback_text, link):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()
            ext = "png" if image_path.lower().endswith(".png") else "jpeg"
            img_src = f"data:image/{ext};base64,{encoded_string}"
        except Exception:
            img_src = f"https://placehold.co/600x400/eeeeee/000000?text={fallback_text}"
    else:
        img_src = f"https://placehold.co/600x400/eeeeee/000000?text={fallback_text}"
    return f'<a href="{link}" target="_blank"><img src="{img_src}" width="100%" style="border-radius:8px;"></a>'

# --- AI ETAPPE VOORSPELLER ---
def genereer_ai_etappe_voorspellingen(df, etappes, top_x, custom_weights):
    ai_voorspellingen = {}
    for etappe in etappes:
        df_temp = df.copy()
        w = custom_weights[str(etappe["id"])]
        
        df_temp['stage_score'] = (
            (df_temp['SPR'] * w['SPR']) + 
            (df_temp['GC'] * w['GC']) + 
            (df_temp['ITT'] * w['ITT']) + 
            (df_temp['MTN'] * w['MTN'])
        )
        
        top_renners = df_temp.sort_values(by=['stage_score', 'Giro_EV'], ascending=[False, False])['Renner'].head(top_x).tolist()
        
        while len(top_renners) < 10: 
            top_renners.append(None)
            
        ai_voorspellingen[str(etappe["id"])] = top_renners
    return ai_voorspellingen

# --- DATA LADEN ---
@st.cache_data
def load_giro_data():
    prijzen_file = "giro262/sporza_giro26_startlijst.csv"
    stats_file = "renners_stats.csv"
    
    if not os.path.exists(prijzen_file):
        st.error(f"🚨 Het bestand `{prijzen_file}` ontbreekt in je map!")
        return pd.DataFrame()
    if not os.path.exists(stats_file):
        st.error(f"🚨 Het bestand `{stats_file}` ontbreekt in je map!")
        return pd.DataFrame()
        
    try:
        df_prog = pd.read_csv(prijzen_file, sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
        df_stats = pd.read_csv(stats_file, sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip') 
        
        df_prog.columns, df_stats.columns = df_prog.columns.str.strip(), df_stats.columns.str.strip()
        
        if 'Naam' in df_prog.columns: df_prog = df_prog.rename(columns={'Naam': 'Renner'})
        if 'Naam' in df_stats.columns: df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        if 'Ploeg' in df_stats.columns: df_stats = df_stats.rename(columns={'Ploeg': 'Team'})
        
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        norm_to_stats = {normalize_name_logic(n): n for n in df_stats['Renner'].unique()}
        df_prog['Renner_Stats'] = df_prog['Renner'].apply(lambda x: match_naam_slim(x, norm_to_stats))
        
        merged_df = pd.merge(df_prog, df_stats, left_on='Renner_Stats', right_on='Renner', how='left', suffixes=('', '_drop'))
        merged_df = merged_df.drop(columns=[c for c in merged_df.columns if '_drop' in c or c == 'Renner_Stats'])
        
        if 'Prijs' not in merged_df.columns:
            st.error("🚨 Fout in de startlijst: de kolom `Prijs` is niet gevonden.")
            return pd.DataFrame()

        merged_df['Prijs'] = pd.to_numeric(merged_df['Prijs'], errors='coerce').fillna(0)
        merged_df.loc[merged_df['Prijs'] > 1000, 'Prijs'] = merged_df['Prijs'] / 1000000
        merged_df.loc[merged_df['Prijs'] == 0.8, 'Prijs'] = 0.75
        
        merged_df = merged_df[merged_df['Prijs'] > 0].sort_values(by='Prijs', ascending=False).drop_duplicates(subset=['Renner'])
        
        if merged_df.empty:
            st.error("🚨 De bestanden zijn geladen, maar na filtering (Prijs > 0) bleven er 0 renners over.")
            return pd.DataFrame()
            
        for col in ['GC', 'SPR', 'ITT', 'MTN']:
            if col not in merged_df.columns: merged_df[col] = 0
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
            
        return merged_df
    except Exception as e: 
        st.error(f"🚨 Er trad een fout op bij het laden van de data: {e}")
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
    pts_map = [50, 40, 30, 25, 20, 16, 14, 12, 10, 8]
    pred_series = pd.Series(0, index=df.index)
    for stage_id, preds in predictions.items():
        for pos in range(min(top_x, len(preds))):
            renner = preds[pos]
            if renner and renner != "-":
                idx = df[df['Renner'] == renner].index
                if not idx.empty: pred_series.loc[idx[0]] += pts_map[pos]
    return pred_series

# --- SOLVER ---
def solve_giro_team(df, max_bud, max_ren, max_per_team, force_base, ban_base, ev_column):
    prob = pulp.LpProblem("Sporza_Giro_Solver", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Select", df.index, cat='Binary')
    prob += pulp.lpSum([df.loc[i, ev_column] * x[i] for i in df.index])
    prob += pulp.lpSum([x[i] for i in df.index]) == max_ren
    prob += pulp.lpSum([df.loc[i, 'Prijs'] * x[i] for i in df.index]) <= max_bud
    for team in df['Team'].unique():
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
st.title("🇮🇹 Grote Ronde: Sporza Giromanager")
st.markdown("*Data en Statistieken van [Wielerorakel](https://wielerorakel.nl/)*")

df_raw = load_giro_data()
if df_raw.empty: st.stop()

# Sessiestates initialiseren
if "giro_selected_riders" not in st.session_state: 
    st.session_state.giro_selected_riders = []
if "giro_stage_predictions" not in st.session_state:
    st.session_state.giro_stage_predictions = {str(stage["id"]): [None]*10 for stage in GIRO_ETAPPES}
if "giro_weights" not in st.session_state:
    st.session_state.giro_weights = {str(e["id"]): e["w"].copy() for e in GIRO_ETAPPES}

with st.sidebar:
    st.header(f"👤 Profiel: {speler_naam.capitalize()}")
    if speler_naam != "gast":
        c_cloud1, c_cloud2 = st.columns(2)
        with c_cloud1:
            if st.button("💾 Opslaan", type="primary", use_container_width=True):
                data = {
                    "selected_riders": st.session_state.giro_selected_riders, 
                    "predictions": st.session_state.giro_stage_predictions, 
                    "weights": st.session_state.giro_weights,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                supabase.table(TABEL_NAAM).update({DB_KOLOM: data}).eq("username", speler_naam).execute()
                st.success("Opgeslagen!")
        with c_cloud2:
            if st.button("🔄 Inladen", use_container_width=True):
                res = supabase.table(TABEL_NAAM).select(DB_KOLOM).eq("username", speler_naam).execute()
                if res.data and res.data[0].get(DB_KOLOM):
                    db_data = res.data[0][DB_KOLOM]
                    st.session_state.giro_selected_riders = db_data.get("selected_riders", [])
                    st.session_state.giro_stage_predictions = db_data.get("predictions", {str(stage["id"]): [None]*10 for stage in GIRO_ETAPPES})
                    st.session_state.giro_weights = db_data.get("weights", {str(stage["id"]): stage["w"].copy() for stage in GIRO_ETAPPES})
                    st.rerun()
    
    st.divider()
    bouw_methode = st.radio("Samenstel methode:", ["1. Volledig AI (Stats)", "2. Mijn Voorspellingen (+ AI opvulling)"])
    top_x_voorspellingen = st.number_input("Top X per etappe", 1, 10, 3)
    max_budget = st.number_input("Budget (Miljoen)", value=100.0)
    max_renners = st.number_input("Aantal Renners", value=16)
    max_per_ploeg = st.number_input("Max per ploeg", value=3)
    
    df = calculate_giro_ev(df_raw)
    df['Prediction_EV'] = calculate_prediction_ev(df, st.session_state.giro_stage_predictions, top_x_voorspellingen)
    df['Combined_EV'] = (df['Prediction_EV'] * 1000) + df['Giro_EV']

    with st.expander("🔒 Forceren / Uitsluiten"):
        force_base = st.multiselect("🟢 Moet in team:", options=df['Renner'].tolist())
        ban_base = st.multiselect("🔴 Niet in team:", options=[r for r in df['Renner'].tolist() if r not in force_base])

    if st.button("🚀 BEREKEN GIRO TEAM", type="primary", use_container_width=True):
        ev_col = "Giro_EV" if "1." in bouw_methode else "Combined_EV"
        res = solve_giro_team(df, max_budget, max_renners, max_per_ploeg, force_base, ban_base, ev_col)
        if res: st.session_state.giro_selected_riders = res; st.rerun()

    st.divider()
    st.markdown("#### 📥 Exporteer Data")
    d_col1, d_col2 = st.columns(2)
    
    export_data = {
        "team": st.session_state.giro_selected_riders,
        "predictions": st.session_state.giro_stage_predictions,
        "weights": st.session_state.giro_weights
    }
    d_col1.download_button(
        label="📄 JSON", 
        data=json.dumps(export_data, indent=2), 
        file_name="sporza_giro_export.json", 
        mime="application/json", 
        use_container_width=True
    )
    
    if st.session_state.giro_selected_riders:
        export_df = df[df['Renner'].isin(st.session_state.giro_selected_riders)].copy()
        csv_data = export_df.to_csv(index=False, sep=";").encode('utf-8-sig')
        d_col2.download_button(
            label="📊 Excel (CSV)", 
            data=csv_data, 
            file_name="sporza_giro_team.csv", 
            mime="text/csv", 
            use_container_width=True
        )
    else:
        d_col2.button("📊 Excel (CSV)", disabled=True, use_container_width=True, help="Bereken eerst een team")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Jouw Selectie", "📅 Etappe Voorspellingen", "📋 Database (Giro)", "ℹ️ Uitleg"])

with tab1:
    if st.session_state.giro_selected_riders:
        start_team_df = df[df['Renner'].isin(st.session_state.giro_selected_riders)].copy()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 Budget over", f"€ {max_budget - start_team_df['Prijs'].sum():.2f}M")
        m2.metric("🚴 Renners", f"{len(start_team_df)} / {max_renners}")
        m3.metric("🎯 EV (AI)", f"{start_team_df['Giro_EV'].sum()}")
        m4.metric("🏆 EV (Jouw)", f"{start_team_df['Prediction_EV'].sum()}")
        st.dataframe(start_team_df.sort_values(by='Prijs', ascending=False), hide_index=True, use_container_width=True)

with tab2:
    st.subheader(f"🏆 Voorspel de Top {top_x_voorspellingen}")
    c1, c2 = st.columns([1, 4])
    if c1.button("🤖 Vul in met AI"):
        st.session_state.giro_stage_predictions = genereer_ai_etappe_voorspellingen(df, GIRO_ETAPPES, top_x_voorspellingen, st.session_state.giro_weights)
        st.rerun()
    if c2.button("🗑️ Wis alles"):
        st.session_state.giro_stage_predictions = {str(s["id"]): [None]*10 for s in GIRO_ETAPPES}
        st.rerun()
    
    renners_opties = ["-"] + sorted(df['Renner'].tolist())
    for etappe in GIRO_ETAPPES:
        stage_id = str(etappe["id"])
        
        cw = st.session_state.giro_weights[stage_id]
        weight_str = f"SPR:{int(cw['SPR']*100)}% GC:{int(cw['GC']*100)}% ITT:{int(cw['ITT']*100)}% MTN:{int(cw['MTN']*100)}%"
        
        with st.expander(f"Etappe {etappe['id']}: {etappe['route']} ({etappe['type']}) | 🤖 {weight_str}"):
            
            giro_link = "https://www.giroditalia.it/en/the-route/"
            map_path = f"giro262/giro26-{etappe['id']}-map.jpg"
            prof_path = f"giro262/giro26-{etappe['id']}-hp.jpg" 
            
            st.markdown(f"*(Klik op een afbeelding voor de officiële info)*")
            i1, i2 = st.columns(2)
            i1.markdown(get_clickable_image_html(map_path, f"Kaart+Etappe+{etappe['id']}", giro_link), unsafe_allow_html=True)
            i2.markdown(get_clickable_image_html(prof_path, f"Profiel+Etappe+{etappe['id']}", giro_link), unsafe_allow_html=True)
            
            # Subtiele sectie voor weging & voorspelling, dicht bij elkaar
            st.markdown("###### Weging & Voorspelling")
            wc1, wc2, wc3, wc4 = st.columns(4)
            new_spr = wc1.number_input("Sprint (SPR)", 0.0, 1.0, cw["SPR"], 0.1, key=f"wspr_{stage_id}")
            new_gc  = wc2.number_input("Klassement (GC)", 0.0, 1.0, cw["GC"], 0.1, key=f"wgc_{stage_id}")
            new_itt = wc3.number_input("Tijdrit (ITT)", 0.0, 1.0, cw["ITT"], 0.1, key=f"witt_{stage_id}")
            new_mtn = wc4.number_input("Klim/Aanval (MTN)", 0.0, 1.0, cw["MTN"], 0.1, key=f"wmtn_{stage_id}")
            
            st.session_state.giro_weights[stage_id] = {"SPR": new_spr, "GC": new_gc, "ITT": new_itt, "MTN": new_mtn}
            
            for i in range(0, top_x_voorspellingen, 5):
                cols = st.columns(min(5, top_x_voorspellingen - i))
                for j, col in enumerate(cols):
                    pos = i + j
                    val = st.session_state.giro_stage_predictions[stage_id][pos]
                    idx = renners_opties.index(val) if val in renners_opties else 0
                    new_val = col.selectbox(f"Pos {pos+1}", renners_opties, index=idx, key=f"s{stage_id}p{pos}")
                    st.session_state.giro_stage_predictions[stage_id][pos] = new_val if new_val != "-" else None

with tab3:
    st.dataframe(df.sort_values('Giro_EV', ascending=False), hide_index=True, use_container_width=True)
