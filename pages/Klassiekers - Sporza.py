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
st.set_page_config(page_title="Sporza Klassiekers AI", layout="wide", page_icon="🚴")

# --- CHECK INLOG & DATABASE SETUP ---
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

# --- HULPFUNCTIES ---
def normalize_name_logic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def get_file_mod_time(filepath):
    return os.path.getmtime(filepath) if os.path.exists(filepath) else 0

# --- DATA LADEN (SPORZA SPECIFIEK) ---
@st.cache_data
def load_and_merge_data(prog_mod_time, stats_mod_time):
    try:
        df_prog = pd.read_csv("sporza_prijzen_startlijst.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
        df_prog.columns = df_prog.columns.str.strip()
        if 'Naam' in df_prog.columns and 'Renner' not in df_prog.columns:
            df_prog = df_prog.rename(columns={'Naam': 'Renner'})
        
        df_stats = pd.read_csv("renners_stats.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip') 
        df_stats.columns = df_stats.columns.str.strip()
        if 'Naam' in df_stats.columns and 'Renner' not in df_stats.columns:
            df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        
        overlap_cols = [c for c in df_stats.columns if c in df_prog.columns and c != 'Renner']
        df_stats = df_stats.drop(columns=overlap_cols)
        
        short_names = df_prog['Renner'].unique()
        full_names = df_stats['Renner'].unique()
        norm_to_full = {normalize_name_logic(n): n for n in full_names}
        norm_full_names = list(norm_to_full.keys())
        name_mapping = {}
        
        manual_overrides = {
            "Poel": "Mathieu van der Poel", "Aert": "Wout van Aert", "Lie": "Arnaud De Lie",
            "Gils": "Maxim Van Gils", "Broek": "Frank van den Broek",
            "Magnier": "Paul Magnier", "Pogacar": "Tadej Pogačar", "Skujins": "Toms Skujiņš",
            "Kooij": "Olav Kooij"
        }
        
        for short in short_names:
            if short in manual_overrides:
                name_mapping[short] = manual_overrides[short]
            else:
                norm_short = normalize_name_logic(short)
                match_res = process.extractOne(norm_short, norm_full_names, scorer=fuzz.token_set_ratio)
                if match_res and match_res[1] > 75:
                    name_mapping[short] = norm_to_full[match_res[0]]
                else:
                    name_mapping[short] = short

        df_prog['Renner_Full'] = df_prog['Renner'].map(name_mapping)
        merged_df = pd.merge(df_prog, df_stats, left_on='Renner_Full', right_on='Renner', how='left')
        
        if 'Renner_x' in merged_df.columns:
            merged_df = merged_df.drop(columns=['Renner_x', 'Renner_y'], errors='ignore')
            
        merged_df['Prijs'] = pd.to_numeric(merged_df['Prijs'], errors='coerce').fillna(0).astype(int)
        merged_df = merged_df.sort_values(by='Prijs', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=['Renner_Full'], keep='first')
        merged_df = merged_df.rename(columns={'Renner_Full': 'Renner'})
        
        ALLE_KOERSEN = ["OML", "KBK", "SAM", "STR", "NOK", "BKC", "MSR", "RVB", "E3", "IFF", "DDV", "RVV", "SP", "PR", "RVL", "BRP", "AGT", "WAP", "LBL"]
        available_races = [k for k in ALLE_KOERSEN if k in merged_df.columns]
        
        all_stats_cols = ['COB', 'HLL', 'SPR', 'AVG', 'FLT', 'MTN', 'ITT', 'GC', 'OR', 'TTL']
        for col in available_races + all_stats_cols:
            if col not in merged_df.columns:
                merged_df[col] = 0
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
            
        if 'Team' not in merged_df.columns:
            merged_df['Team'] = 'Onbekend'
        else:
            merged_df['Team'] = merged_df['Team'].fillna('Onbekend')
        
        koers_stat_map = {
            "OML": "COB", "KBK": "SPR", "SAM": "COB", "STR": "HLL", "NOK": "SPR", 
            "BKC": "SPR", "MSR": "AVG", "RVB": "SPR", "E3": "COB", "IFF": "SPR", 
            "DDV": "COB", "RVV": "COB", "SP": "SPR", "PR": "COB", "RVL": "SPR", 
            "BRP": "HLL", "AGT": "HLL", "WAP": "HLL", "LBL": "HLL"
        }
        
        return merged_df, available_races, koers_stat_map
    except Exception as e:
        st.error(f"Fout in dataverwerking: {e}")
        return pd.DataFrame(), [], {}

@st.cache_data
def get_uitslagen(file_mod_time, alle_renners):
    if not os.path.exists("uitslagen.csv"):
        return pd.DataFrame()
    try:
        df_raw_uitslagen = pd.read_csv("uitslagen.csv", sep=None, engine='python')
        df_raw_uitslagen.columns = [str(c).strip().title() for c in df_raw_uitslagen.columns]
        
        if 'Race' not in df_raw_uitslagen.columns or 'Rider' not in df_raw_uitslagen.columns or 'Rnk' not in df_raw_uitslagen.columns:
            return pd.DataFrame()
            
        scorito_naar_sporza_map = {
            'OHN': 'OML', 'SB': 'STR', 'BDP': 'RVB', 
            'GW': 'IFF', 'BP': 'BRP', 'AGR': 'AGT', 'WP': 'WAP'
        }
            
        uitslag_parsed = []
        for index, row in df_raw_uitslagen.iterrows():
            koers_origineel = str(row['Race']).strip().upper()
            koers = scorito_naar_sporza_map.get(koers_origineel, koers_origineel)
            
            rank_str = str(row['Rnk']).strip().upper()
            if rank_str in ['DNS', 'NAN', '']:
                continue
            
            rider_name = str(row['Rider']).strip()
            match = process.extractOne(rider_name, alle_renners, scorer=fuzz.token_set_ratio)
            if match and match[1] > 70:
                uitslag_parsed.append({
                    "Race": koers,
                    "Rnk": rank_str,
                    "Renner": match[0]
                })
        return pd.DataFrame(uitslag_parsed)
    except:
        return pd.DataFrame()

def calculate_sporza_ev(df, available_races, koers_stat_map, method):
    df = df.copy()
    pts_monument = [125, 100, 80, 70, 60, 50, 45, 40, 37, 34, 31, 28, 25, 22, 20, 18, 16, 14, 12, 10]
    pts_wt = [100, 80, 65, 55, 48, 40, 36, 32, 30, 27, 24, 22, 20, 18, 16, 14, 12, 10, 9, 8]
    pts_non_wt = [80, 64, 52, 44, 38, 32, 29, 26, 24, 22, 20, 18, 16, 14, 12, 11, 10, 9, 8, 7]
    
    monuments = ["MSR", "RVV", "PR", "LBL"]
    world_tour = ["OML", "STR", "RVB", "E3", "IFF", "DDV", "AGT", "WAP"]
    
    race_evs = {}
    for koers in available_races:
        stat = koers_stat_map.get(koers, 'AVG')
        starters = df[df[koers] == 1].copy()
        starters = starters.sort_values(by=[stat, 'AVG'], ascending=[False, False])
        
        if koers in monuments: scorito_pts = pts_monument
        elif koers in world_tour: scorito_pts = pts_wt
        else: scorito_pts = pts_non_wt
        
        race_ev = pd.Series(0.0, index=df.index)
        for i, idx in enumerate(starters.index):
            val = 0.0
            if "Sporza Ranking" in method:
                val = scorito_pts[i] if i < len(scorito_pts) else 0.0
            elif "Originele Curve" in method:
                val = (starters.loc[idx, stat] / 100)**4 * scorito_pts[0]
                
            if i == 0: val += 30
            elif i == 1: val += 25
            elif i == 2: val += 20
            race_ev.loc[idx] = val
        
        race_evs[koers] = race_ev
        df[f'EV_{koers}'] = race_ev

    df['EV_all'] = sum(race_evs.values()) if race_evs else 0.0
    df['Sporza_EV'] = df['EV_all'].fillna(0).round(0).astype(int)
    df['Waarde (EV/M)'] = (df['Sporza_EV'] / df['Prijs']).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)
    
    return df

def bepaal_klassieker_type(row):
    cob = row.get('COB', 0)
    hll = row.get('HLL', 0)
    spr = row.get('SPR', 0)
    
    elite = []
    if cob >= 85: elite.append('Kassei')
    if hll >= 85: elite.append('Heuvel')
    if spr >= 85: elite.append('Sprint')
    
    if len(elite) == 3: return 'Allround / Multispecialist'
    elif len(elite) == 2: return ' / '.join(elite)
    elif len(elite) == 1: return elite[0]
    else:
        s = {'Kassei': cob, 'Heuvel': hll, 'Sprint': spr, 'Klimmer': row.get('MTN', 0), 'Tijdrit': row.get('ITT', 0), 'Klassement': row.get('GC', 0)}
        if sum(s.values()) == 0: return 'Onbekend'
        return max(s, key=s.get)

# --- SPORZA SOLVER ---
def solve_sporza_dynamic(df, available_races, t_moments, force_base, ban_base, exclude_list):
    prob = pulp.LpProblem("Sporza_Solver_Dynamic", pulp.LpMaximize)
    K = len(t_moments)

    x = pulp.LpVariable.dicts("Base", df.index, cat='Binary')
    y = [pulp.LpVariable.dicts(f"Uit_{k}", df.index, cat='Binary') for k in range(K)]
    z = [pulp.LpVariable.dicts(f"In_{k}", df.index, cat='Binary') for k in range(K)]
    s = {r: pulp.LpVariable.dicts(f"Start_{r}", df.index, cat='Binary') for r in available_races}

    prob += pulp.lpSum([s[r][i] * df.loc[i, f'EV_{r}'] for r in available_races for i in df.index])
    prob += pulp.lpSum([x[i] for i in df.index]) == 20

    for i in df.index:
        renner = df.loc[i, 'Renner']
        if renner in force_base: prob += x[i] == 1
        if renner in ban_base: prob += x[i] == 0
        if renner in exclude_list: 
            prob += x[i] == 0
            for k in range(K): prob += z[k][i] == 0

    for k in range(K):
        prob += pulp.lpSum([y[k][i] for i in df.index]) == 1 
        prob += pulp.lpSum([z[k][i] for i in df.index]) == 1 

    for i in df.index:
        prob += x[i] + pulp.lpSum([z[k][i] for k in range(K)]) <= 1

    for k in range(K):
        for i in df.index:
            active_before = x[i] - pulp.lpSum([y[m][i] for m in range(k)]) + pulp.lpSum([z[m][i] for m in range(k)])
            prob += y[k][i] <= active_before

    penalties = [0, 0, 0, 0, 1, 3, 6, 10, 15]
    teams = df['Team'].unique()

    for p in range(K + 1):
        budget_limit = 120 - penalties[p]
        prob += pulp.lpSum([(x[j] - pulp.lpSum([y[m][j] for m in range(p)]) + pulp.lpSum([z[m][j] for m in range(p)])) * df.loc[j, 'Prijs'] for j in df.index]) <= budget_limit
        
        for team in teams:
            team_indices = df[df['Team'] == team].index
            prob += pulp.lpSum([(x[j] - pulp.lpSum([y[m][j] for m in range(p)]) + pulp.lpSum([z[m][j] for m in range(p)])) for j in team_indices]) <= 4

    for r in available_races:
        idx_r = available_races.index(r)
        active_transfers = []
        for k, m in enumerate(t_moments):
            if available_races.index(m) < idx_r:
                active_transfers.append(k)

        for i in df.index:
            active_at_r = x[i] - pulp.lpSum([y[k][i] for k in active_transfers]) + pulp.lpSum([z[k][i] for k in active_transfers])
            prob += s[r][i] <= active_at_r
            prob += s[r][i] <= df.loc[i, r]

        prob += pulp.lpSum([s[r][i] for i in df.index]) <= 12

    time_limit = 20 if K <= 2 else 40
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    if pulp.LpStatus[prob.status] == 'Optimal':
        base_team = [df.loc[i, 'Renner'] for i in df.index if x[i].varValue > 0.5]
        transfer_plan = []
        for k in range(K):
            uit = [df.loc[i, 'Renner'] for i in df.index if y[k][i].varValue > 0.5]
            erin = [df.loc[i, 'Renner'] for i in df.index if z[k][i].varValue > 0.5]
            if uit and erin:
                transfer_plan.append({"uit": uit[0], "in": erin[0], "moment": t_moments[k]})
        return base_team, transfer_plan
    return [], []

# --- HOOFDCODE ---
prog_time = get_file_mod_time("sporza_prijzen_startlijst.csv")
stats_time = get_file_mod_time("renners_stats.csv")
df_raw, available_races, koers_mapping = load_and_merge_data(prog_time, stats_time)

if df_raw.empty:
    st.warning("Data is leeg of kon niet worden geladen. Controleer of 'sporza_prijzen_startlijst.csv' bestaat.")
    st.stop()

if "sporza_selected_riders" not in st.session_state: st.session_state.sporza_selected_riders = []
if "sporza_transfer_plan" not in st.session_state: st.session_state.sporza_transfer_plan = []

with st.sidebar:
    st.title("🚴 Sporza AI Coach")
    st.header(f"👤 Profiel: {speler_naam.capitalize()}")
    
    st.write("☁️ **Cloud Database**")
    if speler_naam != "gast":
        c_cloud1, c_cloud2 = st.columns(2)
        with c_cloud1:
            if st.button("💾 Opslaan", type="primary", use_container_width=True, key="sporza_opslaan"):
                try:
                    team_data = {
                        "selected_riders": st.session_state.sporza_selected_riders, 
                        "transfer_plan": st.session_state.sporza_transfer_plan, 
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    supabase.table(TABEL_NAAM).update({"sporza_team": team_data}).eq("username", speler_naam).execute()
                    st.success("Cloud-backup geslaagd!")
                except Exception as e: 
                    st.error(f"Fout: {e}")
        with c_cloud2:
            if st.button("🔄 Inladen", use_container_width=True, key="sporza_inladen"):
                try:
                    res = supabase.table(TABEL_NAAM).select("sporza_team").eq("username", speler_naam).execute()
                    if res.data and res.data[0].get('sporza_team'):
                        d = res.data[0]['sporza_team']
                        st.session_state.sporza_selected_riders = d.get("selected_riders", [])
                        st.session_state.sporza_transfer_plan = d.get("transfer_plan", [])
                        st.success(f"Team geladen (van {d.get('ts', '?')})")
                        st.rerun()
                    else: 
                        st.warning("Geen team gevonden in de cloud.")
                except Exception as e: 
                    st.error(f"Fout: {e}")
    else:
        st.info("Log in met een account om cloud-opslag te gebruiken.")

    st.divider()
    st.write("📁 **Lokale Backup (.json)**")
    
    save_data = {"selected_riders": st.session_state.sporza_selected_riders, "transfer_plan": st.session_state.sporza_transfer_plan}
    st.download_button("📥 Download Team als .JSON", data=json.dumps(save_data), file_name=f"{speler_naam}_sporza_team.json", mime="application/json", use_container_width=True)
    
    uploaded_file = st.file_uploader("📂 Upload Team (.json)", type="json")
    if uploaded_file is not None and st.button("Laad .json in", use_container_width=True):
        try:
            ld = json.load(uploaded_file)
            st.session_state.sporza_selected_riders = ld.get("selected_riders", [])
            st.session_state.sporza_transfer_plan = ld.get("transfer_plan", [])
            st.success("✅ Lokaal bestand geladen!")
            st.rerun()
        except Exception as e:
            st.error("Fout bij inladen.")
            
    st.divider()
    
    ev_method = st.selectbox("🧮 Rekenmodel (EV)", ["1. Sporza Ranking (Dynamisch)", "2. Originele Curve (Macht 4)"])
    toon_uitslagen = st.checkbox("🏁 Koersen zijn begonnen (Toon uitslagen in Matrix)", value=False)
    
    st.divider()
    st.markdown("### 🔁 Transfer Strategie")
    num_transfers = st.slider("Aantal geplande transfers", 0, 5, 0)
    
    t_moments = []
    if num_transfers > 0:
        st.write("Wanneer wil je de wissels inzetten?")
        for i in range(num_transfers):
            default_idx = min(len(available_races)-2, 13)
            moment = st.selectbox(f"Wissel {i+1} ná:", options=available_races[:-1], index=default_idx, key=f"t_{i}")
            t_moments.append(moment)
            
        t_moments = sorted(t_moments, key=lambda x: available_races.index(x))
        
        penalties = [0, 0, 0, 0, 1, 3, 6, 10, 15]
        penalty = penalties[num_transfers]
        if penalty > 0:
            st.error(f"⚠️ Let op: {num_transfers} transfers kost je € {penalty}M totaalbudget na je laatste wissel!")
        else:
            st.success("✅ Deze transfers zijn gratis in Sporza.")

    df = calculate_sporza_ev(df_raw, available_races, koers_mapping, ev_method)
    df['Type'] = df.apply(bepaal_klassieker_type, axis=1)

    with st.expander("🔒 Renners Forceren / Uitsluiten", expanded=False):
        force_base = st.multiselect("🟢 Moet in start-team:", options=df['Renner'].tolist())
        ban_base = st.multiselect("🔴 Niet in start-team:", options=[r for r in df['Renner'].tolist() if r not in force_base])
        exclude_list = st.multiselect("🚫 Compleet negeren (hele jaar):", options=[r for r in df['Renner'].tolist() if r not in force_base + ban_base])

    st.write("")
    if st.button("🚀 BEREKEN SPORZA TEAM", type="primary", use_container_width=True):
        with st.spinner("Wiskundige berekening loopt... Dit kan bij meerdere transfers even duren."):
            res, plan = solve_sporza_dynamic(df, available_races, t_moments, force_base, ban_base, exclude_list)
            if res:
                st.session_state.sporza_selected_riders = res
                st.session_state.sporza_transfer_plan = plan
                st.rerun()
            else:
                st.error("Geen geldige combinatie mogelijk binnen budget (120M) en ploegrestricties (Max 4 per ploeg).")

st.title("🚴 Voorjaarsklassiekers: Sporza Wielermanager")
st.markdown("**Met dank aan:** [Wielerorakel.nl](https://www.cyclingoracle.com/)")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Jouw Team & Transfers", "🗓️ Startlijst & Uitslagen", "👑 Kopmannen", "📋 Database", "ℹ️ Uitleg"])

if not st.session_state.sporza_selected_riders:
    with tab1: st.info("👈 Kies je instellingen en klik op **Bereken Sporza Team** om te beginnen!")
else:
    all_display_riders = list(set(st.session_state.sporza_selected_riders + [t['in'] for t in st.session_state.sporza_transfer_plan]))
    current_df = df[df['Renner'].isin(all_display_riders)].copy()
    
    def bepaal_rol_en_moment(naam):
        for t in st.session_state.sporza_transfer_plan:
            if naam == t['uit']: return f"Verkocht na {t['moment']}"
            if naam == t['in']: return f"Gekocht na {t['moment']}"
        return 'Basis (Blijft)'

    current_df['Rol'] = current_df['Renner'].apply(bepaal_rol_en_moment)
    start_team_df = current_df[current_df['Renner'].isin(st.session_state.sporza_selected_riders)]
    
    with tab1:
        st.subheader("📊 Dashboard")
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Start Budget (Overig)", f"€ {120 - start_team_df['Prijs'].sum()}M")
        m2.metric("🚴 Renners (Start)", f"{len(start_team_df)} / 20")
        m3.metric("🎯 Start EV (Totaal)", f"{start_team_df['Sporza_EV'].sum()}")
        st.divider()
        
        col_t1, col_t2 = st.columns([1, 1], gap="large")
        with col_t1:
            st.markdown("**🛡️ Jouw Start-Team**")
            st.dataframe(start_team_df[['Renner', 'Prijs', 'Team', 'Type', 'Rol']].sort_values(by='Prijs', ascending=False), hide_index=True, use_container_width=True)
        
        with col_t2:
            st.markdown(f"**🔁 Het Transfer Plan ({len(st.session_state.sporza_transfer_plan)} gepland)**")
            if not st.session_state.sporza_transfer_plan:
                st.info("Geen transfers ingepland.")
            else:
                temp_team = list(st.session_state.sporza_selected_riders)
                penalties = [0, 0, 0, 0, 1, 3, 6, 10, 15]
                for i, t in enumerate(st.session_state.sporza_transfer_plan):
                    if t['uit'] in temp_team: temp_team.remove(t['uit'])
                    if t['in'] not in temp_team: temp_team.append(t['in'])
                    
                    budget_limiet = 120 - penalties[i+1]
                    budget_now = budget_limiet - df[df['Renner'].isin(temp_team)]['Prijs'].sum()
                    kosten_text = "Gratis" if i < 3 else f"-€{penalties[i+1]-penalties[i]}M Boete"
                    
                    st.markdown(f"***Wissel {i+1} (ná {t['moment']} | Speling: €{budget_now}M | {kosten_text})***")
                    c_uit, c_in = st.columns(2)
                    with c_uit: st.error(f"❌ {t['uit']}")
                    with c_in: st.success(f"📥 {t['in']}")
                    st.write("")

        st.divider()

        with st.container(border=True):
            st.subheader("🛠️ Team Finetuner (Start-Team aanpassen)")
            st.markdown("Vervang renners in je team. De AI zoekt direct naar passend budget en repareert je geplande transfers en 12-starters logica.")
            
            c_fine1, c_fine2 = st.columns(2)
            with c_fine1: 
                to_replace = st.multiselect("❌ Selecteer renner(s) om te verwijderen:", options=st.session_state.sporza_selected_riders)
            
            to_add = []
            if to_replace:
                kept_riders = [r for r in st.session_state.sporza_selected_riders if r not in to_replace]
                cost_kept = df[df['Renner'].isin(kept_riders)]['Prijs'].sum()
                max_affordable = 120 - cost_kept
                
                with c_fine2: 
                    available_replacements = [r for r in df['Renner'].tolist() if r not in st.session_state.sporza_selected_riders]
                    to_add_manual = st.multiselect("📥 Handmatige vervanger(s):", options=available_replacements)
                    
                sugg_df = df[~df['Renner'].isin(st.session_state.sporza_selected_riders)][df['Prijs'] <= max_affordable].sort_values(by='Sporza_EV', ascending=False).head(5)
                
                sugg_keuze = []
                if not sugg_df.empty:
                    st.info(f"💡 **Top AI Suggesties (Totaal overgebleven budget voor {len(to_replace)} renner(s): € {max_affordable}M):**")
                    st.dataframe(sugg_df[['Renner', 'Prijs', 'Waarde (EV/M)', 'Sporza_EV', 'Type']], hide_index=True, use_container_width=True)
                    sugg_keuze = st.multiselect("👉 Of selecteer hier direct een AI-suggestie:", options=sugg_df['Renner'].tolist())

                to_add = list(set(to_add_manual + sugg_keuze))
                
                if to_add:
                    st.markdown("**📊 Vergelijking:**")
                    compare_riders = list(set(to_replace + to_add))
                    compare_df = df[df['Renner'].isin(compare_riders)].copy()
                    compare_cols = ['Renner', 'Prijs', 'Waarde (EV/M)', 'Sporza_EV'] + available_races
                    comp_display = compare_df[compare_cols].copy()
                    
                    def mark_status(renner):
                        if renner in to_replace: return '❌ Eruit'
                        if renner in to_add: return '📥 Erin'
                        return ''
                        
                    comp_display.insert(1, 'Actie', comp_display['Renner'].apply(mark_status))
                    comp_display[available_races] = comp_display[available_races].applymap(lambda x: '✅' if x == 1 else '-')
                    
                    def style_compare(row):
                        if row['Actie'] == '❌ Eruit': return ['background-color: rgba(255, 99, 71, 0.2)'] * len(row)
                        if row['Actie'] == '📥 Erin': return ['background-color: rgba(144, 238, 144, 0.2)'] * len(row)
                        return [''] * len(row)
                        
                    st.dataframe(comp_display.style.apply(style_compare, axis=1), hide_index=True, use_container_width=True)

                    if st.button("🚀 VOER WIJZIGING DOOR", type="primary", use_container_width=True):
                        new_force_base = kept_riders + to_add
                        new_ban_base = [r for r in df['Renner'].tolist() if r not in new_force_base]
                        
                        if len(new_force_base) == 20:
                            with st.spinner("AI herberekent de optimale matrix & transfers..."):
                                new_res, new_plan = solve_sporza_dynamic(df, available_races, t_moments, new_force_base, new_ban_base, [])
                                
                                if new_res:
                                    st.session_state.sporza_selected_riders = new_res
                                    st.session_state.sporza_transfer_plan = new_plan
                                    st.rerun()
                                else:
                                    st.error("Wissel geweigerd! Budget (120M) overschreden of de 4-renners-per-ploeg limiet is gebroken.")
                        else:
                            st.error(f"Selecteer exact {len(to_replace)} vervanger(s). Je hebt er nu {len(to_add)} gekozen.")

    with tab2:
        st.header("🗓️ Startlijst & Uitslagen")
        if toon_uitslagen:
            st.success("✅ Actuele uitslagen ingeladen! Top 20 finishes worden beloond met een medaille (🏅).")
        else:
            st.write("De AI selecteert per koers maximaal 12 renners als 'Starter' (✅). De rest zit op de 'Bank' (🪑).")
        
        matrix_df = current_df[['Renner', 'Prijs', 'Rol'] + available_races].set_index('Renner')
        active_matrix = matrix_df.copy()

        for r in current_df['Renner']:
            rol = current_df.loc[current_df['Renner'] == r, 'Rol'].values[0]
            if 'Verkocht na' in rol:
                moment = rol.replace('Verkocht na ', '')
                if moment in available_races:
                    idx = available_races.index(moment) + 1
                    active_matrix.loc[r, available_races[idx:]] = 0
            elif 'Gekocht na' in rol:
                moment = rol.replace('Gekocht na ', '')
                if moment in available_races:
                    idx = available_races.index(moment) + 1
                    active_matrix.loc[r, available_races[:idx]] = 0

        display_matrix = active_matrix[available_races].applymap(lambda x: ' ' if x == 0 else x)
        display_matrix.insert(0, 'Rol', matrix_df['Rol'])
        
        df_uitslagen = pd.DataFrame()
        if toon_uitslagen:
            u_time = get_file_mod_time("uitslagen.csv")
            df_uitslagen = get_uitslagen(u_time, df['Renner'].tolist())
            
        verreden_koersen = df_uitslagen['Race'].unique() if not df_uitslagen.empty else []

        totals_dict = {}
        kopmannen_dict = {}
        for c in available_races:
            active_riders_in_race = [r for r in active_matrix.index if active_matrix.loc[r, c] == 1]
            starters_df = current_df[current_df['Renner'].isin(active_riders_in_race)].sort_values(by=f'EV_{c}', ascending=False).head(12)
            starters_names = starters_df['Renner'].values
            
            kopman = starters_names[0] if len(starters_names) > 0 else "-"
            kopmannen_dict[c] = kopman
            totals_dict[c] = str(len(starters_names))
            
            is_verreden = c in verreden_koersen
            df_k = df_uitslagen[df_uitslagen['Race'] == c] if is_verreden else pd.DataFrame()
            
            for r in active_riders_in_race:
                if is_verreden:
                    res = df_k[df_k['Renner'] == r]
                    if not res.empty:
                        rank_str = res['Rnk'].values[0]
                        if str(rank_str).isdigit() and int(rank_str) <= 20:
                            display_matrix.loc[r, c] = f"🏅 {rank_str}"
                        else:
                            display_matrix.loc[r, c] = str(rank_str)
                    else:
                        display_matrix.loc[r, c] = "❌ DNF"
                else:
                    if r in starters_names: display_matrix.loc[r, c] = '✅'
                    else: display_matrix.loc[r, c] = '🪑'

        for t in st.session_state.sporza_transfer_plan:
            moment = t['moment']
            if moment in display_matrix.columns and f'🔁 {moment}' not in display_matrix.columns:
                idx = display_matrix.columns.get_loc(moment) + 1
                display_matrix.insert(idx, f'🔁 {moment}', '|')

        def color_rows(row):
            if 'Verkocht' in row['Rol']: return ['background-color: rgba(255, 99, 71, 0.2)'] * len(row)
            if 'Gekocht' in row['Rol']: return ['background-color: rgba(144, 238, 144, 0.2)'] * len(row)
            return [''] * len(row)

        totals_df = pd.DataFrame([totals_dict], index=['🚀 AANTAL STARTERS (Max 12)'])
        st.dataframe(totals_df, use_container_width=True)
        st.dataframe(display_matrix.style.apply(color_rows, axis=1), use_container_width=True)

    with tab3:
        st.header("👑 Kopmannen Advies")
        st.write("In Sporza kies je slechts **1 kopman** per koers voor bonuspunten. Hier is de beste keuze uit jouw geselecteerde 12 starters.")
        
        kop_res = []
        for c in available_races:
            active_riders_in_race = [r for r in active_matrix.index if active_matrix.loc[r, c] == 1]
            starters_df = current_df[current_df['Renner'].isin(active_riders_in_race)].sort_values(by=f'EV_{c}', ascending=False).head(12)
            top = starters_df['Renner'].tolist()
            
            kop_res.append({
                "Koers": c, 
                "👑 Absolute Kopman": top[0] if len(top)>0 else "-", 
                "Alternatief 1": top[1] if len(top)>1 else "-",
                "Alternatief 2": top[2] if len(top)>2 else "-"
            })
        st.dataframe(pd.DataFrame(kop_res), hide_index=True, use_container_width=True)

with tab4:
    st.header("📋 Database: Alle Renners (Sporza Prijzen)")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1: search_name = st.text_input("🔍 Zoek op naam of Ploeg:")
    with col_f2: price_filter = st.slider("💰 Prijs range (Miljoen)", int(df['Prijs'].min()), int(df['Prijs'].max()), (int(df['Prijs'].min()), int(df['Prijs'].max())), 1)
    with col_f3: race_filter = st.multiselect("🏁 Rijdt geselecteerde koersen:", options=available_races)

    f_df = df.copy()
    if search_name: f_df = f_df[f_df['Renner'].str.contains(search_name, case=False, na=False) | f_df['Team'].str.contains(search_name, case=False, na=False)]
    f_df = f_df[(f_df['Prijs'] >= price_filter[0]) & (f_df['Prijs'] <= price_filter[1])]
    if race_filter: f_df = f_df[f_df[race_filter].sum(axis=1) == len(race_filter)]

    d_df = f_df[['Renner', 'Team', 'Prijs', 'Waarde (EV/M)', 'Type', 'Sporza_EV'] + available_races].copy()
    d_df['Prijs'] = d_df['Prijs'].apply(lambda x: f"€ {x}M")
    d_df[available_races] = d_df[available_races].applymap(lambda x: '✅' if x == 1 else '-')
    
    st.dataframe(d_df.sort_values(by='Sporza_EV', ascending=False), use_container_width=True, hide_index=True)

with tab5:
    st.header("ℹ️ Uitgebreide Handleiding & AI Uitleg")

    st.markdown("""
    Deze applicatie gebruikt wiskundige optimalisatie (Integer Linear Programming) om het beruchte *Knapsack Problem* (rugzakprobleem) op te lossen. Voor de **Sporza Wielermanager** is dit extreem complex, omdat de AI niet alleen 20 renners moet kiezen, maar ook per koers moet bepalen wie er op de bank zit en of transfers de budgetboete waard zijn.

    Hieronder vind je een gedetailleerde uitleg van de werking en hoe je de tool optimaal gebruikt.

    ---

    ### 🧠 1. Hoe berekent de AI de waarde van een renner? (Expected Value)
    Sporza deelt punten anders uit dan andere spellen. De AI berekent de **Expected Value (EV)** per koers op basis van:
    * **Statistieken & Profiel:** Elke koers heeft een specifiek profiel (Kassei, Heuvel, Sprint, Allround). De AI kijkt naar de bijbehorende skill van de renner.
    * **Koers Categorieën:** Sporza maakt onderscheid in punten: *Monumenten* (max 125pt), *WorldTour* (max 100pt) en *Niet-WT* (max 80pt). De EV past zich hierop aan.
    * **Kopman Bonus:** Bij Sporza krijgt de kopman vaste bonuspunten als hij top 6 rijdt (+30, +25, etc.), geen vermenigvuldiger (zoals x2). De AI neemt de statistische kans hierop mee in de berekening.
    * **Rekenmodellen:** Je kunt in de zijbalk kiezen hoe agressief de AI de statistieken vertaalt naar punten (bijv. de Originele Curve).

    ---

    ### 🪑 2. De 12-Starters Regel (Bankzitters)
    In Sporza mag je 20 renners in je team hebben, maar **slechts 12 mogen er daadwerkelijk starten** per koers. 
    De AI berekent volautomatisch wie jouw 12 beste renners zijn voor bijvoorbeeld de Ronde van Vlaanderen, en zet de overige renners op de Bank (🪑). Punten van bankzitters tellen niet mee. Hierdoor 'weet' de AI dat een té brede selectie zonde van het budget is en zal hij vaker investeren in absolute piekmomenten.

    ---

    ### 🔁 3. Het Transfer- & Boetesysteem
    Je kunt in de zijbalk tot wel 5 transfers vooruit plannen. De wiskundige solver weet precies wat dit kost:
    * **Transfer 1 t/m 3:** Gratis (Totaalbudget voor je actieve team blijft 120 Miljoen)
    * **Transfer 4:** Kost 1 Miljoen (Totaalbudget zakt naar 119 Miljoen)
    * **Transfer 5:** Kost nog eens 2 Miljoen extra (Totaalbudget zakt naar 117 Miljoen)
    
    De AI berekent tegelijkertijd je start-team én al je geplande wissels in de toekomst. Hij weegt af of het rendabel is om budget in leveren voor een extra transfer, en zorgt dat je na *elke* wissel altijd een geldig team overhoudt.

    ---

    ### 🛡️ 4. Strikte Ploeglimieten
    De regel in Sporza luidt: **Maximaal 4 renners per wielerploeg** (bijv. max 4 van Visma of UAE). De wiskundige solver controleert dit niet alleen bij de start, maar dwingt dit ook af over je hele transfer-tijdlijn. Je kunt dus nooit per ongeluk een 5e Alpecin renner inkopen.

    ---

    ### 🛠️ 5. Finetuning & Handmatige Ingrepen
    Soms wil je de algoritmes overrulen met je eigen wielerkennis:
    * **Team Finetuner (Dashboard):** In het hoofdscherm kun je handmatig een geselecteerde renner aanklikken om te verwijderen. De AI toont direct de 5 beste alternatieven die je je nog kunt veroorloven en kijkt zelfs of de toekomstige geplande transfers en de 'max-4' regel nog wel wiskundig passen.
    * **Renners Forceren:** Via 'Moet in start-team' (zijbalk) dwing je de AI om een renner te kopen.
    * **Renners Uitsluiten:** Geloof je niet in de vorm van een renner? Gebruik 'Niet in start-team' of 'Compleet negeren' om de AI te dwingen een alternatief te zoeken.

    ---

    ### 💾 6. Back-ups & Data Exporteren (Inladen en Opslaan)
    * **Opslaan (Back-up maken):** Sla je team op in de cloud via de knop linksboven in de zijbalk, of download lokaal via '.JSON'. 
    * **Inladen (Team terughalen):** Laad je team in via de cloud-knop of upload een `.json` bestand. Het script controleert automatisch via *Fuzzy Matching* of oude namen nog correct in de database staan.
    """)
