import streamlit as st
import pandas as pd
import pulp
import json
import plotly.express as px
import plotly.graph_objects as go
import unicodedata
import os
import itertools
from thefuzz import process, fuzz
from supabase import create_client, Client
from datetime import datetime

# --- CONFIGURATIE ---
st.set_page_config(page_title="Scorito Klassiekers AI", layout="wide", page_icon="🏆")

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
TABEL_NAAM = st.secrets["TABEL_NAAM"]

# --- HULPFUNCTIE: NORMALISATIE ---
def normalize_name_logic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- HULPFUNCTIES: BESTANDSCONTROLE & UITSLAGEN ---
def get_file_mod_time(filepath):
    return os.path.getmtime(filepath) if os.path.exists(filepath) else 0

def get_verreden_koersen():
    if os.path.exists("uitslagen.csv"):
        try:
            df_u = pd.read_csv("uitslagen.csv", sep='\t', engine='python')
            if 'Race' not in df_u.columns:
                df_u = pd.read_csv("uitslagen.csv", sep=None, engine='python')
            if 'Race' in df_u.columns:
                sporza_naar_scorito_map = {'OML': 'OHN', 'STR': 'SB', 'RVB': 'BDP', 'IFF': 'GW', 'BRP': 'BP', 'AGT': 'AGR', 'WAP': 'WP'}
                races = [str(x).strip().upper() for x in df_u['Race'].unique()]
                return [sporza_naar_scorito_map.get(r, r) for r in races]
        except:
            pass
    return []

@st.cache_data
def get_uitslagen(file_mod_time, alle_renners):
    if not os.path.exists("uitslagen.csv"):
        return pd.DataFrame()
    try:
        df_raw_uitslagen = pd.read_csv("uitslagen.csv", sep=None, engine='python')
        df_raw_uitslagen.columns = [str(c).strip().title() for c in df_raw_uitslagen.columns]
        
        if 'Race' not in df_raw_uitslagen.columns or 'Rider' not in df_raw_uitslagen.columns or 'Rnk' not in df_raw_uitslagen.columns:
            return pd.DataFrame()
            
        sporza_naar_scorito_map = {'OML': 'OHN', 'STR': 'SB', 'RVB': 'BDP', 'IFF': 'GW', 'BRP': 'BP', 'AGT': 'AGR', 'WAP': 'WP'}
            
        uitslag_parsed = []
        for index, row in df_raw_uitslagen.iterrows():
            koers_origineel = str(row['Race']).strip().upper()
            koers = sporza_naar_scorito_map.get(koers_origineel, koers_origineel)
            rank_str = str(row['Rnk']).strip().upper()
            if rank_str in ['DNS', 'NAN', '']: continue
            rider_name = str(row['Rider']).strip()
            match = process.extractOne(rider_name, alle_renners, scorer=fuzz.token_set_ratio)
            if match and match[1] > 70:
                uitslag_parsed.append({"Race": koers, "Rnk": rank_str, "Renner": match[0]})
        return pd.DataFrame(uitslag_parsed)
    except:
        return pd.DataFrame()

def evaluate_plan_ev(df_eval, base_team, plan, available_races):
    current_active = set(base_team)
    totaal = 0
    for race in available_races:
        for t in plan:
            if t['moment'] == race:
                if t['uit'] in current_active: current_active.remove(t['uit'])
                current_active.add(t['in'])
        for r in current_active:
            res = df_eval.loc[df_eval['Renner'] == r, f'EV_{race}']
            if not res.empty: totaal += res.values[0]
    return totaal

# --- DATA LADEN ---
@st.cache_data
def load_and_merge_data(prog_mod_time, scorito_mod_time, stats_mod_time):
    try:
        df_prog = pd.read_csv("sporza_prijzen_startlijst.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
        df_prog.columns = df_prog.columns.str.strip()
        if 'Naam' in df_prog.columns: df_prog = df_prog.rename(columns={'Naam': 'Renner'})
        if 'Prijs' in df_prog.columns: df_prog = df_prog.drop(columns=['Prijs'])
        sporza_to_scorito = {'OML': 'OHN', 'STR': 'SB', 'RVB': 'BDP', 'IFF': 'GW', 'BRP': 'BP', 'AGT': 'AGR', 'WAP': 'WP'}
        df_prog = df_prog.rename(columns=sporza_to_scorito)
        
        df_scorito = pd.read_csv("bron_startlijsten.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
        df_scorito.columns = df_scorito.columns.str.strip()
        if 'Naam' in df_scorito.columns: df_scorito = df_scorito.rename(columns={'Naam': 'Renner'})
        if 'Prijs' not in df_scorito.columns and df_scorito['Renner'].astype(str).str.contains(r'\(.*\)', regex=True).any():
            extracted = df_scorito['Renner'].str.extract(r'^(.*?)\s*\(([\d\.]+)[Mm]\)')
            df_scorito['Renner'] = extracted[0].str.strip()
            df_scorito['Prijs'] = pd.to_numeric(extracted[1], errors='coerce') * 1000000
        if 'Prijs' in df_scorito.columns:
            df_scorito['Prijs'] = df_scorito['Prijs'].fillna(0)
            df_scorito.loc[df_scorito['Prijs'] == 800000, 'Prijs'] = 750000
        df_prijzen = df_scorito[['Renner', 'Prijs']].drop_duplicates(subset=['Renner'])

        df_stats = pd.read_csv("renners_stats.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip') 
        df_stats.columns = df_stats.columns.str.strip()
        if 'Naam' in df_stats.columns: df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        if 'Team' not in df_stats.columns and 'Ploeg' in df_stats.columns: df_stats = df_stats.rename(columns={'Ploeg': 'Team'})
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        
        scorito_names = df_prijzen['Renner'].unique()
        stats_names = df_stats['Renner'].unique()
        norm_to_scorito = {normalize_name_logic(n): n for n in scorito_names}
        norm_to_stats = {normalize_name_logic(n): n for n in stats_names}

        df_prog['Renner_Scorito'] = df_prog['Renner']
        df_prog['Renner_Stats'] = df_prog['Renner']
        for i, row in df_prog.iterrows():
            short = normalize_name_logic(row['Renner'])
            ms = process.extractOne(short, list(norm_to_scorito.keys()), scorer=fuzz.token_set_ratio)
            if ms and ms[1] > 75: df_prog.at[i, 'Renner_Scorito'] = norm_to_scorito[ms[0]]
            mst = process.extractOne(short, list(norm_to_stats.keys()), scorer=fuzz.token_set_ratio)
            if mst and mst[1] > 75: df_prog.at[i, 'Renner_Stats'] = norm_to_stats[mst[0]]
                
        merged_df = pd.merge(df_prog, df_prijzen, left_on='Renner_Scorito', right_on='Renner', how='left', suffixes=('', '_drop1'))
        merged_df = pd.merge(merged_df, df_stats, left_on='Renner_Stats', right_on='Renner', how='left', suffixes=('', '_drop2'))
        merged_df = merged_df.drop(columns=[c for c in merged_df.columns if '_drop' in c or 'Renner_' in c])
        merged_df['Prijs'] = pd.to_numeric(merged_df['Prijs'], errors='coerce').fillna(0).astype(int)
        merged_df = merged_df[merged_df['Prijs'] > 0].sort_values(by='Prijs', ascending=False).drop_duplicates(subset=['Renner'])
        
        ALLE_KOERSEN = ['OHN', 'KBK', 'SB', 'PN', 'TA', 'MSR', 'BDP', 'E3', 'GW', 'DDV', 'RVV', 'SP', 'PR', 'BP', 'AGR', 'WP', 'LBL']
        available_races = [k for k in ALLE_KOERSEN if k in merged_df.columns]
        for col in available_races + ['COB', 'HLL', 'SPR', 'AVG', 'MTN', 'ITT']:
            if col not in merged_df.columns: merged_df[col] = 0
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
        merged_df['HLL/MTN'] = merged_df[['HLL', 'MTN']].max(axis=1)
        merged_df['Total_Races'] = merged_df[available_races].sum(axis=1)
        koers_stat_map = {'OHN':'COB','KBK':'SPR','SB':'HLL','PN':'HLL/MTN','TA':'SPR','MSR':'AVG','BDP':'SPR','E3':'COB','GW':'SPR','DDV':'COB','RVV':'COB','SP':'SPR','PR':'COB','BP':'HLL','AGR':'HLL','WP':'HLL','LBL':'HLL'}
        return merged_df, available_races, koers_stat_map
    except Exception as e:
        st.error(f"Fout: {e}"); return pd.DataFrame(), [], {}

def calculate_dynamic_ev(df, available_races, koers_stat_map, method, skip_races=[]):
    df = df.copy()
    pts = [100, 90, 80, 72, 64, 58, 52, 46, 40, 36, 32, 28, 24, 20, 16, 14, 12, 10, 8, 6]
    for k in available_races:
        stat = koers_stat_map.get(k, 'AVG')
        starters = df[df[k] == 1].sort_values(by=[stat, 'AVG'], ascending=False)
        ev_col = pd.Series(0.0, index=df.index)
        if k not in skip_races:
            for i, idx in enumerate(starters.index):
                v = 0.0
                if "Scorito" in method:
                    v = pts[i] if i < 20 else 0.0
                elif "Macht 4" in method:
                    v = (starters.loc[idx, stat]/100)**4 * 100
                elif "Macht 10" in method:
                    v = (starters.loc[idx, stat]/100)**10 * 100
                elif "Tiers" in method:
                    if i < 3:
                        v = 80
                    elif i < 8:
                        v = 45
                    elif i < 15:
                        v = 20
                    else:
                        v = 0
                
                # Kopman multipliers
                if i == 0:
                    v *= 3
                elif i == 1:
                    v *= 2.5
                elif i == 2:
                    v *= 2
                    
                ev_col.loc[idx] = v
        df[f'EV_{k}'] = ev_col
    df['EV_all'] = df[[f'EV_{k}' for k in available_races]].sum(axis=1)
    df['Scorito_EV'] = df['EV_all'].round(0).astype(int)
    df['Waarde (EV/M)'] = (df['Scorito_EV'] / (df['Prijs'] / 1000000)).fillna(0).round(1)
    return df

def bepaal_klassieker_type(row):
    c, h, s = row.get('COB', 0), row.get('HLL', 0), row.get('SPR', 0)
    elite = [t for t, v in zip(['Kassei', 'Heuvel', 'Sprint'], [c, h, s]) if v >= 85]
    if len(elite) >= 3: return 'Multispecialist'
    if elite: return ' / '.join(elite)
    stats = {'Kassei': c, 'Heuvel': h, 'Sprint': s, 'Klimmer': row.get('MTN', 0)}
    return max(stats, key=stats.get)

# --- SOLVERS ---
def solve_knapsack_dynamic(df, total_budget, min_budget, max_riders, force, ban, exclude):
    prob = pulp.LpProblem("Scorito", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("B", df.index, cat='Binary')
    prob += pulp.lpSum([df.loc[i, 'EV_all'] * x[i] for i in df.index])
    prob += pulp.lpSum([x[i] for i in df.index]) == max_riders
    prob += pulp.lpSum([df.loc[i, 'Prijs'] * x[i] for i in df.index]) <= total_budget
    prob += pulp.lpSum([df.loc[i, 'Prijs'] * x[i] for i in df.index]) >= min_budget
    for i in df.index:
        r = df.loc[i, 'Renner']
        if r in force: prob += x[i] == 1
        if r in ban or r in exclude: prob += x[i] == 0
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=15))
    return [df.loc[i, 'Renner'] for i in df.index if x[i].varValue > 0.5] if pulp.LpStatus[prob.status] == 'Optimal' else []

def rebuild_team_and_transfers(df, max_bud, min_bud, max_ren, new_base, t_moments, use_tr):
    if not use_tr: return new_base, []
    prob = pulp.LpProblem("Rebuild", pulp.LpMaximize)
    df_idx = df.index; x = pulp.LpVariable.dicts("X", df_idx, cat='Binary')
    y = [pulp.LpVariable.dicts(f"Y{k}", df_idx, cat='Binary') for k in range(3)]
    z = [pulp.LpVariable.dicts(f"Z{k}", df_idx, cat='Binary') for k in range(3)]
    obj = pulp.lpSum([x[i] * df.loc[i, 'EV_all'] for i in df_idx])
    for k in range(3):
        split = available_races.index(t_moments[k]) + 1 if t_moments[k] != 'GEEN' else len(available_races)
        ev_y = df[[f'EV_{r}' for r in available_races[:split]]].sum(axis=1)
        ev_z = df[[f'EV_{r}' for r in available_races[split:]]].sum(axis=1)
        obj += pulp.lpSum([y[k][i] * ev_y[i] + z[k][i] * ev_z[i] for i in df_idx])
    prob += obj
    for i in df_idx:
        prob += x[i] + sum(y[k][i] for k in range(3)) + sum(z[k][i] for k in range(3)) <= 1
        if df.loc[i, 'Renner'] in new_base: prob += x[i] + sum(y[k][i] for k in range(3)) == 1
        else: prob += x[i] + sum(y[k][i] for k in range(3)) == 0
    for k in range(3):
        prob += pulp.lpSum(y[k][i] for i in df_idx) <= 1
        prob += pulp.lpSum(y[k][i] for i in df_idx) == pulp.lpSum(z[k][i] for i in df_idx)
    prob += pulp.lpSum(x[i] for i in df_idx) + sum(pulp.lpSum(y[k][i] for i in df_idx) for k in range(3)) == max_ren
    p = df['Prijs']
    prob += pulp.lpSum((x[i]+y[0][i]+y[1][i]+y[2][i])*p[i] for i in df_idx) <= max_bud
    prob += pulp.lpSum((x[i]+z[0][i]+y[1][i]+y[2][i])*p[i] for i in df_idx) <= max_bud
    prob += pulp.lpSum((x[i]+z[0][i]+z[1][i]+y[2][i])*p[i] for i in df_idx) <= max_bud
    prob += pulp.lpSum((x[i]+z[0][i]+z[1][i]+z[2][i])*p[i] for i in df_idx) <= max_bud
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=15))
    if pulp.LpStatus[prob.status] == 'Optimal':
        bt = [df.loc[i, 'Renner'] for i in df_idx if x[i].varValue > 0.5]
        tp = []
        for k in range(3):
            u = [df.loc[i, 'Renner'] for i in df_idx if y[k][i].varValue > 0.5]
            e = [df.loc[i, 'Renner'] for i in df_idx if z[k][i].varValue > 0.5]
            if u and e: tp.append({"uit": u[0], "in": e[0], "moment": t_moments[k]}); bt.append(u[0])
        return bt, tp
    return None, None

# --- MAIN ---
prog_t = get_file_mod_time("sporza_prijzen_startlijst.csv")
scor_t = get_file_mod_time("bron_startlijsten.csv")
stat_t = get_file_mod_time("renners_stats.csv")
df_raw, available_races, koers_mapping = load_and_merge_data(prog_t, scor_t, stat_t)

if "selected_riders" not in st.session_state: st.session_state.selected_riders = []
if "transfer_plan" not in st.session_state: st.session_state.transfer_plan = []

with st.sidebar:
    st.header(f"👤 Profiel: {speler_naam.capitalize()}")
    
    st.write("☁️ **Cloud Database**")
    c_cloud1, c_cloud2 = st.columns(2)
    with c_cloud1:
        if st.button("💾 Opslaan", type="primary", use_container_width=True):
            try:
                team_data = {"selected_riders": st.session_state.selected_riders, "transfer_plan": st.session_state.transfer_plan, "ts": datetime.now().strftime("%Y-%m-%d %H:%M")}
                supabase.table(TABEL_NAAM).upsert({"username": speler_naam, "scorito_team": team_data}, on_conflict="username").execute()
                st.success("Cloud-backup geslaagd!")
            except Exception as e: st.error(f"Fout: {e}")
    with c_cloud2:
        if st.button("🔄 Inladen", use_container_width=True):
            try:
                res = supabase.table(TABEL_NAAM).select("scorito_team").eq("username", speler_naam).execute()
                if res.data and res.data[0]['scorito_team']:
                    d = res.data[0]['scorito_team']
                    st.session_state.selected_riders = d.get("selected_riders", [])
                    st.session_state.transfer_plan = d.get("transfer_plan", [])
                    st.success(f"Team geladen (van {d.get('ts', '?')})"); st.rerun()
                else: st.warning("Geen team gevonden.")
            except Exception as e: st.error(f"Fout: {e}")

    st.divider()
    st.write("📁 **Lokale Backup (.json)**")
    
    # Exporteren
    save_data = {"selected_riders": st.session_state.selected_riders, "transfer_plan": st.session_state.transfer_plan}
    st.download_button("📥 Download als .JSON", data=json.dumps(save_data), file_name=f"{speler_naam}_scorito_team.json", mime="application/json", use_container_width=True)
    
    # Importeren
    uploaded_file = st.file_uploader("📂 Upload Team (.json)", type="json")
    if uploaded_file is not None and st.button("Laad .json in", use_container_width=True):
        try:
            ld = json.load(uploaded_file)
            oude_selectie = ld.get("selected_riders", [])
            oud_plan = ld.get("transfer_plan", [])
            
            huidige_renners = df_raw['Renner'].tolist()
            def update_naam(naam):
                if naam in huidige_renners: return naam
                match = process.extractOne(naam, huidige_renners, scorer=fuzz.token_set_ratio)
                return match[0] if match and match[1] > 80 else naam

            st.session_state.selected_riders = [update_naam(r) for r in oude_selectie if update_naam(r) in huidige_renners]
            
            nieuw_plan = []
            if isinstance(oud_plan, dict) and "uit" in oud_plan and "in" in oud_plan:
                for r_uit, r_in in zip(oud_plan["uit"], oud_plan["in"]):
                    nieuw_plan.append({"uit": update_naam(r_uit), "in": update_naam(r_in), "moment": "PR"})
            elif isinstance(oud_plan, list):
                for t in oud_plan:
                    nieuw_plan.append({"uit": update_naam(t["uit"]), "in": update_naam(t["in"]), "moment": t["moment"]})

            st.session_state.transfer_plan = nieuw_plan
            st.success("✅ Lokaal bestand geladen!")
            st.rerun()
        except Exception as e:
            st.error(f"Fout bij inladen: {e}")

    st.divider()
    verreden = get_verreden_koersen()
    skip_r = []
    if verreden:
        st.success(f"Koersen bezig (t/m {verreden[-1]})")
        if st.checkbox("🔮 Alleen Resterende EV", value=True): skip_r = [k for k in available_races if k in verreden]
    
    ev_m = st.selectbox("Model", ["1. Scorito Ranking", "2. Macht 4", "3. Macht 10", "4. Tiers"])
    use_tr = st.checkbox("Met wissels", value=True)
    t_moms = ["GEEN"]*3
    if use_tr:
        t1 = st.selectbox("Wissel 1 na:", available_races[:-1], index=available_races.index('PR') if 'PR' in available_races else 0)
        t2 = st.selectbox("Wissel 2 na:", available_races[:-1], index=available_races.index('PR') if 'PR' in available_races else 0)
        t3 = st.selectbox("Wissel 3 na:", available_races[:-1], index=available_races.index('PR') if 'PR' in available_races else 0)
        t_moms = sorted([t1, t2, t3], key=lambda x: available_races.index(x))

    max_ren = st.number_input("Renners", value=20); max_bud = st.number_input("Budget", value=45000000, step=500000)
    df = calculate_dynamic_ev(df_raw, available_races, koers_mapping, ev_m, skip_r)
    
    force = st.multiselect("Forceer:", df['Renner'].tolist())
    ban = st.multiselect("Ban:", [r for r in df['Renner'].tolist() if r not in force])
    
    if st.button("🚀 BEREKEN TEAM", type="secondary", use_container_width=True):
        res = solve_knapsack_dynamic(df, max_bud, 43000000, max_ren, force, ban, [])
        if res:
            bt, tp = rebuild_team_and_transfers(df, max_bud, 43000000, max_ren, res, t_moms, use_tr)
            if bt: st.session_state.selected_riders = bt; st.session_state.transfer_plan = tp; st.rerun()
        else: st.error("Geen oplossing.")

st.title("🏆 Voorjaarsklassiekers: Scorito")
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Team & Dashboard", "🗓️ Matrix", "📊 Kopmannen", "📋 Database"])

if st.session_state.selected_riders:
    curr = df[df['Renner'].isin(list(set(st.session_state.selected_riders + [t['in'] for t in st.session_state.transfer_plan])))].copy()
    curr['Rol'] = curr['Renner'].apply(lambda x: next((f"Wissel na {t['moment']}" for t in st.session_state.transfer_plan if x in [t['uit'], t['in']]), "Basis"))
    
    with tab1:
        ev_tot = evaluate_plan_ev(df, st.session_state.selected_riders, st.session_state.transfer_plan, available_races)
        c1, c2, c3 = st.columns(3)
        c1.metric("Budget over", f"€{max_bud - df[df['Renner'].isin(st.session_state.selected_riders)]['Prijs'].sum():,.0f}")
        c2.metric("Team EV", f"{ev_tot:.0f}")
        c3.metric("Transfers", f"{len(st.session_state.transfer_plan)}/3")
        st.dataframe(curr[['Renner', 'Prijs', 'Rol', 'Scorito_EV']].sort_values('Prijs', ascending=False), hide_index=True, use_container_width=True)

    with tab2:
        m = curr[['Renner'] + available_races].set_index('Renner')
        st.dataframe(m.applymap(lambda x: '✅' if x==1 else '-'), use_container_width=True)
    
    with tab3:
        advies = []
        for r in available_races:
            starters = curr[curr[r]==1].sort_values(by=[koers_mapping.get(r, 'AVG'), 'AVG'], ascending=False)
            top = starters['Renner'].tolist()
            advies.append({"Koers": r, "K1": top[0] if len(top)>0 else "-", "K2": top[1] if len(top)>1 else "-", "K3": top[2] if len(top)>2 else "-"})
        st.dataframe(pd.DataFrame(advies), hide_index=True, use_container_width=True)

with tab4:
    st.dataframe(df[['Renner', 'Team', 'Prijs', 'Scorito_EV', 'Waarde (EV/M)']].sort_values('Scorito_EV', ascending=False), hide_index=True, use_container_width=True)
