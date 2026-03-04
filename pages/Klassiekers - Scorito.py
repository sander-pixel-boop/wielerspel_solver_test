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

# --- CONFIGURATIE ---
st.set_page_config(page_title="Scorito Klassiekers AI", layout="wide", page_icon="🏆")

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
                # Vertaal Sporza afkortingen naar Scorito voor de sidebar check
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
            
        sporza_naar_scorito_map = {
            'OML': 'OHN', 'STR': 'SB', 'RVB': 'BDP', 
            'IFF': 'GW', 'BRP': 'BP', 'AGT': 'AGR', 'WAP': 'WP'
        }
            
        uitslag_parsed = []
        for index, row in df_raw_uitslagen.iterrows():
            koers_origineel = str(row['Race']).strip().upper()
            koers = sporza_naar_scorito_map.get(koers_origineel, koers_origineel)
            
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

def evaluate_plan_ev(df_eval, base_team, plan, available_races):
    current_active = set(base_team)
    totaal = 0
    for race in available_races:
        for t in plan:
            if t['moment'] == race:
                if t['uit'] in current_active: current_active.remove(t['uit'])
                current_active.add(t['in'])
        for r in current_active:
            totaal += df_eval.loc[df_eval['Renner'] == r, f'EV_{race}'].values[0]
    return totaal

# --- DATA LADEN (PROGRAMMA VAN SPORZA, PRIJS VAN SCORITO) ---
@st.cache_data
def load_and_merge_data(prog_mod_time, scorito_mod_time, stats_mod_time):
    try:
        # 1. Haal programma uit Sporza bestand
        df_prog = pd.read_csv("sporza_prijzen_startlijst.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
        df_prog.columns = df_prog.columns.str.strip()
        if 'Naam' in df_prog.columns and 'Renner' not in df_prog.columns:
            df_prog = df_prog.rename(columns={'Naam': 'Renner'})
        
        # Verwijder de Sporza prijs
        if 'Prijs' in df_prog.columns:
            df_prog = df_prog.drop(columns=['Prijs'])
            
        # Vertaal de koersnamen van Sporza naar Scorito
        sporza_to_scorito = {'OML': 'OHN', 'STR': 'SB', 'RVB': 'BDP', 'IFF': 'GW', 'BRP': 'BP', 'AGT': 'AGR', 'WAP': 'WP'}
        df_prog = df_prog.rename(columns=sporza_to_scorito)
        
        # 2. Haal de Scorito prijzen uit bron_startlijsten.csv
        df_scorito = pd.read_csv("bron_startlijsten.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
        df_scorito.columns = df_scorito.columns.str.strip()
        if 'Naam' in df_scorito.columns and 'Renner' not in df_scorito.columns:
            df_scorito = df_scorito.rename(columns={'Naam': 'Renner'})
            
        if 'Prijs' not in df_scorito.columns and df_scorito['Renner'].astype(str).str.contains(r'\(.*\)', regex=True).any():
            extracted = df_scorito['Renner'].str.extract(r'^(.*?)\s*\(([\d\.]+)[Mm]\)')
            df_scorito['Renner'] = extracted[0].str.strip()
            df_scorito['Prijs'] = pd.to_numeric(extracted[1], errors='coerce') * 1000000

        if 'Prijs' in df_scorito.columns:
            df_scorito['Prijs'] = df_scorito['Prijs'].fillna(0)
            df_scorito.loc[df_scorito['Prijs'] == 800000, 'Prijs'] = 750000
            
        df_prijzen = df_scorito[['Renner', 'Prijs']].drop_duplicates(subset=['Renner'])

        # 3. Haal statistieken uit renners_stats.csv
        df_stats = pd.read_csv("renners_stats.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip') 
        df_stats.columns = df_stats.columns.str.strip()
        if 'Naam' in df_stats.columns and 'Renner' not in df_stats.columns:
            df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        
        if 'Team' not in df_stats.columns and 'Ploeg' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Ploeg': 'Team'})
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        
        # Fuzzy Matching Setup
        scorito_names = df_prijzen['Renner'].unique()
        stats_names = df_stats['Renner'].unique()
        
        norm_to_scorito = {normalize_name_logic(n): n for n in scorito_names}
        norm_scorito_list = list(norm_to_scorito.keys())
        
        norm_to_stats = {normalize_name_logic(n): n for n in stats_names}
        norm_stats_list = list(norm_to_stats.keys())

        df_prog['Renner_Scorito'] = df_prog['Renner']
        df_prog['Renner_Stats'] = df_prog['Renner']

        for i, row in df_prog.iterrows():
            short = row['Renner']
            # Match met Scorito Prijs
            best_scorito = process.extractOne(normalize_name_logic(short), norm_scorito_list, scorer=fuzz.token_set_ratio)
            if best_scorito and best_scorito[1] > 75:
                df_prog.at[i, 'Renner_Scorito'] = norm_to_scorito[best_scorito[0]]
            
            # Match met Stats
            best_stats = process.extractOne(normalize_name_logic(short), norm_stats_list, scorer=fuzz.token_set_ratio)
            if best_stats and best_stats[1] > 75:
                df_prog.at[i, 'Renner_Stats'] = norm_to_stats[best_stats[0]]
                
        # Mergen van de 3 bronnen
        merged_df = pd.merge(df_prog, df_prijzen, left_on='Renner_Scorito', right_on='Renner', how='left', suffixes=('', '_drop1'))
        merged_df = pd.merge(merged_df, df_stats, left_on='Renner_Stats', right_on='Renner', how='left', suffixes=('', '_drop2'))
        
        # Opruimen kolommen
        drop_cols = [c for c in merged_df.columns if '_drop' in c or c in ['Renner_Scorito', 'Renner_Stats']]
        merged_df = merged_df.drop(columns=drop_cols)
        
        # Alleen renners behouden die een Scorito prijs hebben (Prijs > 0)
        merged_df['Prijs'] = pd.to_numeric(merged_df['Prijs'], errors='coerce').fillna(0).astype(int)
        merged_df = merged_df[merged_df['Prijs'] > 0]
        
        merged_df = merged_df.sort_values(by='Prijs', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=['Renner'], keep='first')
        
        ALLE_KOERSEN = ['OHN', 'KBK', 'SB', 'PN', 'TA', 'MSR', 'BDP', 'E3', 'GW', 'DDV', 'RVV', 'SP', 'PR', 'BP', 'AGR', 'WP', 'LBL']
        available_races = [k for k in ALLE_KOERSEN if k in merged_df.columns]
        
        all_stats_cols = ['COB', 'HLL', 'SPR', 'AVG', 'FLT', 'MTN', 'ITT', 'GC', 'OR', 'TTL']
        for col in available_races + all_stats_cols:
            if col not in merged_df.columns:
                merged_df[col] = 0
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
            
        merged_df['HLL/MTN'] = merged_df[['HLL', 'MTN']].max(axis=1).astype(int)
            
        if 'Team' not in merged_df.columns:
            merged_df['Team'] = 'Onbekend'
        else:
            merged_df['Team'] = merged_df['Team'].fillna('Onbekend')
        
        merged_df['Total_Races'] = merged_df[available_races].sum(axis=1).astype(int)
        koers_stat_map = {'OHN':'COB','KBK':'SPR','SB':'HLL','PN':'HLL/MTN','TA':'SPR','MSR':'AVG','BDP':'SPR','E3':'COB','GW':'SPR','DDV':'COB','RVV':'COB','SP':'SPR','PR':'COB','BP':'HLL','AGR':'HLL','WP':'HLL','LBL':'HLL'}
        
        return merged_df, available_races, koers_stat_map
    except Exception as e:
        st.error(f"Fout in dataverwerking: {e}")
        return pd.DataFrame(), [], {}

def calculate_dynamic_ev(df, available_races, koers_stat_map, method, skip_races=[]):
    df = df.copy()
    scorito_pts = [100, 90, 80, 72, 64, 58, 52, 46, 40, 36, 32, 28, 24, 20, 16, 14, 12, 10, 8, 6]
    
    race_evs = {}
    for koers in available_races:
        stat = koers_stat_map.get(koers, 'AVG')
        starters = df[df[koers] == 1].copy()
        starters = starters.sort_values(by=[stat, 'AVG'], ascending=[False, False])
        
        race_ev = pd.Series(0.0, index=df.index)
        
        if koers not in skip_races:
            for i, idx in enumerate(starters.index):
                val = 0.0
                if "Scorito Ranking" in method:
                    val = scorito_pts[i] if i < len(scorito_pts) else 0.0
                elif "Originele Curve" in method:
                    val = (starters.loc[idx, stat] / 100)**4 * 100
                elif "Extreme Curve" in method:
                    val = (starters.loc[idx, stat] / 100)**10 * 100
                elif "Tiers" in method:
                    if i < 3: val = 80.0
                    elif i < 8: val = 45.0
                    elif i < 15: val = 20.0
                    else: val = 0.0
                    
                if i == 0: val *= 3.0
                elif i == 1: val *= 2.5
                elif i == 2: val *= 2.0
                race_ev.loc[idx] = val
        
        race_evs[koers] = race_ev
        df[f'EV_{koers}'] = race_ev

    df['EV_all'] = sum(race_evs.values()) if race_evs else 0.0
    df['Scorito_EV'] = df['EV_all'].fillna(0).round(0).astype(int)
    df['Waarde (EV/M)'] = (df['Scorito_EV'] / (df['Prijs'] / 1000000)).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)
    
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

# --- HOOFD SOLVER ---
def solve_knapsack_dynamic(df, total_budget, min_budget, max_riders, force_base, ban_base, exclude_list):
    prob = pulp.LpProblem("Scorito_Solver", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Base", df.index, cat='Binary')
    
    prob += pulp.lpSum([df.loc[i, 'EV_all'] * x[i] for i in df.index])
    prob += pulp.lpSum([x[i] for i in df.index]) == max_riders
    prob += pulp.lpSum([df.loc[i, 'Prijs'] * x[i] for i in df.index]) <= total_budget
    prob += pulp.lpSum([df.loc[i, 'Prijs'] * x[i] for i in df.index]) >= min_budget
    
    for i in df.index:
        renner = df.loc[i, 'Renner']
        if renner in force_base: prob += x[i] == 1
        if renner in ban_base: prob += x[i] == 0
        if renner in exclude_list: prob += x[i] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=15))
    if pulp.LpStatus[prob.status] == 'Optimal':
        return [df.loc[i, 'Renner'] for i in df.index if x[i].varValue > 0.5]
    return []

# --- NOODWISSEL SOLVER ---
def find_emergency_replacements(df_eval, base_team, transfer_plan, injured_riders, last_race, max_budget, available_races):
    all_historical_riders = set(base_team + [t['in'] for t in transfer_plan])
    candidates = df_eval[~df_eval['Renner'].isin(all_historical_riders)].copy()
    
    idx = available_races.index(last_race)
    remaining_races = available_races[idx+1:]
    candidates['EV_remaining'] = candidates[[f'EV_{r}' for r in remaining_races]].sum(axis=1)
    
    prob = pulp.LpProblem("Noodwissel", pulp.LpMaximize)
    vars = pulp.LpVariable.dicts("C", candidates.index, cat='Binary')
    
    prob += pulp.lpSum([candidates.loc[i, 'EV_remaining'] * vars[i] for i in candidates.index])
    prob += pulp.lpSum([vars[i] for i in candidates.index]) == len(injured_riders)
    
    current_team = set(base_team)
    for race in available_races[:idx+1]:
        for t in transfer_plan:
            if t['moment'] == race:
                if t['uit'] in current_team: current_team.remove(t['uit'])
                current_team.add(t['in'])
                
    for inj in injured_riders:
        if inj in current_team:
            current_team.remove(inj)
            
    base_cost_now = df_eval[df_eval['Renner'].isin(current_team)]['Prijs'].sum()
    prob += base_cost_now + pulp.lpSum([candidates.loc[i, 'Prijs'] * vars[i] for i in candidates.index]) <= max_budget
    
    temp_team = set(current_team)
    for race in available_races[idx+1:]:
        changed = False
        for t in transfer_plan:
            if t['moment'] == race:
                if t['uit'] in temp_team: temp_team.remove(t['uit'])
                temp_team.add(t['in'])
                changed = True
        if changed:
            cost_future = df_eval[df_eval['Renner'].isin(temp_team)]['Prijs'].sum()
            prob += cost_future + pulp.lpSum([candidates.loc[i, 'Prijs'] * vars[i] for i in candidates.index]) <= max_budget
            
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        return [candidates.loc[i, 'Renner'] for i in candidates.index if vars[i].varValue > 0.5]
    return []

# --- TEAM REBUILDER (VOOR FINETUNER) ---
def rebuild_team_and_transfers(df, max_bud, min_bud, max_ren, new_base_team, t_moments, use_transfers):
    prob = pulp.LpProblem("Scorito_Rebuild_Solver", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Base", df.index, cat='Binary')
    
    if use_transfers:
        y_vars = [pulp.LpVariable.dicts(f"Y{k}", df.index, cat='Binary') for k in range(3)]
        z_vars = [pulp.LpVariable.dicts(f"Z{k}", df.index, cat='Binary') for k in range(3)]
        
        obj = pulp.lpSum([x[i] * df.loc[i, 'EV_all'] for i in df.index])
        for k in range(3):
            split_idx = available_races.index(t_moments[k]) + 1 if t_moments[k] != 'GEEN' else len(available_races)
            races_before = available_races[:split_idx]
            races_after = available_races[split_idx:]
            
            df[f'EV_y{k}'] = df[[f'EV_{r}' for r in races_before]].sum(axis=1) if races_before else 0.0
            df[f'EV_z{k}'] = df[[f'EV_{r}' for r in races_after]].sum(axis=1) if races_after else 0.0
            
            obj += pulp.lpSum([y_vars[k][i] * df.loc[i, f'EV_y{k}'] + z_vars[k][i] * df.loc[i, f'EV_z{k}'] for i in df.index])
            
        prob += obj
        
        for i in df.index:
            prob += x[i] + y_vars[0][i] + y_vars[1][i] + y_vars[2][i] + z_vars[0][i] + z_vars[1][i] + z_vars[2][i] <= 1
            renner = df.loc[i, 'Renner']
            if renner in new_base_team:
                prob += x[i] + sum([y_vars[k][i] for k in range(3)]) == 1
            else:
                prob += x[i] + sum([y_vars[k][i] for k in range(3)]) == 0

        for k in range(3):
            prob += pulp.lpSum([y_vars[k][i] for i in df.index]) <= 1  
            prob += pulp.lpSum([y_vars[k][i] for i in df.index]) == pulp.lpSum([z_vars[k][i] for i in df.index]) 
            
        prob += pulp.lpSum([x[i] for i in df.index]) + pulp.lpSum([y_vars[0][i] for i in df.index]) + pulp.lpSum([y_vars[1][i] for i in df.index]) + pulp.lpSum([y_vars[2][i] for i in df.index]) == max_ren
        
        prob += pulp.lpSum([(x[i] + y_vars[0][i] + y_vars[1][i] + y_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) <= max_bud
        prob += pulp.lpSum([(x[i] + z_vars[0][i] + y_vars[1][i] + y_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) <= max_bud
        prob += pulp.lpSum([(x[i] + z_vars[0][i] + z_vars[1][i] + y_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) <= max_bud
        prob += pulp.lpSum([(x[i] + z_vars[0][i] + z_vars[1][i] + z_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) <= max_bud

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=15))
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            base_team_res = [df.loc[i, 'Renner'] for i in df.index if x[i].varValue > 0.5]
            transfer_plan_res = []
            for k in range(3):
                uit = [df.loc[i, 'Renner'] for i in df.index if y_vars[k][i].varValue > 0.5]
                erin = [df.loc[i, 'Renner'] for i in df.index if z_vars[k][i].varValue > 0.5]
                if uit and erin:
                    transfer_plan_res.append({"uit": uit[0], "in": erin[0], "moment": t_moments[k]})
                    base_team_res.append(uit[0])
            return base_team_res, transfer_plan_res
        return None, None
    else:
        return new_base_team, []

# --- HOOFDCODE ---
prog_time = get_file_mod_time("sporza_prijzen_startlijst.csv")
scorito_time = get_file_mod_time("bron_startlijsten.csv")
stats_time = get_file_mod_time("renners_stats.csv")

df_raw, available_races, koers_mapping = load_and_merge_data(prog_time, scorito_time, stats_time)

if df_raw.empty:
    st.warning("Data is leeg of kon niet worden geladen.")
    st.stop()

if "selected_riders" not in st.session_state: st.session_state.selected_riders = []
if "transfer_plan" not in st.session_state: st.session_state.transfer_plan = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏆 AI Coach")
    
    verreden_races = get_verreden_koersen()
    actieve_koersen = [k for k in available_races if k in verreden_races]
    skip_races = []
    
    if actieve_koersen:
        st.success(f"✅ Gereden koersen gedetecteerd (t/m {actieve_koersen[-1]})")
        toon_uitslagen = st.checkbox("🏁 Koersen zijn begonnen (Toon uitslagen in Matrix)", value=True)
        if st.checkbox("🔮 Toon alleen Resterende EV", value=True, help="Negeer behaalde punten."):
            skip_races = actieve_koersen
    else:
        toon_uitslagen = False

    ev_method = st.selectbox("🧮 Rekenmodel (EV)", ["1. Scorito Ranking (Dynamisch)", "2. Originele Curve (Macht 4)", "3. Extreme Curve (Macht 10)", "4. Tiers & Spreiding (Realistisch)"])
    use_transfers = st.checkbox("🔁 Bereken met wissel-strategie", value=True)
    
    t_moments = ["GEEN", "GEEN", "GEEN"]
    if use_transfers:
        st.markdown("**🗓️ Wanneer wil je wisselen? (Kies max 3)**")
        default_index = available_races.index('PR') if 'PR' in available_races else len(available_races) - 2
        t1 = st.selectbox("Wissel 1 na:", options=available_races[:-1], index=default_index)
        t2 = st.selectbox("Wissel 2 na:", options=available_races[:-1], index=default_index)
        t3 = st.selectbox("Wissel 3 na:", options=available_races[:-1], index=default_index)
        t_moments = sorted([t1, t2, t3], key=lambda x: available_races.index(x))
    
    with st.expander("⚙️ Budget & Limieten", expanded=False):
        max_ren = st.number_input("Totaal aantal renners", value=20)
        max_bud = st.number_input("Max Budget", value=45000000, step=500000)
        min_bud = st.number_input("Min Budget", value=43000000, step=500000)
        
    df = calculate_dynamic_ev(df_raw, available_races, koers_mapping, ev_method, skip_races)

    with st.expander("🚑 Directe Noodwissel (Blessure)", expanded=False):
        st.info("Valt een renner uit? Vind direct de beste wissel op dít moment en blijf netjes binnen de 3-wissels limiet.")
        
        if len(st.session_state.selected_riders) > 0:
            default_lr_idx = available_races.index(actieve_koersen[-1]) if actieve_koersen else 0
            last_race = st.selectbox("Laatst gereden koers (Moment van wissel):", options=available_races[:-1], index=default_lr_idx)
            
            active_at_moment = list(st.session_state.selected_riders)
            idx_last = available_races.index(last_race)
            for t in st.session_state.transfer_plan:
                if available_races.index(t['moment']) <= idx_last:
                    if t['uit'] in active_at_moment: active_at_moment.remove(t['uit'])
                    if t['in'] not in active_at_moment: active_at_moment.append(t['in'])
                    
            injured_selection = st.multiselect("Geblesseerde renner(s) eruit:", options=active_at_moment)
            
            if injured_selection:
                planned_transfers_copy = list(st.session_state.transfer_plan)
                
                auto_drop = []
                for i, t in enumerate(planned_transfers_copy):
                    if t['uit'] in injured_selection and available_races.index(t['moment']) >= idx_last:
                        auto_drop.append(i)
                        
                for idx in sorted(auto_drop, reverse=True):
                    st.write(f"💡 *Geplande verkoop van {planned_transfers_copy[idx]['uit']} vervalt automatisch ter compensatie.*")
                    planned_transfers_copy.pop(idx)
                    
                drops_needed = (len(planned_transfers_copy) + len(injured_selection)) - 3
                
                ai_auto_drop = False
                drop_choices = []
                if drops_needed > 0:
                    st.warning(f"🚨 Je overschrijdt de limiet (max 3). Er moeten {drops_needed} geplande wissel(s) vervallen.")
                    ai_auto_drop = st.checkbox("🤖 Laat de AI de minst pijnlijke wissel(s) opofferen", value=True)
                    if not ai_auto_drop:
                        opts = {i: f"{t['uit']} -> {t['in']} (na {t['moment']})" for i, t in enumerate(planned_transfers_copy)}
                        drop_choices = st.multiselect("Selecteer handmatig welke wissel(s) moeten vervallen:", options=list(opts.keys()), format_func=lambda x: opts[x], max_selections=drops_needed)
                    
                if st.button("Vind & Voer Wissel Uit", type="primary", use_container_width=True):
                    if drops_needed > 0 and not ai_auto_drop and len(drop_choices) != drops_needed:
                        st.error(f"Selecteer exact {drops_needed} wissel(s) om te annuleren.")
                    else:
                        if drops_needed > 0 and ai_auto_drop:
                            best_ev = -1
                            best_plan = None
                            
                            for drop_indices in itertools.combinations(range(len(planned_transfers_copy)), drops_needed):
                                temp_plan = [t for i, t in enumerate(planned_transfers_copy) if i not in drop_indices]
                                repls = find_emergency_replacements(df, st.session_state.selected_riders, temp_plan, injured_selection, last_race, max_bud, available_races)
                                
                                if repls:
                                    temp_full_plan = temp_plan + [{"uit": u, "in": r, "moment": last_race} for u, r in zip(injured_selection, repls)]
                                    ev = evaluate_plan_ev(df, st.session_state.selected_riders, temp_full_plan, available_races)
                                    if ev > best_ev:
                                        best_ev = ev
                                        best_plan = temp_full_plan
                                        
                            if best_plan:
                                st.session_state.transfer_plan = best_plan
                                st.rerun()
                            else:
                                st.error("Geen budgettaire oplossing gevonden met deze opofferingen.")
                        else:
                            temp_plan = [t for i, t in enumerate(planned_transfers_copy) if i not in drop_choices]
                            replacements = find_emergency_replacements(df, st.session_state.selected_riders, temp_plan, injured_selection, last_race, max_bud, available_races)
                            if replacements:
                                temp_full_plan = temp_plan + [{"uit": u, "in": r, "moment": last_race} for u, r in zip(injured_selection, replacements)]
                                st.session_state.transfer_plan = temp_full_plan
                                st.rerun()
                            else:
                                st.error("Niet genoeg budget voor een geldige vervanger!")
        else:
            st.warning("Je moet eerst een start-team berekenen of inladen.")

    with st.expander("🔒 Renners Forceren / Uitsluiten", expanded=False):
        force_base = st.multiselect("🟢 Moet in start-team:", options=df['Renner'].tolist())
        ban_base = st.multiselect("🔴 Niet in start-team:", options=[r for r in df['Renner'].tolist() if r not in force_base])
        exclude_list = st.multiselect("🚫 Compleet negeren (hele jaar):", options=[r for r in df['Renner'].tolist() if r not in force_base + ban_base])

    st.write("")
    if st.button("🚀 BEREKEN NIEUW START-TEAM", type="secondary", use_container_width=True):
        res = solve_knapsack_dynamic(df, max_bud, min_bud, max_ren, force_base, ban_base, exclude_list)
        if res:
            st.session_state.selected_riders = res
            st.session_state.transfer_plan = [] 
            new_res, new_plan = rebuild_team_and_transfers(df, max_bud, min_bud, max_ren, res, t_moments, use_transfers)
            if new_res:
                st.session_state.selected_riders = new_res
                st.session_state.transfer_plan = new_plan
            st.rerun()
        else:
            st.error("Geen oplossing mogelijk met deze eisen.")

    st.divider()
    with st.expander("📂 Oude Teams Inladen", expanded=False):
        uploaded_file = st.file_uploader("Upload een JSON backup:", type="json")
        if uploaded_file is not None:
            if st.button("Laad Backup in", use_container_width=True):
                try:
                    ld = json.load(uploaded_file)
                    oude_selectie = ld.get("selected_riders", [])
                    oud_plan = ld.get("transfer_plan", [])
                    
                    huidige_renners = df['Renner'].tolist()
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
                    st.rerun()
                except Exception as e:
                    st.error(f"Fout bij inladen: {e}")

st.title("🏆 Voorjaarsklassiekers: Scorito")
st.markdown("**Met dank aan:** [Wielerorakel.nl](https://www.cyclingoracle.com/) | [Kopmanpuzzel](https://kopmanpuzzel.up.railway.app/)")
st.divider()

# --- TAB STRUCTUUR OPZETTEN ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Jouw Team & Transfers", "🗓️ Startlijst & Uitslagen", "📊 Kopmannen", "📋 Database (Alle)", "ℹ️ Uitleg"])

if not st.session_state.selected_riders:
    with tab1:
        st.info("👈 Kies je instellingen in de zijbalk en klik op **Bereken Nieuw Start-Team** of **Oude Teams Inladen** om te beginnen!")
    with tab2:
        st.info("👈 Bereken eerst een team om de startlijst matrix te kunnen zien.")
    with tab3:
        st.info("👈 Bereken eerst een team voor het kopmannen advies.")
else:
    all_display_riders = list(set(st.session_state.selected_riders + [t['in'] for t in st.session_state.transfer_plan]))
    current_df = df[df['Renner'].isin(all_display_riders)].copy()

    def bepaal_rol_en_moment(naam):
        for t in st.session_state.transfer_plan:
            if naam == t['uit']: return f"Verkocht na {t['moment']}"
            if naam == t['in']: return f"Gekocht na {t['moment']}"
        return 'Basis (Blijft)'

    current_df['Rol'] = current_df['Renner'].apply(bepaal_rol_en_moment)
    current_df['Type'] = current_df.apply(bepaal_klassieker_type, axis=1)

    matrix_df = current_df[['Renner', 'Rol', 'Type', 'Prijs'] + available_races].set_index('Renner')
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

    with tab1:
        st.subheader("📊 Dashboard")
        
        start_team_df = current_df[current_df['Renner'].isin(st.session_state.selected_riders)]
        start_cost = start_team_df['Prijs'].sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Budget over (Start)", f"€ {max_bud - start_cost:,.0f}")
        m2.metric("🚴 Renners (Start)", f"{len(st.session_state.selected_riders)} / {max_ren}")
        
        totaal_ev = evaluate_plan_ev(df, st.session_state.selected_riders, st.session_state.transfer_plan, available_races)
                    
        m3.metric("🎯 Team EV (Totaal)", f"{totaal_ev:.0f}")
        st.divider()
        
        col_t1, col_t2 = st.columns([1, 1], gap="large")
        with col_t1:
            st.markdown("**🛡️ Jouw Start-Team**")
            st.dataframe(start_team_df[['Renner', 'Prijs', 'Type', 'Rol']].sort_values(by='Prijs', ascending=False), hide_index=True, use_container_width=True)
        
        with col_t2:
            c_tr_head, c_tr_btn = st.columns([3,1])
            with c_tr_head: st.markdown(f"**🔁 Transfer Plan ({len(st.session_state.transfer_plan)}/3)**")
            with c_tr_btn:
                if st.session_state.transfer_plan:
                    if st.button("🗑️ Annuleer laatste", use_container_width=True):
                        st.session_state.transfer_plan.pop()
                        st.rerun()

            if not st.session_state.transfer_plan:
                st.info("Nog geen transfers doorgevoerd.")
            else:
                temp_team = list(st.session_state.selected_riders)
                for i, t in enumerate(st.session_state.transfer_plan):
                    if t['uit'] in temp_team: temp_team.remove(t['uit'])
                    if t['in'] not in temp_team: temp_team.append(t['in'])
                    budget_now = max_bud - df[df['Renner'].isin(temp_team)]['Prijs'].sum()
                    
                    st.markdown(f"***Wissel {i+1} (ná {t['moment']} | Resterend budget: €{budget_now/1000000:.2f}M)***")
                    c_uit, c_in = st.columns(2)
                    with c_uit: st.error(f"❌ {t['uit']}")
                    with c_in: st.success(f"📥 {t['in']}")
                    st.write("")

        st.divider()
        
        with st.container(border=True):
            st.subheader("🛠️ Team Finetuner (Start-Team aanpassen)")
            st.markdown("Vervang vóór de start van het spel een renner in je team. De AI past je toekomstige transfers volautomatisch aan als dat nodig is om binnen budget te blijven.")
            
            c_fine1, c_fine2 = st.columns(2)
            with c_fine1: 
                to_replace = st.multiselect("❌ Selecteer renner(s) om te verwijderen:", options=st.session_state.selected_riders)
            
            to_add = []
            if to_replace:
                freed_budget = df[df['Renner'].isin(to_replace)]['Prijs'].sum()
                max_affordable = freed_budget + (max_bud - start_cost)
                
                with c_fine2: 
                    available_replacements = [r for r in df['Renner'].tolist() if r not in st.session_state.selected_riders]
                    to_add_manual = st.multiselect("📥 Handmatige vervanger(s):", options=available_replacements)
                    
                sugg_df = df[~df['Renner'].isin(st.session_state.selected_riders)][df['Prijs'] <= max_affordable].sort_values(by='Scorito_EV', ascending=False).head(5)
                
                sugg_keuze = []
                if not sugg_df.empty:
                    st.info(f"💡 **Top Suggesties (Budget per renner: € {max_affordable/1000000:.1f}M):**")
                    sugg_df['Type'] = sugg_df.apply(bepaal_klassieker_type, axis=1)
                    st.dataframe(sugg_df[['Renner', 'Prijs', 'Waarde (EV/M)', 'Scorito_EV', 'Type']], hide_index=True, use_container_width=True)
                    sugg_keuze = st.multiselect("👉 Of selecteer hier direct een AI-suggestie:", options=sugg_df['Renner'].tolist())

                to_add = list(set(to_add_manual + sugg_keuze))
                
                if to_add:
                    st.markdown("**📊 Vergelijking:**")
                    compare_riders = list(set(to_replace + to_add))
                    compare_df = df[df['Renner'].isin(compare_riders)].copy()
                    compare_cols = ['Renner', 'Prijs', 'Waarde (EV/M)', 'Scorito_EV'] + available_races
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
                        new_force_base = [r for r in st.session_state.selected_riders if r not in to_replace] + to_add
                        
                        if len(new_force_base) == max_ren:
                            new_res, new_plan = rebuild_team_and_transfers(df, max_bud, min_bud, max_ren, new_force_base, t_moments, use_transfers)
                            
                            if new_res:
                                st.session_state.selected_riders = new_res
                                st.session_state.transfer_plan = new_plan
                                st.rerun()
                            else:
                                st.error("Budget overschreden of niet passend met geplande wisselmomenten.")
                        else:
                            st.error(f"Selecteer exact {len(to_replace)} vervanger(s).")
        
        st.divider()
        st.subheader("💾 Exporteer Team")
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            save_data = {"selected_riders": st.session_state.selected_riders, "transfer_plan": st.session_state.transfer_plan}
            st.download_button("📥 Download als .JSON (Backup)", data=json.dumps(save_data), file_name="scorito_team.json", mime="application/json", use_container_width=True)
        with c_dl2:
            export_df = current_df[['Renner', 'Rol', 'Prijs', 'Team', 'Type', 'Waarde (EV/M)', 'Scorito_EV']].copy()
            st.download_button("📊 Download als .CSV (Excel)", data=export_df.to_csv(index=False).encode('utf-8'), file_name="scorito_team.csv", mime="text/csv", use_container_width=True)

    with tab2:
        st.header("🗓️ Startlijst & Uitslagen")
        
        display_matrix = active_matrix[available_races].applymap(lambda x: '✅' if x == 1 else '-')
        
        if toon_uitslagen:
            st.success("✅ Actuele uitslagen ingeladen! Top 20 finishes worden beloond met een medaille (🏅).")
            u_time = get_file_mod_time("uitslagen.csv")
            df_uitslagen = get_uitslagen(u_time, df['Renner'].tolist())
            verreden_koersen = df_uitslagen['Race'].unique() if not df_uitslagen.empty else []
            
            for c in available_races:
                if c in verreden_koersen:
                    df_k = df_uitslagen[df_uitslagen['Race'] == c]
                    for r in display_matrix.index:
                        if display_matrix.loc[r, c] == '✅':
                            res = df_k[df_k['Renner'] == r]
                            if not res.empty:
                                rank_str = res['Rnk'].values[0]
                                if str(rank_str).isdigit() and int(rank_str) <= 20:
                                    display_matrix.loc[r, c] = f"🏅 {rank_str}"
                                else:
                                    display_matrix.loc[r, c] = str(rank_str)
                            else:
                                display_matrix.loc[r, c] = "❌ DNF"

        display_matrix.insert(0, 'Rol', matrix_df['Rol'])
        display_matrix.insert(1, 'Type', matrix_df['Type'])
        
        for t in st.session_state.transfer_plan:
            moment = t['moment']
            if moment in display_matrix.columns and f'🔁 {moment}' not in display_matrix.columns:
                idx = display_matrix.columns.get_loc(moment) + 1
                display_matrix.insert(idx, f'🔁 {moment}', '|')
        
        def color_rows(row):
            if 'Verkocht' in row['Rol']: return ['background-color: rgba(255, 99, 71, 0.2)'] * len(row)
            if 'Gekocht' in row['Rol']: return ['background-color: rgba(144, 238, 144, 0.2)'] * len(row)
            return [''] * len(row)

        st.dataframe(display_matrix.style.apply(color_rows, axis=1), use_container_width=True)

    with tab3:
        st.header("📊 Kopmannen Advies")
        kop_res = []
        type_vertaling = {'COB': 'Kassei', 'SPR': 'Sprint', 'HLL': 'Heuvel', 'MTN': 'Klimmer', 'GC': 'Klassement', 'AVG': 'Allround', 'HLL/MTN': 'Heuvel/Klimmer'}
        
        for c in available_races:
            starters = active_matrix[active_matrix[c] == 1]
            if not starters.empty:
                stat = koers_mapping.get(c, 'AVG')
                koers_type = type_vertaling.get(stat, stat)
                top = current_df[current_df['Renner'].isin(starters.index)].sort_values(by=[stat, 'AVG'], ascending=False)['Renner'].tolist()
                kop_res.append({
                    "Koers": c, "Type": koers_type, 
                    "🥇 Kopman 1": top[0] if len(top)>0 else "-", 
                    "🥈 Kopman 2": top[1] if len(top)>1 else "-", 
                    "🥉 Kopman 3": top[2] if len(top)>2 else "-"
                })
        st.dataframe(pd.DataFrame(kop_res), hide_index=True, use_container_width=True)

with tab4:
    st.header("📋 Database: Alle Renners")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1: search_name = st.text_input("🔍 Zoek op naam of Ploeg:")
    with col_f2: price_filter = st.slider("💰 Prijs range", int(df['Prijs'].min()), int(df['Prijs'].max()), (int(df['Prijs'].min()), int(df['Prijs'].max())), 250000)
    with col_f3: race_filter = st.multiselect("🏁 Rijdt geselecteerde koersen:", options=available_races)

    f_df = df.copy()
    f_df['Type'] = f_df.apply(bepaal_klassieker_type, axis=1)
    if search_name: f_df = f_df[f_df['Renner'].str.contains(search_name, case=False, na=False) | f_df['Team'].str.contains(search_name, case=False, na=False)]
    f_df = f_df[(f_df['Prijs'] >= price_filter[0]) & (f_df['Prijs'] <= price_filter[1])]
    if race_filter: f_df = f_df[f_df[race_filter].sum(axis=1) == len(race_filter)]

    d_df = f_df[['Renner', 'Team', 'Prijs', 'Waarde (EV/M)', 'Type', 'Scorito_EV'] + available_races].copy()
    d_df['Prijs'] = d_df['Prijs'].apply(lambda x: f"€ {x/1000000:.2f}M")
    d_df[available_races] = d_df[available_races].applymap(lambda x: '✅' if x == 1 else '-')
    
    st.dataframe(d_df.sort_values(by='Scorito_EV', ascending=False), use_container_width=True, hide_index=True)

with tab5:
    st.header("ℹ️ Uitgebreide Handleiding & AI Uitleg")
    
    st.markdown("""
    Deze applicatie gebruikt wiskundige optimalisatie (Integer Linear Programming) om het beruchte *Knapsack Problem* (rugzakprobleem) op te lossen. Het doel is simpel: bouw een team met de hoogst mogelijke verwachte punten, zonder de budget- en spellimieten te overschrijden.

    Hieronder vind je een gedetailleerde uitleg van de werking en hoe je de tool optimaal gebruikt.

    ---

    ### 🧠 1. Hoe berekent de AI de waarde van een renner? (Expected Value)
    Scorito draait om punten scoren. De AI voorspelt deze punten via de **Expected Value (EV)**. De EV van een renner wordt per koers opgebouwd uit:
    * **Statistieken & Profiel:** Elke koers heeft een specifiek profiel (Kassei, Heuvel, Sprint, Allround). De AI kijkt naar de bijbehorende skill van de renner (bijv. de `COB`-statistiek voor Roubaix).
    * **Kopman Multipliers:** In Scorito halen kopmannen de meeste punten (3x, 2.5x, 2x). De AI bepaalt per koers volautomatisch wie je 3 beste renners zijn en past deze multipliers toe op hun verwachte punten.
    * **Rekenmodellen:** Je kunt in de zijbalk kiezen hoe agressief de AI de statistieken vertaalt naar punten:
        1. *Scorito Ranking:* Kijkt puur of een renner in de top-20 van een koers kan eindigen en deelt de vaste Scorito-punten uit.
        2. *Originele Curve (Macht 4):* Geeft een exponentiële bonus aan absolute topspecialisten ten opzichte van subtoppers.
        3. *Extreme Curve (Macht 10):* Dwingt de AI om uitsluitend absolute wereldtoppers te selecteren en gokt minder op breedte.
        4. *Tiers:* Verdeelt renners in vaste categorieën (Kopman, outsider, knecht).

    ---

    ### 🔁 2. De Dynamische Wisselstrategie
    In Scorito mag je maximaal 3 renners wisselen. Waar veel spelers standaard na Parijs-Roubaix wisselen, laat deze AI je de **tijdlijn** volledig zelf bepalen.

    * **De 4 Tijdvakken:** Zodra je wisselmomenten instelt (bijv. na KBK en na PR), snapt de AI dat het seizoen is opgedeeld in blokken. 
    * **Sluitende Begroting:** De wiskundige solver garandeert dat je op *geen enkel moment* over de €45 miljoen gaat. De som van je startteam mag max €45M zijn, maar ook de som van je team ná wissel 1, én na wissel 2.

    ---

    ### 🚑 3. Noodwissels (Tijdens het spel)
    Zit je midden in het voorjaar en valt een belangrijke pion weg door een valpartij of ziekte? 
    
    1. Klik in de zijbalk op **🚑 Directe Noodwissel (Blessure)**.
    2. Kies de **laatst verreden koers** (Dit is het moment dat de wissel virtueel wordt doorgevoerd).
    3. Selecteer de geblesseerde renner en klik op Bereken.
    4. De AI leest direct jouw actuele team uit, haalt de geblesseerde renner eruit, kijkt naar je resterende budget, en zoekt binnen 2 seconden de absolute topvervanger die in de **resterende koersen** de meeste punten pakt.
    5. *Let op:* Had je al 3 wissels gepland staan verderop in het seizoen? Dan zal de AI één van die toekomstige wissels automatisch of op jouw verzoek opofferen om onder de limiet van 3 transfers te blijven.

    ---

    ### 🛠️ 4. Finetuning & Handmatige Ingrepen
    Soms wil je de algoritmes overrulen met je eigen wielerkennis:
    * **Team Finetuner (Dashboard):** In het hoofdscherm kun je handmatig een geselecteerde renner aanklikken om te verwijderen. De AI toont direct de 5 beste alternatieven die je je nog kunt veroorloven en kijkt zelfs of de toekomstige geplande transfers nog wel wiskundig passen.
    * **Renners Forceren:** Via 'Moet in start-team' (zijbalk) dwing je de AI om een renner te kopen.
    * **Renners Uitsluiten:** Geloof je niet in de vorm van een renner? Gebruik 'Niet in start-team' of 'Compleet negeren' om de AI te dwingen een alternatief te zoeken.

    ---

    ### 💾 5. Back-ups & Data Exporteren (Inladen en Opslaan)
    Je wilt je berekende selectie niet kwijtraken. Daarom bevat dit dashboard een export- en importfunctie.
    
    **Opslaan (Back-up maken):**
    Onderaan het dashboard (Tab 1: Jouw Team & Transfers) vind je 'Exporteer Team'.
    * **Download als JSON:** Dit is je belangrijkste back-up bestand. Het bevat de harde code van je exacte 20 renners én je geplande wisselmomenten. Sla deze lokaal op.
    * **Download als CSV:** Wil je je team liever in Excel bekijken of delen met vrienden? Download hem dan als `.csv`. Let op: dit is puur een visueel overzicht.
    
    **Inladen (Team terughalen):**
    Wil je de volgende dag verder werken aan je team? 
    1. Ga in de linker zijbalk naar **📂 Oude Teams Inladen**.
    2. Upload hier je bewaarde `.json` bestand.
    3. Het systeem plaatst direct je basis-20 terug in het geheugen en zet eventuele wissels klaar in je Transfer Plan. Het script controleert zelfs volautomatisch of oude namen nog steeds (zonder spelfouten) in de actuele database staan via *Fuzzy Matching*!
    """)
