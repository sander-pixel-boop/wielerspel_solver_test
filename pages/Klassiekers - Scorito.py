import streamlit as st
import pandas as pd
import pulp
import json
import plotly.express as px
import plotly.graph_objects as go
from thefuzz import process, fuzz

# --- CONFIGURATIE ---
st.set_page_config(page_title="Scorito Klassiekers AI (TEST)", layout="wide", page_icon="🧪")

# --- DATA LADEN (KLASSIEKERS SCORITO) ---
@st.cache_data
def load_and_merge_data():
    try:
        df_prog = pd.read_csv("bron_startlijsten.csv", sep=None, engine='python', on_bad_lines='skip')
        df_prog = df_prog.rename(columns={'RvB': 'BDP', 'IFF': 'GW'})
        
        if 'Prijs' not in df_prog.columns and df_prog['Renner'].astype(str).str.contains(r'\(.*\)', regex=True).any():
            extracted = df_prog['Renner'].str.extract(r'^(.*?)\s*\(([\d\.]+)[Mm]\)')
            df_prog['Renner'] = extracted[0].str.strip()
            df_prog['Prijs'] = pd.to_numeric(extracted[1], errors='coerce') * 1000000
            
        for col in df_prog.columns:
            if col not in ['Renner', 'Prijs']:
                df_prog[col] = df_prog[col].apply(lambda x: 1 if str(x).strip() in ['✓', 'v', 'V', '1', '1.0'] else 0)

        if 'Prijs' in df_prog.columns:
            df_prog['Prijs'] = df_prog['Prijs'].fillna(0)
            df_prog.loc[df_prog['Prijs'] == 800000, 'Prijs'] = 750000
        
        df_stats = pd.read_csv("renners_stats.csv", sep='\t') 
        if 'Naam' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        
        if 'Team' not in df_stats.columns and 'Ploeg' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Ploeg': 'Team'})
            
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        
        short_names = df_prog['Renner'].unique()
        full_names = df_stats['Renner'].unique()
        name_mapping = {}
        
        manual_overrides = {
            "Poel": "Mathieu van der Poel", "Aert": "Wout van Aert", "Lie": "Arnaud De Lie",
            "Gils": "Maxim Van Gils", "Broek": "Frank van den Broek",
            "Magnier": "Paul Magnier", "Pogacar": "Tadej Pogačar", "Skujins": "Toms Skujiņš",
            "Kooij": "Olav Kooij", "Gianni Vermeersch": "Gianni Vermeersch", "Florian Vermeersch": "Florian Vermeersch"
        }
        
        for short in short_names:
            if short in manual_overrides:
                name_mapping[short] = manual_overrides[short]
            else:
                match_res = process.extractOne(short, full_names, scorer=fuzz.token_set_ratio)
                name_mapping[short] = match_res[0] if match_res and match_res[1] > 75 else short

        df_prog['Renner_Full'] = df_prog['Renner'].map(name_mapping)
        merged_df = pd.merge(df_prog, df_stats, left_on='Renner_Full', right_on='Renner', how='left')
        
        if 'Renner_x' in merged_df.columns:
            merged_df = merged_df.drop(columns=['Renner_x', 'Renner_y'], errors='ignore')
            
        merged_df = merged_df.sort_values(by='Prijs', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=['Renner_Full'], keep='first')
        merged_df = merged_df.rename(columns={'Renner_Full': 'Renner'})
        
        early_races = ['OHN', 'KBK', 'SB', 'PN', 'TA', 'MSR', 'BDP', 'E3', 'GW', 'DDV', 'RVV', 'SP', 'PR']
        late_races = ['BP', 'AGR', 'WP', 'LBL']
        
        available_early = [k for k in early_races if k in merged_df.columns]
        available_late = [k for k in late_races if k in merged_df.columns]
        available_races = available_early + available_late
        
        all_stats_cols = ['COB', 'HLL', 'SPR', 'AVG', 'FLT', 'MTN', 'ITT', 'GC', 'OR', 'TTL']
        for col in available_races + all_stats_cols + ['Prijs']:
            if col not in merged_df.columns:
                merged_df[col] = 0
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
            
        merged_df['HLL/MTN'] = merged_df[['HLL', 'MTN']].max(axis=1).astype(int)
        merged_df['Total_Races'] = merged_df[available_races].sum(axis=1).astype(int)
        
        koers_stat_map = {'OHN':'COB','KBK':'SPR','SB':'HLL','PN':'HLL/MTN','TA':'SPR','MSR':'AVG','BDP':'SPR','E3':'COB','GW':'SPR','DDV':'COB','RVV':'COB','SP':'SPR','PR':'COB','BP':'HLL','AGR':'HLL','WP':'HLL','LBL':'HLL'}
        
        return merged_df, available_early, available_late, koers_stat_map
    except Exception as e:
        st.error(f"Fout in dataverwerking: {e}")
        return pd.DataFrame(), [], [], {}

# --- REKENFUNCTIES ---
def calculate_ev(df, early_races, late_races, koers_stat_map, method):
    df = df.copy()
    df['EV_early'] = 0.0
    df['EV_late'] = 0.0
    scorito_pts = [100, 90, 80, 72, 64, 58, 52, 46, 40, 36, 32, 28, 24, 20, 16, 14, 12, 10, 8, 6]
    
    def get_race_ev(koers):
        stat = koers_stat_map.get(koers, 'AVG')
        starters = df[df[koers] == 1].copy()
        starters = starters.sort_values(by=[stat, 'AVG'], ascending=[False, False])
        race_ev = pd.Series(0.0, index=df.index)
        for i, idx in enumerate(starters.index):
            val = 0.0
            if "Scorito Ranking" in method: val = scorito_pts[i] if i < len(scorito_pts) else 0.0
            elif "Originele Curve" in method: val = (starters.loc[idx, stat] / 100)**4 * 100
            elif "Extreme Curve" in method: val = (starters.loc[idx, stat] / 100)**10 * 100
            elif "Tiers" in method:
                if i < 3: val = 80.0
                elif i < 8: val = 45.0
                elif i < 15: val = 20.0
                else: val = 0.0
            if i == 0: val *= 3.0
            elif i == 1: val *= 2.5
            elif i == 2: val *= 2.0
            race_ev.loc[idx] = val
        return race_ev

    for koers in early_races: df['EV_early'] += get_race_ev(koers)
    for koers in late_races: df['EV_late'] += get_race_ev(koers)
    df['Scorito_EV'] = (df['EV_early'] + df['EV_late']).round(0).astype(int)
    df['Waarde (EV/M)'] = (df['Scorito_EV'] / (df['Prijs'] / 1000000)).replace([float('inf')], 0).fillna(0).round(1)
    return df

def bepaal_klassieker_type(row):
    s = {'Kassei': row.get('COB', 0), 'Heuvel': row.get('HLL', 0), 'Sprint': row.get('SPR', 0)}
    return max(s, key=s.get)

# --- SOLVER ---
def solve_knapsack_with_transfers(dataframe, total_budget, min_budget, max_riders, min_per_race, force_early, ban_early, exclude_list, frozen_x, frozen_y, frozen_z, force_any, early_races, late_races, use_transfers):
    prob = pulp.LpProblem("Scorito_Solver", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("Base", dataframe.index, cat='Binary')
    y = pulp.LpVariable.dicts("Early", dataframe.index, cat='Binary')
    z = pulp.LpVariable.dicts("Late", dataframe.index, cat='Binary')
    
    prob += pulp.lpSum([x[i] * dataframe.loc[i, 'Scorito_EV'] + y[i] * dataframe.loc[i, 'EV_early'] + z[i] * dataframe.loc[i, 'EV_late'] for i in dataframe.index])
    
    for i in dataframe.index:
        renner = dataframe.loc[i, 'Renner']
        prob += x[i] + y[i] + z[i] <= 1
        if renner in force_early: prob += x[i] + y[i] == 1
        if renner in exclude_list: prob += x[i] + y[i] + z[i] == 0
        if renner in frozen_x: prob += x[i] == 1
        if renner in frozen_y: prob += y[i] == 1
        if renner in frozen_z: prob += z[i] == 1
        if renner in force_any: prob += x[i] + y[i] + z[i] == 1

    prob += pulp.lpSum([x[i] for i in dataframe.index]) == max_riders - 3
    prob += pulp.lpSum([y[i] for i in dataframe.index]) == 3
    prob += pulp.lpSum([z[i] for i in dataframe.index]) == 3
    prob += pulp.lpSum([(x[i] + y[i]) * dataframe.loc[i, 'Prijs'] for i in dataframe.index]) <= total_budget
    prob += pulp.lpSum([(x[i] + z[i]) * dataframe.loc[i, 'Prijs'] for i in dataframe.index]) <= total_budget
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] == 'Optimal':
        bt = [dataframe.loc[i, 'Renner'] for i in dataframe.index if x[i].varValue > 0.5]
        et = [dataframe.loc[i, 'Renner'] for i in dataframe.index if y[i].varValue > 0.5]
        lt = [dataframe.loc[i, 'Renner'] for i in dataframe.index if z[i].varValue > 0.5]
        return bt + et, {"uit": et, "in": lt}
    return None, None

# --- UI LOGICA ---
df_raw, early_races, late_races, koers_mapping = load_and_merge_data()
race_cols = early_races + late_races

if "selected_riders" not in st.session_state: st.session_state.selected_riders = []
if "transfer_plan" not in st.session_state: st.session_state.transfer_plan = None

with st.sidebar:
    st.title("🧪 Test Lab")
    ev_method = st.selectbox("Rekenmodel", ["1. Scorito Ranking (Dynamisch)", "2. Originele Curve (Macht 4)", "3. Extreme Curve (Macht 10)", "4. Tiers & Spreiding"])
    max_bud = st.number_input("Budget", value=45000000, step=500000)
    df = calculate_ev(df_raw, early_races, late_races, koers_mapping, ev_method)
    
    if st.button("🚀 BEREKEN OPTIMAAL TEAM", use_container_width=True):
        res, tp = solve_knapsack_with_transfers(df, max_bud, 40000000, 20, 3, [], [], [], [], [], [], [], early_races, late_races, True)
        if res:
            st.session_state.selected_riders = res
            st.session_state.transfer_plan = tp
            st.rerun()

st.title("🏆 Scorito Klassiekers Solver")
st.warning("⚠️ DIT IS DE TEST-OMGEVING (Beta)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Jouw Team", "📋 Database", "🗓️ Kalender", "📊 Terugblik & Uitslagen", "ℹ️ Uitleg"])

with tab1:
    if st.session_state.selected_riders:
        st.subheader("Geselecteerd Team")
        st.write(", ".join(st.session_state.selected_riders))
        
        # CSV Export functie zoals gevraagd
        export_df = df[df['Renner'].isin(st.session_state.selected_riders)].copy()
        export_df['Rol'] = export_df['Renner'].apply(lambda r: "Basis" if r not in st.session_state.transfer_plan['uit'] else "Verkopen na PR")
        if st.session_state.transfer_plan:
            in_df = df[df['Renner'].isin(st.session_state.transfer_plan['in'])].copy()
            in_df['Rol'] = "Kopen na PR"
            export_df = pd.concat([export_df, in_df])
            
        st.download_button("📊 Download Team Matrix (.csv)", data=export_df.to_csv(index=False), file_name="mijn_team.csv", mime="text/csv")
    else:
        st.info("Bereken een team in de zijbalk.")

with tab4:
    st.header("📊 Terugblik: Analyse van de Uitslagen")
    try:
        df_uitslagen = pd.read_csv("uitslagen.csv")
        if not df_uitslagen.empty:
            jouw_team = st.session_state.get('selected_riders', [])
            if st.session_state.transfer_plan:
                jouw_team = list(set(jouw_team + st.session_state.transfer_plan['in']))
            
            # 1. Score berekenen
            score_df = df_uitslagen[df_uitslagen['Naam'].isin(jouw_team)]
            totaal_score = score_df['Punten'].sum()
            
            # 2. De droomploeg (Perfecte 20)
            perf_df = df_uitslagen.groupby('Naam')['Punten'].sum().reset_index()
            perfect_20 = perf_df.sort_values(by='Punten', ascending=False).head(20)
            
            c1, c2 = st.columns(2)
            c1.metric("Jouw Punten", f"{totaal_score} ptn")
            c2.metric("Perfect Team Score", f"{perfect_20['Punten'].sum()} ptn")

            st.divider()
            
            # 3. Kopman Analyse per koers
            st.subheader("🎯 Kopman Validatie")
            koersen_klaar = df_uitslagen['Koers'].unique()
            for k in koersen_klaar:
                with st.expander(f"Uitslag Analyse: {k}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Top 3 Werkelijkheid:**")
                        st.table(df_uitslagen[df_uitslagen['Koers'] == k].sort_values(by='Positie').head(3)[['Positie', 'Naam', 'Punten']])
                    with col_b:
                        if jouw_team:
                            stat = koers_mapping.get(k, 'AVG')
                            st.write(f"**AI Voorspelling (via {stat}):**")
                            top3_ai = df[df['Renner'].isin(jouw_team) & (df[k] == 1)].sort_values(by=[stat, 'AVG'], ascending=False).head(3)
                            st.table(top3_ai[['Renner', stat]])
        else:
            st.info("Voeg data toe aan 'uitslagen.csv' om de analyse te zien.")
    except Exception:
        st.error("Bestand 'uitslagen.csv' niet gevonden in GitHub.")

with tab5:
    st.write("Techniek achter de Post-Race analyse...")
