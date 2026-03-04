import streamlit as st
import pandas as pd
import pulp
import unicodedata
import os
from thefuzz import process, fuzz

# --- CONFIGURATIE ---
st.set_page_config(page_title="Sporza Klassiekers AI", layout="wide", page_icon="🚴")

# --- HULPFUNCTIE: NORMALISATIE ---
def normalize_name_logic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- DATA LADEN (SPORZA SPECIFIEK) ---
@st.cache_data
def load_and_merge_data():
    try:
        df_prog = pd.read_csv("bron_startlijsten.csv", sep='\t', engine='python', encoding='utf-8-sig')
        
        # Kolommen hernoemen als ze niet exact matchen
        if 'Naam' in df_prog.columns and 'Renner' not in df_prog.columns:
            df_prog = df_prog.rename(columns={'Naam': 'Renner'})
        
        # Check of data goed geladen is
        if 'Prijs' not in df_prog.columns or 'Team' not in df_prog.columns:
            st.error("CSV-bestand mist 'Prijs' of 'Team' kolommen. Zorg voor tab-gescheiden data.")
            return pd.DataFrame(), [], {}

        df_stats = pd.read_csv("renners_stats.csv", sep='\t', encoding='utf-8-sig') 
        if 'Naam' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Naam': 'Renner'})
            
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        
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
            
        merged_df = merged_df.sort_values(by='Prijs', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=['Renner_Full'], keep='first')
        merged_df = merged_df.rename(columns={'Renner_Full': 'Renner'})
        
        # Sporza Specifieke Koersen (Zorg dat namen in CSV matchen met deze lijst of pas lijst aan)
        ALLE_KOERSEN = ["OML", "KBK", "SAM", "STR", "NOK", "BKC", "MSR", "RVB", "E3", "IFF", "DDV", "RVV", "SP", "PR", "RVL", "BRP", "AGT", "WAP", "LBL"]
        available_races = [k for k in ALLE_KOERSEN if k in merged_df.columns]
        
        all_stats_cols = ['COB', 'HLL', 'SPR', 'AVG', 'FLT', 'MTN', 'ITT', 'GC', 'OR', 'TTL']
        for col in available_races + all_stats_cols + ['Prijs']:
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

# --- EV BEREKENING (SPORZA PUNTEN) ---
def calculate_sporza_ev(df, available_races, koers_stat_map, method):
    df = df.copy()
    
    # Sporza Puntenverdeling
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
                
            # LET OP: Kopman bonus in Sporza is een absoluut getal (30, 25, 20), geen multiplier!
            # Voor de EV doen we een grove aanname dat de nummer 1 ook altijd kopman gemaakt wordt
            if i == 0: val += 30
            elif i == 1: val += 25
            elif i == 2: val += 20
            
            race_ev.loc[idx] = val
        
        race_evs[koers] = race_ev
        df[f'EV_{koers}'] = race_ev

    df['EV_all'] = sum(race_evs.values()) if race_evs else 0.0
    df['Sporza_EV'] = df['EV_all'].fillna(0).round(0).astype(int)
    # Waarde is nu punten per Miljoen (Sporza prijzen zijn 2 t/m 14)
    df['Waarde (EV/M)'] = (df['Sporza_EV'] / df['Prijs']).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)
    
    return df

# --- SPORZA SOLVER (Inclusief 12-Starters regel) ---
def solve_sporza_base(df, available_races):
    prob = pulp.LpProblem("Sporza_Solver", pulp.LpMaximize)
    
    # x: Is de renner in het team van 20?
    x = pulp.LpVariable.dicts("Base", df.index, cat='Binary')
    
    # s: Is de renner OPGESTELD (starter) in een specifieke koers?
    # Dit is een dictionary van dictionaries
    s = {}
    for koers in available_races:
        s[koers] = pulp.LpVariable.dicts(f"Start_{koers}", df.index, cat='Binary')
    
    # OBJECTIEF: Maximaliseer de EV van renners die *daadwerkelijk opgesteld* (s) staan, niet alleen in het team (x) zitten.
    obj = 0
    for koers in available_races:
        obj += pulp.lpSum([s[koers][i] * df.loc[i, f'EV_{koers}'] for i in df.index])
    prob += obj
    
    # RESTRICTIE 1: Teamgrootte = 20
    prob += pulp.lpSum([x[i] for i in df.index]) == 20
    
    # RESTRICTIE 2: Budget = Maximaal 120 Miljoen
    prob += pulp.lpSum([df.loc[i, 'Prijs'] * x[i] for i in df.index]) <= 120
    
    # RESTRICTIE 3: Maximaal 4 renners per Team
    teams = df['Team'].unique()
    for team in teams:
        team_indices = df[df['Team'] == team].index
        prob += pulp.lpSum([x[i] for i in team_indices]) <= 4
        
    # RESTRICTIE 4: Starters Logica per Koers
    for koers in available_races:
        # Een renner kan alleen starten als hij in je team zit
        for i in df.index:
            prob += s[koers][i] <= x[i]
            # Een renner kan alleen starten als hij de koers ook echt rijdt (volgens startlijst)
            prob += s[koers][i] <= df.loc[i, koers]
            
        # Je mag maximaal 12 renners opstellen per koers
        prob += pulp.lpSum([s[koers][i] for i in df.index]) <= 12

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        return [df.loc[i, 'Renner'] for i in df.index if x[i].varValue > 0.5]
    return []

# --- HOOFDCODE ---
df_raw, available_races, koers_mapping = load_and_merge_data()

if df_raw.empty:
    st.warning("Data is leeg of kon niet worden geladen.")
    st.stop()

if "selected_riders" not in st.session_state: st.session_state.selected_riders = []

with st.sidebar:
    st.title("🚴 Sporza AI Coach")
    ev_method = st.selectbox("🧮 Rekenmodel (EV)", ["1. Sporza Ranking (Dynamisch)", "2. Originele Curve (Macht 4)"])
    
    df = calculate_sporza_ev(df_raw, available_races, koers_mapping, ev_method)

    st.write("")
    if st.button("🚀 BEREKEN START-TEAM (Basis-20)", type="primary", use_container_width=True):
        res = solve_sporza_base(df, available_races)
        if res:
            st.session_state.selected_riders = res
            st.rerun()
        else:
            st.error("Geen geldige combinatie mogelijk binnen budget (120M) en ploegrestricties (Max 4).")

st.title("🚴 Voorjaarsklassiekers: Sporza Wielermanager")
st.divider()

if not st.session_state.selected_riders:
    st.info("👈 Klik op **Bereken Start-Team** in de zijbalk om de AI aan het werk te zetten!")
else:
    current_df = df[df['Renner'].isin(st.session_state.selected_riders)].copy()
    
    st.subheader("📊 Jouw Sporza Selectie")
    m1, m2, m3 = st.columns(3)
    m1.metric("💰 Budget over", f"€ {120 - current_df['Prijs'].sum()}M")
    m2.metric("🚴 Renners", f"{len(current_df)} / 20")
    m3.metric("🎯 Team EV (Potentieel)", f"{current_df['Sporza_EV'].sum()}")
    
    st.dataframe(current_df[['Renner', 'Prijs', 'Team', 'Sporza_EV', 'Waarde (EV/M)']].sort_values(by='Prijs', ascending=False), hide_index=True, use_container_width=True)
    
    st.divider()
    st.subheader("🗓️ Startlijst (Met de 12-Starters regel)")
    st.write("De AI selecteert per koers maximaal 12 renners uit je 20-koppige selectie die het meest waarschijnlijk punten scoren.")
    
    display_matrix = current_df[['Renner', 'Prijs'] + available_races].set_index('Renner')
    display_matrix[available_races] = display_matrix[available_races].applymap(lambda x: '✅' if x == 1 else '-')
    
    # Berekening in de matrix (Max 12)
    totals_dict = {}
    for c in available_races:
        starters = current_df[current_df[c] == 1].sort_values(by=f'EV_{c}', ascending=False).head(12)
        totals_dict[c] = str(len(starters))
        
        # Geef de bankzitters (buiten de top 12) een ander icoon
        for idx in current_df[current_df[c] == 1].index:
            renner = current_df.loc[idx, 'Renner']
            if renner not in starters['Renner'].values:
                display_matrix.loc[renner, c] = '🪑 (Bank)'

    totals_df = pd.DataFrame([totals_dict], index=['🚀 AANTAL STARTERS (Max 12)'])
    st.dataframe(totals_df, use_container_width=True)
    st.dataframe(display_matrix, use_container_width=True)
