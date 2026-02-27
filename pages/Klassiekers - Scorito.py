import streamlit as st
import pandas as pd
import pulp
import json
import plotly.express as px
import plotly.graph_objects as go
import unicodedata
from thefuzz import process, fuzz

# --- CONFIGURATIE ---
st.set_page_config(page_title="Scorito Klassiekers AI", layout="wide", page_icon="🏆")

# --- HULPFUNCTIE: NORMALISATIE (Leestekens verwijderen voor matching) ---
def normalize_name_logic(text):
    if not isinstance(text, str):
        return ""
    # Omzetten naar kleine letters en witruimte trimmen
    text = text.lower().strip()
    # Normaliseer Unicode (splitst letters en accenten, bijv. 'ü' -> 'u' + '¨')
    nfkd_form = unicodedata.normalize('NFKD', text)
    # Behoud alleen de basis karakters (geen accent-tekens)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- DATA LADEN (KLASSIEKERS SCORITO) ---
@st.cache_data
def load_and_merge_data():
    try:
        # 1. Programma inladen (met utf-8-sig voor Excel-compatibiliteit)
        df_prog = pd.read_csv("bron_startlijsten.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
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
        
        # 2. Stats inladen
        df_stats = pd.read_csv("renners_stats.csv", sep='\t', encoding='utf-8-sig') 
        if 'Naam' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        
        if 'Team' not in df_stats.columns and 'Ploeg' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Ploeg': 'Team'})
            
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        
        # --- EXTRA DATA TOEVOEGEN VOOR ONBEKENDE RENNERS ---
        extra_riders = [
            {"Renner": "Felix Großschartner", "Team": "UAE Team Emirates", "COB": 35, "HLL": 84, "SPR": 45, "AVG": 78, "MTN": 78, "ITT": 65, "GC": 75, "FLT": 68, "OR": 72, "TTL": 724},
            {"Renner": "Søren Wærenskjold", "Team": "Uno-X Mobility", "COB": 82, "HLL": 42, "SPR": 78, "AVG": 72, "MTN": 30, "ITT": 88, "GC": 45, "FLT": 85, "OR": 68, "TTL": 690},
            {"Renner": "Natnael Tesfatsion", "Team": "Lidl-Trek", "COB": 38, "HLL": 82, "SPR": 74, "AVG": 76, "MTN": 65, "ITT": 35, "GC": 58, "FLT": 62, "OR": 70, "TTL": 620},
            {"Renner": "Filip Maciejuk", "Team": "Red Bull - BORA - hansgrohe", "COB": 68, "HLL": 45, "SPR": 52, "AVG": 64, "MTN": 40, "ITT": 78, "GC": 38, "FLT": 75, "OR": 58, "TTL": 558},
            {"Renner": "Sven Erik Bystrøm", "Team": "Groupama-FDJ", "COB": 72, "HLL": 65, "SPR": 58, "AVG": 70, "MTN": 52, "ITT": 45, "GC": 48, "FLT": 70, "OR": 64, "TTL": 592},
            {"Renner": "Kamil Małecki", "Team": "Q36.5 Pro Cycling", "COB": 74, "HLL": 58, "SPR": 62, "AVG": 68, "MTN": 42, "ITT": 40, "GC": 38, "FLT": 68, "OR": 65, "TTL": 563},
            {"Renner": "Stefan Küng", "Team": "Groupama-FDJ", "COB": 92, "HLL": 65, "SPR": 40, "AVG": 85, "MTN": 45, "ITT": 95, "GC": 60},
            {"Renner": "Romain Grégoire", "Team": "Groupama-FDJ", "COB": 55, "HLL": 88, "SPR": 72, "AVG": 82, "MTN": 68, "ITT": 60, "GC": 70},
            {"Renner": "Michel Hessmann", "Team": "Team Visma | Lease a Bike", "COB": 45, "HLL": 60, "SPR": 30, "AVG": 58, "MTN": 68, "ITT": 72, "GC": 62},
            {"Renner": "Alexander Krieger", "Team": "Tudor Pro Cycling", "COB": 58, "HLL": 35, "SPR": 74, "AVG": 60, "MTN": 25, "ITT": 32, "GC": 20},
            {"Renner": "Torstein Træen", "Team": "Team Bahrain Victorious", "COB": 25, "HLL": 70, "SPR": 28, "AVG": 55, "MTN": 84, "ITT": 45, "GC": 78},
            {"Renner": "Filippo Agostinacchio", "Team": "Astana Qazaqstan", "COB": 25, "HLL": 68, "SPR": 55, "AVG": 60, "MTN": 40, "ITT": 35, "GC": 30},
            {"Renner": "Sakarias Loland", "Team": "Uno-X Mobility", "COB": 65, "HLL": 50, "SPR": 60, "AVG": 60, "MTN": 30, "ITT": 40, "GC": 25},
            {"Renner": "Louis Mahoudo", "Team": "Nantes Atlantique", "COB": 42, "HLL": 62, "SPR": 65, "AVG": 55, "MTN": 45, "ITT": 38, "GC": 40},
            {"Renner": "Mads Øxenberg Hansen", "Team": "Team DSM-Firmenich PostNL", "COB": 55, "HLL": 40, "SPR": 60, "AVG": 55, "MTN": 30, "ITT": 62, "GC": 20},
            {"Renner": "Axel Huens", "Team": "TDT-Unibet", "COB": 62, "HLL": 58, "SPR": 65, "AVG": 60, "MTN": 35, "ITT": 42, "GC": 38},
            {"Renner": "Jannik Steimle", "Team": "Q36.5 Pro Cycling", "COB": 72, "HLL": 45, "SPR": 65, "AVG": 62, "MTN": 25, "ITT": 78, "GC": 25},
            {"Renner": "Martin Tjotta", "Team": "Arkéa - B&B Hotels", "COB": 32, "HLL": 74, "SPR": 50, "AVG": 62, "MTN": 75, "ITT": 35, "GC": 68}
        ]
        df_stats = pd.concat([df_stats, pd.DataFrame(extra_riders)], ignore_index=True)
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        
        # 3. VERBETERDE NAAM MATCHING (Inclusief Küng/Gregoire fix)
        short_names = df_prog['Renner'].unique()
        full_names = df_stats['Renner'].unique()
        
        norm_to_full = {normalize_name_logic(n): n for n in full_names}
        norm_full_names = list(norm_to_full.keys())
        
        name_mapping = {}
        manual_overrides = {
            "Poel": "Mathieu van der Poel", "Aert": "Wout van Aert", "Lie": "Arnaud De Lie",
            "Gils": "Maxim Van Gils", "Broek": "Frank van den Broek",
            "Magnier": "Paul Magnier", "Pogacar": "Tadej Pogačar", "Skujins": "Toms Skujiņš",
            "Kooij": "Olav Kooij", "Kung": "Stefan Küng", "Gregoire": "Romain Grégoire",
            "Grossschartner": "Felix Großschartner", "Waerenskjold": "Søren Wærenskjold",
            "Traeen": "Torstein Træen", "Malecki": "Kamil Małecki", "Bystrom": "Sven Erik Bystrøm",
            "Agostinacchio": "Filippo Agostinacchio", "Hessmann": "Michel Hessmann",
            "Krieger": "Alexander Krieger", "Loland": "Sakarias Loland", "Maciejuk": "Filip Maciejuk",
            "Mahoudo": "Louis Mahoudo", "Oxenberg": "Mads Øxenberg Hansen",
            "Renard-Haquin": "Axel Huens", "Steimle": "Jannik Steimle", "Tesfatsion": "Natnael Tesfatsion",
            "Tjotta": "Martin Tjotta"
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
            
        if 'Team' not in merged_df.columns:
            merged_df['Team'] = 'Onbekend'
        else:
            merged_df['Team'] = merged_df['Team'].fillna('Onbekend')
        
        merged_df['Total_Races'] = merged_df[available_races].sum(axis=1).astype(int)
        
        koers_stat_map = {'OHN':'COB','KBK':'SPR','SB':'HLL','PN':'HLL/MTN','TA':'SPR','MSR':'AVG','BDP':'SPR','E3':'COB','GW':'SPR','DDV':'COB','RVV':'COB','SP':'SPR','PR':'COB','BP':'HLL','AGR':'HLL','WP':'HLL','LBL':'HLL'}
        
        return merged_df, available_early, available_late, koers_stat_map
    except Exception as e:
        st.error(f"Fout in dataverwerking: {e}")
        return pd.DataFrame(), [], [], {}

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
            if "Scorito Ranking" in method:
                val = scorito_pts[i] if i < 20 else 0.0
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
        return race_ev

    for koers in early_races: df['EV_early'] += get_race_ev(koers)
    for koers in late_races: df['EV_late'] += get_race_ev(koers)
        
    df['EV_early'] = df['EV_early'].fillna(0).round(0).astype(int)
    df['EV_late'] = df['EV_late'].fillna(0).round(0).astype(int)
    df['Scorito_EV'] = df['EV_early'] + df['EV_late']
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

# --- SOLVER ---
def solve_knapsack_with_transfers(dataframe, total_budget, min_budget, max_riders, min_per_race, force_early, ban_early, exclude_list, frozen_x, frozen_y, frozen_z, force_any, early_races, late_races, use_transfers):
    prob = pulp.LpProblem("Scorito_Solver", pulp.LpMaximize)
    
    if use_transfers:
        x = pulp.LpVariable.dicts("Base", dataframe.index, cat='Binary')
        y = pulp.LpVariable.dicts("Early", dataframe.index, cat='Binary')
        z = pulp.LpVariable.dicts("Late", dataframe.index, cat='Binary')
        
        prob += pulp.lpSum([x[i] * dataframe.loc[i, 'Scorito_EV'] + y[i] * dataframe.loc[i, 'EV_early'] + z[i] * dataframe.loc[i, 'EV_late'] for i in dataframe.index])
        
        for i in dataframe.index:
            renner = dataframe.loc[i, 'Renner']
            prob += x[i] + y[i] + z[i] <= 1
            if renner in force_early: prob += x[i] + y[i] == 1
            if renner in ban_early: prob += x[i] + y[i] == 0
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
        prob += pulp.lpSum([(x[i] + y[i]) * dataframe.loc[i, 'Prijs'] for i in dataframe.index]) >= min_budget
        
        for koers in early_races: prob += pulp.lpSum([(x[i] + y[i]) * dataframe.loc[i, koers] for i in dataframe.index]) >= min_per_race
        for koers in late_races: prob += pulp.lpSum([(x[i] + z[i]) * dataframe.loc[i, koers] for i in dataframe.index]) >= min_per_race

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            base_team = [dataframe.loc[i, 'Renner'] for i in dataframe.index if x[i].varValue > 0.5]
            early_team = [dataframe.loc[i, 'Renner'] for i in dataframe.index if y[i].varValue > 0.5]
            late_team = [dataframe.loc[i, 'Renner'] for i in dataframe.index if z[i].varValue > 0.5]
            return base_team + early_team, {"uit": early_team, "in": late_team}
            
    else:
        rider_vars = pulp.LpVariable.dicts("Riders", dataframe.index, cat='Binary')
        prob += pulp.lpSum([dataframe.loc[i, 'Scorito_EV'] * rider_vars[i] for i in dataframe.index])
        prob += pulp.lpSum([rider_vars[i] for i in dataframe.index]) == max_riders
        prob += pulp.lpSum([dataframe.loc[i, 'Prijs'] * rider_vars[i] for i in dataframe.index]) <= total_budget
        prob += pulp.lpSum([dataframe.loc[i, 'Prijs'] * rider_vars[i] for i in dataframe.index]) >= min_budget
        for koers in early_races + late_races: prob += pulp.lpSum([dataframe.loc[i, koers] * rider_vars[i] for i in dataframe.index]) >= min_per_race
        for i in dataframe.index:
            renner = dataframe.loc[i, 'Renner']
            if renner in force_early: prob += rider_vars[i] == 1
            if renner in ban_early: prob += rider_vars[i] == 0
            if renner in exclude_list: prob += rider_vars[i] == 0
            if renner in frozen_x: prob += rider_vars[i] == 1
            if renner in force_any: prob += rider_vars[i] == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=15))
        if pulp.LpStatus[prob.status] == 'Optimal':
            selected = [dataframe.loc[i, 'Renner'] for i in dataframe.index if rider_vars[i].varValue > 0.5]
            return selected[:max_riders], None
            
    return None, None

# --- HOOFDCODE ---
df_raw, early_races, late_races, koers_mapping = load_and_merge_data()
if df_raw.empty:
    st.warning("Data is leeg of kon niet worden geladen.")
    st.stop()
    
race_cols = early_races + late_races

if "selected_riders" not in st.session_state: st.session_state.selected_riders = []
if "transfer_plan" not in st.session_state: st.session_state.transfer_plan = None
if "last_finetune" not in st.session_state: st.session_state.last_finetune = None

# --- SIDEBAR (CONTROLECENTRUM) ---
with st.sidebar:
    st.title("🏆 AI Coach")
    
    ev_method = st.selectbox("🧮 Rekenmodel (EV)", ["1. Scorito Ranking (Dynamisch)", "2. Originele Curve (Macht 4)", "3. Extreme Curve (Macht 10)", "4. Tiers & Spreiding (Realistisch)"])
    use_transfers = st.checkbox("🔁 Bereken met 3 wissels (Parijs-Roubaix)", value=True)
    
    with st.expander("⚙️ Budget & Limieten", expanded=False):
        max_ren = st.number_input("Totaal aantal renners", value=20)
        max_bud = st.number_input("Max Budget", value=45000000, step=500000)
        min_bud = st.number_input("Min Budget", value=43000000, step=500000)
        min_per_koers = st.slider("Min. renners per koers", 0, 15, 3)
        
    df = calculate_ev(df_raw, early_races, late_races, koers_mapping, ev_method)
    
    with st.expander("🔒 Renners Forceren / Uitsluiten", expanded=False):
        force_early = st.multiselect("🟢 Moet in team:", options=df['Renner'].tolist())
        ban_early = st.multiselect("🔴 Niet als basis (evt wissel):", options=[r for r in df['Renner'].tolist() if r not in force_early])
        exclude_list = st.multiselect("🚫 Compleet negeren:", options=[r for r in df['Renner'].tolist() if r not in force_early + ban_early])

    st.write("")
    if st.button("🚀 BEREKEN OPTIMAAL TEAM", type="primary", use_container_width=True):
        st.session_state.last_finetune = None
        res, transfer_plan = solve_knapsack_with_transfers(
            df, max_bud, min_bud, max_ren, min_per_koers, force_early, ban_early, exclude_list, [], [], [], [], early_races, late_races, use_transfers
        )
        if res:
            st.session_state.selected_riders = res
            st.session_state.transfer_plan = transfer_plan
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
                    oud_plan = ld.get("transfer_plan")
                    
                    huidige_renners = df['Renner'].tolist()
                    def update_naam(naam):
                        if naam in huidige_renners: return naam
                        match = process.extractOne(naam, huidige_renners, scorer=fuzz.token_set_ratio)
                        return match[0] if match and match[1] > 80 else naam

                    st.session_state.selected_riders = [update_naam(r) for r in oude_selectie if update_naam(r) in huidige_renners]
                    if oud_plan and isinstance(oud_plan, dict):
                        st.session_state.transfer_plan = {
                            "uit": [update_naam(r) for r in oud_plan.get("uit", []) if update_naam(r) in huidige_renners],
                            "in": [update_naam(r) for r in oud_plan.get("in", []) if update_naam(r) in huidige_renners]
                        }
                    else:
                        st.session_state.transfer_plan = None
                        
                    st.session_state.last_finetune = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Fout bij inladen: {e}")

st.title("🏆 Voorjaarsklassiekers: Scorito")
st.markdown("**🔗 Handige links:** [Wielerorakel.nl](https://www.cyclingoracle.com/) | [Kopmanpuzzel](https://kopmanpuzzel.up.railway.app/)")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Jouw Team & Analyse", "📋 Alle Renners (Database)", "🗓️ Kalender & Profielen", "ℹ️ Uitleg & Documentatie"])

with tab1:
    if st.session_state.selected_riders:
        all_display_riders = st.session_state.selected_riders + (st.session_state.transfer_plan['in'] if st.session_state.transfer_plan else [])
        current_df = df[df['Renner'].isin(all_display_riders)].copy()
        
        # --- SUGGESTIES / GHOST CHECK ---
        ghosts = current_df[current_df['Total_Races'] == 0]['Renner'].tolist()
        if ghosts:
            st.warning("🚨 **Actie Vereist:** Er zijn updates in de startlijsten vergeleken met je opgeslagen team.")
            if st.button("🗑️ Verwijder spookrijders uit mijn selectie"):
                st.session_state.selected_riders = [r for r in st.session_state.selected_riders if r not in ghosts]
                st.rerun()

        def bepaal_rol(naam):
            if st.session_state.transfer_plan:
                if naam in st.session_state.transfer_plan['uit']: return 'Verkopen na PR'
                if naam in st.session_state.transfer_plan['in']: return 'Kopen na PR'
            return 'Basis'
            
        current_df['Rol'] = current_df['Renner'].apply(bepaal_rol)
        current_df['Type'] = current_df.apply(bepaal_klassieker_type, axis=1)
        start_team_df = current_df[current_df['Rol'] != 'Kopen na PR']

        st.subheader("📊 Dashboard")
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Budget over (Start)", f"€ {max_bud - start_team_df['Prijs'].sum():,.0f}")
        m2.metric("🚴 Renners (Start)", f"{len(start_team_df)} / {max_ren}")
        m3.metric("🎯 Team EV", f"{current_df['Scorito_EV'].sum():.0f}")

        c1, c2 = st.columns(2); c3, c4 = st.columns(2)
        with c1:
            avg_stats = start_team_df[['COB', 'HLL', 'SPR', 'AVG']].mean()
            fig = go.Figure(go.Scatterpolar(r=avg_stats.values.tolist() + [avg_stats[0]], theta=['Kassei','Heuvel','Sprint','Allround','Kassei'], fill='toself'))
            fig.update_layout(height=350, title="Gemiddeld Team Profiel", polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig_donut = px.pie(current_df, names='Type', values='Prijs', hole=0.4, title="Budget Verdeling per Type")
            fig_donut.update_layout(height=350); st.plotly_chart(fig_donut, use_container_width=True)
        with c3:
            fig_teams = px.bar(current_df['Team'].value_counts().reset_index().rename(columns={'count':'Aantal'}), x='Team', y='Aantal', title="Spreiding over Teams")
            st.plotly_chart(fig_teams, use_container_width=True)

        st.divider()
        with st.container(border=True):
            st.subheader("🛠️ Team Finetuner")
            col_a, col_b = st.columns(2)
            with col_a: to_replace = st.multiselect("❌ Verwijderen uit team:", all_display_riders)
            with col_b: to_add = st.multiselect("📥 Handmatige vervangers:", [r for r in df['Renner'].tolist() if r not in all_display_riders])
            if st.button("🚀 VOER WIJZIGING DOOR", use_container_width=True):
                f_x = [r for r in st.session_state.selected_riders if r not in to_replace]
                res, tp = solve_knapsack_with_transfers(df, max_bud, min_bud, max_ren, min_per_koers, [], [], exclude_list + to_replace, f_x, [], to_add, [], early_races, late_races, use_transfers)
                if res: st.session_state.selected_riders, st.session_state.transfer_plan = res, tp; st.rerun()

        st.subheader("🗓️ Opstellingsschema & Kopmannen")
        matrix = current_df[['Renner', 'Prijs', 'Team', 'Type'] + race_cols].copy()
        for c in race_cols:
            stat = koers_mapping.get(c, 'AVG')
            starters = current_df[current_df[c] == 1].copy()
            if st.session_state.transfer_plan:
                if race_cols.index(c) < race_cols.index('PR'): starters = starters[~starters['Renner'].isin(st.session_state.transfer_plan['in'])]
                elif race_cols.index(c) > race_cols.index('PR'): starters = starters[~starters['Renner'].isin(st.session_state.transfer_plan['uit'])]
            top = starters.sort_values(by=[stat, 'AVG'], ascending=False).head(3)['Renner'].tolist()
            matrix[c] = current_df.apply(lambda r: '🥇 Kop 1' if r['Renner'] == (top[0] if top else '') else ('🥈 Kop 2' if r['Renner'] == (top[1] if len(top)>1 else '') else ('🥉 Kop 3' if r['Renner'] == (top[2] if len(top)>2 else '') else ('✅' if (r['Renner'] in starters['Renner'].values) else '-'))), axis=1)
        st.dataframe(matrix, use_container_width=True, hide_index=True)
        
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            save_data = {"selected_riders": st.session_state.selected_riders, "transfer_plan": st.session_state.transfer_plan}
            st.download_button("📥 Download .JSON Backup", data=json.dumps(save_data), file_name="team_backup.json", use_container_width=True)
        with c_dl2:
            export_df = current_df[['Renner', 'Rol', 'Prijs', 'Team', 'Type', 'Waarde (EV/M)', 'Scorito_EV']].copy()
            st.download_button("📊 Download .CSV Matrix", data=export_df.to_csv(index=False), file_name="team_matrix.csv", use_container_width=True)

    else: st.info("👈 Bereken een team via de sidebar.")

with tab2:
    st.header("📋 Database: Alle Renners")
    st.dataframe(df[['Renner', 'Team', 'Prijs', 'Scorito_EV', 'Waarde (EV/M)', 'COB', 'HLL', 'SPR', 'AVG']].sort_values(by='Scorito_EV', ascending=False), use_container_width=True, hide_index=True)

with tab3:
    st.header("🗓️ Kalender & Toegekende Profielen")
    kalender_data = [
        {"Koers": "Omloop Het Nieuwsblad", "Afkorting": "OHN", "Profiel AI": "Kassei (COB)", "Fase": "Voor Roubaix"},
        {"Koers": "Kuurne-Brussel-Kuurne", "Afkorting": "KBK", "Profiel AI": "Sprint (SPR)", "Fase": "Voor Roubaix"},
        {"Koers": "Strade Bianche", "Afkorting": "SB", "Profiel AI": "Heuvel (HLL)", "Fase": "Voor Roubaix"},
        {"Koers": "Parijs-Nice (Etappe 7)", "Afkorting": "PN", "Profiel AI": "Heuvel/Klimmer (HLL/MTN)", "Fase": "Voor Roubaix"},
        {"Koers": "Tirreno-Adriatico (Etappe 7)", "Afkorting": "TA", "Profiel AI": "Sprint (SPR)", "Fase": "Voor Roubaix"},
        {"Koers": "Milaan-San Remo", "Afkorting": "MSR", "Profiel AI": "Allround (AVG)", "Fase": "Voor Roubaix"},
        {"Koers": "Bredene Koksijde Classic", "Afkorting": "BDP", "Profiel AI": "Sprint (SPR)", "Fase": "Voor Roubaix"},
        {"Koers": "E3 Saxo Classic", "Afkorting": "E3", "Profiel AI": "Kassei (COB)", "Fase": "Voor Roubaix"},
        {"Koers": "Gent-Wevelgem", "Afkorting": "GW", "Profiel AI": "Sprint (SPR)", "Fase": "Voor Roubaix"},
        {"Koers": "Dwars door Vlaanderen", "Afkorting": "DDV", "Profiel AI": "Kassei (COB)", "Fase": "Voor Roubaix"},
        {"Koers": "Ronde van Vlaanderen", "Afkorting": "RVV", "Profiel AI": "Kassei (COB)", "Fase": "Voor Roubaix"},
        {"Koers": "Scheldeprijs", "Afkorting": "SP", "Profiel AI": "Sprint (SPR)", "Fase": "Voor Roubaix"},
        {"Koers": "Parijs-Roubaix", "Afkorting": "PR", "Profiel AI": "Kassei (COB)", "Fase": "Wisselmoment"},
        {"Koers": "Brabantse Pijl", "Afkorting": "BP", "Profiel AI": "Heuvel (HLL)", "Fase": "Na Roubaix"},
        {"Koers": "Amstel Gold Race", "Afkorting": "AGR", "Profiel AI": "Heuvel (HLL)", "Fase": "Na Roubaix"},
        {"Koers": "Waalse Pijl", "Afkorting": "WP", "Profiel AI": "Heuvel (HLL)", "Fase": "Na Roubaix"},
        {"Koers": "Luik-Bastenaken-Luik", "Afkorting": "LBL", "Profiel AI": "Heuvel (HLL)", "Fase": "Na Roubaix"}
    ]
    st.table(pd.DataFrame(kalender_data))

with tab4:
    st.header("ℹ️ De Techniek: Hoe werkt deze AI?")
    st.markdown("""
    Deze applicatie elimineert emotie en 'gut feeling' uit het samenstellen van je Scorito team. Het is een wiskundige optimalisatie-tool die leunt op de wetten van de lineaire programmering. Het doel? Binnen een budget de maximale hoeveelheid verwachte punten vinden.
    
    ### 📊 1. Data & Validatie
    De tool combineert Skill-scores (0-100) van bronnen als Wielerorakel met actuele prijzen. De 'Fuzzy Matcher' corrigeert namen als **Küng** en **Grégoire**.
    
    ### 🧮 2. Expected Value (EV)
    Elke koers heeft een label (bijv. RVV = `COB`). De AI bepaalt de waarde op basis van:
    * **Scorito Ranking:** De top op de startlijst krijgt de punten van plek 1 t/m 20.
    * **Originele Curve:** Een exponentiële berekening die het verschil tussen een 99 en een 85 uitvergroot (Macht 4).
    * **Kopmanfactor:** De beste 3 renners krijgen x3, x2.5 en x2 bonus.
    
    ### 🤖 3. Het Algoritme (The Knapsack Problem)
    De solver berekent miljoenen combinaties om het wiskundige optimum te vinden: max EV binnen budget.
    """)
