import streamlit as st
import pandas as pd
import pulp
import json
import plotly.express as px
import plotly.graph_objects as go
import unicodedata
import os
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
                return [str(x).strip().upper() for x in df_u['Race'].unique()]
        except:
            pass
    return []

# --- DATA LADEN (CACHE GEKOPPELD AAN BESTANDSDATUM) ---
@st.cache_data
def load_and_merge_data(prog_mod_time, stats_mod_time):
    try:
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
        
        df_stats = pd.read_csv("renners_stats.csv", sep='\t', encoding='utf-8-sig') 
        if 'Naam' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        
        if 'Team' not in df_stats.columns and 'Ploeg' in df_stats.columns:
            df_stats = df_stats.rename(columns={'Ploeg': 'Team'})
            
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
        
        ALLE_KOERSEN = ['OHN', 'KBK', 'SB', 'PN', 'TA', 'MSR', 'BDP', 'E3', 'GW', 'DDV', 'RVV', 'SP', 'PR', 'BP', 'AGR', 'WP', 'LBL']
        available_races = [k for k in ALLE_KOERSEN if k in merged_df.columns]
        
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
        
        return merged_df, available_races, koers_stat_map
    except Exception as e:
        st.error(f"Fout in dataverwerking: {e}")
        return pd.DataFrame(), [], {}

def calculate_dynamic_ev(df, available_races, koers_stat_map, method, skip_races, t_moments):
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
    
    # Berekent EV_y (Punten VOOR wissel) en EV_z (Punten NA wissel)
    for k, t_race in enumerate(t_moments):
        if t_race == 'GEEN':
            df[f'EV_y{k}'] = df['EV_all']
            df[f'EV_z{k}'] = 0.0
        else:
            split_idx = available_races.index(t_race) + 1
            races_before = available_races[:split_idx]
            races_after = available_races[split_idx:]
            
            df[f'EV_y{k}'] = sum([race_evs[r] for r in races_before]) if races_before else 0.0
            df[f'EV_z{k}'] = sum([race_evs[r] for r in races_after]) if races_after else 0.0

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

# --- DE ALLES-OMVATTENDE SOLVER ---
def solve_knapsack_dynamic(df, total_budget, min_budget, max_riders, force_base, ban_base, exclude_list, force_sells, force_buys, t_moments, use_transfers):
    prob = pulp.LpProblem("Scorito_Dynamic_Solver", pulp.LpMaximize)
    
    x = pulp.LpVariable.dicts("Base", df.index, cat='Binary')
    
    if use_transfers:
        y_vars = [pulp.LpVariable.dicts(f"Y{k}", df.index, cat='Binary') for k in range(3)]
        z_vars = [pulp.LpVariable.dicts(f"Z{k}", df.index, cat='Binary') for k in range(3)]
        
        # OBIJECTIEF: Maximaliseer Expected Value (Start team + Punten voor wissel + Punten na wissel)
        obj = pulp.lpSum([x[i] * df.loc[i, 'EV_all'] for i in df.index])
        for k in range(3):
            obj += pulp.lpSum([y_vars[k][i] * df.loc[i, f'EV_y{k}'] + z_vars[k][i] * df.loc[i, f'EV_z{k}'] for i in df.index])
        prob += obj
        
        # RESTRICTIES
        for i in df.index:
            # 1 rol per renner
            prob += x[i] + y_vars[0][i] + y_vars[1][i] + y_vars[2][i] + z_vars[0][i] + z_vars[1][i] + z_vars[2][i] <= 1
            
            renner = df.loc[i, 'Renner']
            if renner in exclude_list:
                prob += x[i] + sum([y_vars[k][i] for k in range(3)]) + sum([z_vars[k][i] for k in range(3)]) == 0
            if renner in force_base:
                prob += x[i] + sum([y_vars[k][i] for k in range(3)]) == 1
            if renner in ban_base:
                prob += x[i] + sum([y_vars[k][i] for k in range(3)]) == 0
            if renner in force_sells:
                prob += sum([y_vars[k][i] for k in range(3)]) == 1
            if renner in force_buys:
                prob += sum([z_vars[k][i] for k in range(3)]) == 1

        for k in range(3):
            prob += pulp.lpSum([y_vars[k][i] for i in df.index]) <= 1  
            prob += pulp.lpSum([y_vars[k][i] for i in df.index]) == pulp.lpSum([z_vars[k][i] for i in df.index]) 
            
        prob += pulp.lpSum([x[i] for i in df.index]) + pulp.lpSum([y_vars[0][i] for i in df.index]) + pulp.lpSum([y_vars[1][i] for i in df.index]) + pulp.lpSum([y_vars[2][i] for i in df.index]) == max_riders
        
        # Budget Check na ELKE wissel (Voorkomt te duur inkopen)
        prob += pulp.lpSum([(x[i] + y_vars[0][i] + y_vars[1][i] + y_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) <= total_budget
        prob += pulp.lpSum([(x[i] + y_vars[0][i] + y_vars[1][i] + y_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) >= min_budget
        
        prob += pulp.lpSum([(x[i] + z_vars[0][i] + y_vars[1][i] + y_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) <= total_budget
        prob += pulp.lpSum([(x[i] + z_vars[0][i] + z_vars[1][i] + y_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) <= total_budget
        prob += pulp.lpSum([(x[i] + z_vars[0][i] + z_vars[1][i] + z_vars[2][i]) * df.loc[i, 'Prijs'] for i in df.index]) <= total_budget

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            base_team = [df.loc[i, 'Renner'] for i in df.index if x[i].varValue > 0.5]
            transfer_plan = []
            
            for k in range(3):
                uit = [df.loc[i, 'Renner'] for i in df.index if y_vars[k][i].varValue > 0.5]
                erin = [df.loc[i, 'Renner'] for i in df.index if z_vars[k][i].varValue > 0.5]
                if uit and erin:
                    transfer_plan.append({"uit": uit[0], "in": erin[0], "moment": t_moments[k]})
                    base_team.append(uit[0])
            
            return base_team, transfer_plan
            
    else:
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
            return [df.loc[i, 'Renner'] for i in df.index if x[i].varValue > 0.5], []
            
    return None, None

# --- HOOFDCODE ---
prog_time = get_file_mod_time("bron_startlijsten.csv")
stats_time = get_file_mod_time("renners_stats.csv")
df_raw, available_races, koers_mapping = load_and_merge_data(prog_time, stats_time)

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
        if st.checkbox("🔮 Toon alleen Resterende EV", value=True, help="Negeer behaalde punten zodat de AI perfect de toekomst voorspelt."):
            skip_races = actieve_koersen

    ev_method = st.selectbox("🧮 Rekenmodel (EV)", ["1. Scorito Ranking (Dynamisch)", "2. Originele Curve (Macht 4)", "3. Extreme Curve (Macht 10)", "4. Tiers & Spreiding (Realistisch)"])
    use_transfers = st.checkbox("🔁 Bereken met wissel-strategie", value=True)
    
    t_moments = ["GEEN", "GEEN", "GEEN"]
    if use_transfers:
        st.markdown("**🗓️ Wanneer wil je wisselen? (Kies max 3 momenten)**")
        default_index = available_races.index('PR') if 'PR' in available_races else len(available_races) - 2
        t1 = st.selectbox("Wissel 1 na:", options=available_races[:-1], index=default_index)
        t2 = st.selectbox("Wissel 2 na:", options=available_races[:-1], index=default_index)
        t3 = st.selectbox("Wissel 3 na:", options=available_races[:-1], index=default_index)
        t_moments = sorted([t1, t2, t3], key=lambda x: available_races.index(x))
    
    with st.expander("⚙️ Budget & Limieten", expanded=False):
        max_ren = st.number_input("Totaal aantal renners", value=20)
        max_bud = st.number_input("Max Budget", value=45000000, step=500000)
        min_bud = st.number_input("Min Budget", value=43000000, step=500000)
        
    df = calculate_dynamic_ev(df_raw, available_races, koers_mapping, ev_method, skip_races, t_moments)

    # --- NIEUWE NOODWISSEL METHODE ---
    st.divider()
    game_locked = st.checkbox("🔒 Spel is begonnen (Start-team vastzetten)", value=False, help="Bevries je huidige 20 renners om uitsluitend de wisselmomenten te optimaliseren.")
    
    force_base = []
    ban_base = []
    ziekenboeg = []
    force_sells = []
    force_buys = []
    
    if game_locked:
        if len(st.session_state.selected_riders) == max_ren:
            current_20 = st.session_state.selected_riders
            force_base = current_20
            ban_base = [r for r in df['Renner'].tolist() if r not in current_20]
            
            st.info("💡 **Tip bij blessures:** Pas hierboven 'Wissel 1 na' aan naar de koers waar je renner geblesseerd raakte (bijv. KBK). Selecteer hem hieronder. De AI herberekent nu *al* je wissels tegelijkertijd binnen je budget!")
            
            ziekenboeg = st.multiselect("🤕 Ziekenboeg (Zet EV op 0):", options=current_20, help="Geef gevallen renners door. De AI zal deze hoogstwaarschijnlijk verkopen als je een wisselmoment hebt toegewezen.")
            force_sells = st.multiselect("❌ Forceer specifieke verkoop:", options=[r for r in current_20 if r not in ziekenboeg])
            force_buys = st.multiselect("📥 Forceer specifieke aankoop:", options=[r for r in df['Renner'].tolist() if r not in current_20])
            
            for z in ziekenboeg:
                df.loc[df['Renner'] == z, [c for c in df.columns if 'EV' in c]] = 0
                df.loc[df['Renner'] == z, 'Scorito_EV'] = 0
        else:
            st.warning("Bereken of laad eerst je complete start-team in om het vast te zetten.")
    else:
        with st.expander("🔒 Handmatige Restricties (Voor het spel)", expanded=False):
            force_base = st.multiselect("🟢 Moet in start-team:", options=df['Renner'].tolist())
            ban_base = st.multiselect("🔴 Niet in start-team (mag evt wel gekocht worden):", options=[r for r in df['Renner'].tolist() if r not in force_base])
    
    # Exclude is altijd beschikbaar
    exclude_list = st.multiselect("🚫 Negeer compleet (verdwijnt uit de berekening):", options=[r for r in df['Renner'].tolist() if r not in force_base])

    st.write("")
    if st.button("🚀 BEREKEN OPTIMAAL TEAM & WISSELS", type="primary", use_container_width=True):
        if game_locked and len(st.session_state.selected_riders) != max_ren:
            st.error("Je kunt je team niet updaten: Start-team ontbreekt of is niet compleet.")
        else:
            res, transfer_plan = solve_knapsack_dynamic(
                df, max_bud, min_bud, max_ren, force_base, ban_base, exclude_list, force_sells + ziekenboeg, force_buys, t_moments, use_transfers
            )
            if res:
                st.session_state.selected_riders = res
                st.session_state.transfer_plan = transfer_plan
                st.rerun()
            else:
                st.error("Geen oplossing mogelijk. Het budget is waarschijnlijk overschreden door je restricties of je vaste kern.")

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

if not st.session_state.selected_riders:
    st.info("👈 Kies je instellingen in de zijbalk en klik op **Bereken Optimaal Team** of laad een eerdere backup in!")
    st.stop()

# --- ANALYSE VERWERKEN ---
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

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Jouw Team & Transfers", "🗓️ Startlijst Matrix", "📊 Kopmannen", "📋 Database (Alle)", "ℹ️ Uitleg"])

with tab1:
    st.subheader("📊 Dashboard")
    
    start_team_df = current_df[current_df['Renner'].isin(st.session_state.selected_riders)]
    start_cost = start_team_df['Prijs'].sum()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("💰 Start Budget (Overig)", f"€ {max_bud - start_cost:,.0f}")
    m2.metric("🚴 Renners (Start)", f"{len(st.session_state.selected_riders)} / {max_ren}")
    
    totaal_ev = 0
    for r in active_matrix.index:
        for c in available_races:
            if active_matrix.loc[r, c] == 1:
                totaal_ev += current_df.loc[current_df['Renner'] == r, f'EV_{c}'].values[0]
                
    m3.metric("🎯 Team EV (Totaal)", f"{totaal_ev:.0f}")
    st.divider()
    
    col_t1, col_t2 = st.columns([1, 1], gap="large")
    with col_t1:
        st.markdown("**🛡️ Jouw Start-Team**")
        st.dataframe(start_team_df[['Renner', 'Prijs', 'Type', 'Rol']].sort_values(by='Prijs', ascending=False), hide_index=True, use_container_width=True)
    
    with col_t2:
        st.markdown(f"**🔁 Het Meesterplan ({len(st.session_state.transfer_plan)}/3)**")

        if not st.session_state.transfer_plan:
            st.info("Je hebt geen wissels gepland (of 0 wissels levert momenteel de meeste punten op).")
        else:
            temp_team = list(st.session_state.selected_riders)
            
            for i, t in enumerate(sorted(st.session_state.transfer_plan, key=lambda x: available_races.index(x['moment']))):
                if t['uit'] in temp_team: temp_team.remove(t['uit'])
                if t['in'] not in temp_team: temp_team.append(t['in'])
                budget_now = max_bud - df[df['Renner'].isin(temp_team)]['Prijs'].sum()
                
                st.markdown(f"***Wissel {i+1} (ná {t['moment']} | Speling: €{budget_now/1000000:.2f}M)***")
                c_uit, c_in = st.columns(2)
                with c_uit: st.error(f"❌ {t['uit']}")
                with c_in: st.success(f"📥 {t['in']}")
                st.write("")

    st.divider()
    st.subheader("💾 Exporteer Team")
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        save_data = {"selected_riders": st.session_state.selected_riders, "transfer_plan": st.session_state.transfer_plan}
        st.download_button("📥 Download als .JSON (Backup)", data=json.dumps(save_data), file_name="scorito_team.json", mime="application/json", use_container_width=True)
    with c_dl2:
        export_df = current_df[['Renner', 'Rol', 'Prijs', 'Team', 'Type', 'Scorito_EV']].copy()
        st.download_button("📊 Download als .CSV (Excel)", data=export_df.to_csv(index=False).encode('utf-8'), file_name="scorito_team.csv", mime="text/csv", use_container_width=True)

with tab2:
    st.header("🗓️ Matrix & Deelnames")
    
    display_matrix = active_matrix[available_races].applymap(lambda x: '✅' if x == 1 else '-')
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
    st.header("ℹ️ Uitleg Noodwissels")
    st.markdown("""
    ### Hoe voer je tijdens het spel een wissel door?
    1. Vink **🔒 Spel is begonnen** aan in de zijbalk. De AI bevriest nu je originele 20 renners.
    2. Zet de momenten bij **'Wanneer wil je wisselen'** goed. Heb je een blessure na Kuurne (KBK)? Zet dan Wissel 1 op 'KBK'. Laat Wissel 2 en 3 staan op Parijs-Roubaix (PR).
    3. Zet de geblesseerde renner in de **Ziekenboeg**.
    4. Klik op Bereken! 
    
    De wiskundige motor berekent nu *in één keer* al je 3 wissels opnieuw. Hij zal de geblesseerde renner na KBK dumpen voor een topper die vroeg in het voorjaar rijdt (bijv. Grégoire), en bewaart perfect je budget om na Roubaix nog een late held (bijv. Evenepoel) te kopen!
    """)
