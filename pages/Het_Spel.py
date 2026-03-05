import streamlit as st
import pandas as pd
import json
import os
from thefuzz import process, fuzz

st.set_page_config(page_title="Custom Klassiekers Spel", layout="wide", page_icon="🎮")

# --- HULPFUNCTIES ---
def normalize_name_logic(text):
    if not isinstance(text, str): return ""
    import unicodedata
    text = text.lower().strip()
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def get_file_mod_time(filepath):
    return os.path.getmtime(filepath) if os.path.exists(filepath) else 0

# --- DATA LADEN ---
@st.cache_data
def load_game_data():
    try:
        df_prog = pd.read_csv("sporza_prijzen_startlijst.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
        df_prog.columns = df_prog.columns.str.strip()
        if 'Naam' in df_prog.columns: df_prog = df_prog.rename(columns={'Naam': 'Renner'})
        
        df_stats = pd.read_csv("renners_stats.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip') 
        df_stats.columns = df_stats.columns.str.strip()
        if 'Naam' in df_stats.columns: df_stats = df_stats.rename(columns={'Naam': 'Renner'})
        df_stats = df_stats.drop_duplicates(subset=['Renner'], keep='first')
        
        overlap_cols = [c for c in df_stats.columns if c in df_prog.columns and c != 'Renner']
        df_stats = df_stats.drop(columns=overlap_cols)
        
        # Fuzzy matching (verkort voor overzichtelijkheid)
        full_names = {normalize_name_logic(n): n for n in df_stats['Renner'].unique()}
        name_mapping = {}
        for short in df_prog['Renner'].unique():
            match = process.extractOne(normalize_name_logic(short), list(full_names.keys()), scorer=fuzz.token_set_ratio)
            name_mapping[short] = full_names[match[0]] if match and match[1] > 75 else short
            
        df_prog['Renner_Full'] = df_prog['Renner'].map(name_mapping)
        df = pd.merge(df_prog, df_stats, left_on='Renner_Full', right_on='Renner', how='left')
        if 'Renner_x' in df.columns: df = df.drop(columns=['Renner_x', 'Renner_y'])
        df = df.rename(columns={'Renner_Full': 'Renner'}).drop_duplicates(subset=['Renner'])
        
        # KALENDER VANAF NA STRADE BIANCHE
        ALLE_KOERSEN = ["OML", "KBK", "SAM", "STR", "NOK", "BKC", "MSR", "RVB", "E3", "IFF", "DDV", "RVV", "SP", "PR", "RVL", "BRP", "AGT", "WAP", "LBL"]
        start_idx = ALLE_KOERSEN.index("STR") + 1
        available_races = [k for k in ALLE_KOERSEN[start_idx:] if k in df.columns]
        
        stat_cols = ['COB', 'HLL', 'SPR', 'AVG']
        for col in available_races + stat_cols:
            if col not in df.columns: df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
        koers_stat_map = {"NOK": "SPR", "BKC": "SPR", "MSR": "AVG", "RVB": "SPR", "E3": "COB", "IFF": "SPR", "DDV": "COB", "RVV": "COB", "SP": "SPR", "PR": "COB", "RVL": "SPR", "BRP": "HLL", "AGT": "HLL", "WAP": "HLL", "LBL": "HLL"}
        
        # Verwachte Ranking Berekenen
        expected_ranks = {}
        for koers in available_races:
            stat = koers_stat_map.get(koers, 'AVG')
            starters = df[df[koers] == 1].sort_values(by=[stat, 'AVG'], ascending=[False, False])
            starters['Expected_Rank'] = range(1, len(starters) + 1)
            expected_ranks[koers] = starters[['Renner', 'Expected_Rank']].set_index('Renner')['Expected_Rank'].to_dict()
            
        return df, available_races, expected_ranks
    except Exception as e:
        st.error(f"Fout: {e}")
        return pd.DataFrame(), [], {}

@st.cache_data
def get_uitslagen(file_mod_time, alle_renners):
    if not os.path.exists("uitslagen.csv"): return pd.DataFrame()
    df_raw = pd.read_csv("uitslagen.csv", sep=None, engine='python')
    df_raw.columns = [str(c).strip().title() for c in df_raw.columns]
    if 'Race' not in df_raw.columns: return pd.DataFrame()
    
    sporza_naar_scorito_map = {'OHN': 'OML', 'SB': 'STR', 'BDP': 'RVB', 'GW': 'IFF', 'BP': 'BRP', 'AGR': 'AGT', 'WP': 'WAP'}
    res = []
    for _, row in df_raw.iterrows():
        koers = sporza_naar_scorito_map.get(str(row['Race']).strip().upper(), str(row['Race']).strip().upper())
        rank_str = str(row['Rnk']).strip().upper()
        if rank_str in ['DNS', 'NAN', '']: continue
        
        match = process.extractOne(str(row['Rider']).strip(), alle_renners, scorer=fuzz.token_set_ratio)
        if match and match[1] > 70:
            res.append({"Race": koers, "Rnk": rank_str, "Renner": match[0]})
    return pd.DataFrame(res)

df, races, exp_ranks = load_game_data()

if df.empty:
    st.stop()

# --- STATE MANAGEMENT ---
if "game_base_team" not in st.session_state: st.session_state.game_base_team = []
if "game_transfers" not in st.session_state: st.session_state.game_transfers = [] # List of dicts: {uit, in, moment}
if "game_picks" not in st.session_state: st.session_state.game_picks = {r: {"extras": [], "joker": None} for r in races}

def get_active_base_team(race):
    active = list(st.session_state.game_base_team)
    race_idx = races.index(race)
    for t in st.session_state.game_transfers:
        if races.index(t['moment']) < race_idx:
            if t['uit'] in active: active.remove(t['uit'])
            if t['in'] not in active: active.append(t['in'])
    return active

# --- UI ---
st.title("🎮 Custom Klassiekers Spel")
st.markdown("Welkom bij je eigen wielerspel! Kies 10 vaste renners, selecteer per koers 3 extra's en bepaal je Joker (> plek 50).")

# Save / Load functionaliteit
with st.sidebar:
    st.header("💾 Bestand")
    save_data = {
        "base_team": st.session_state.game_base_team,
        "transfers": st.session_state.game_transfers,
        "picks": st.session_state.game_picks
    }
    st.download_button("📥 Opslaan (Download Team)", data=json.dumps(save_data), file_name="mijn_team.json", mime="application/json")
    
    st.divider()
    uploaded_file = st.file_uploader("📂 Inladen (Upload Team)", type="json")
    if uploaded_file is not None and st.button("Laad Team in"):
        data = json.load(uploaded_file)
        st.session_state.game_base_team = data.get("base_team", [])
        st.session_state.game_transfers = data.get("transfers", [])
        # Merge picks safely
        saved_picks = data.get("picks", {})
        for r in races:
            if r in saved_picks: st.session_state.game_picks[r] = saved_picks[r]
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Vaste Team & Wissels", "🏁 Per Koers Selectie", "📈 Verwachte Uitslagen", "🏆 Score & Uitslag"])

with tab1:
    st.header("Jouw 10 Vaste Renners")
    st.write("Kies hier je 10 basisrenners voor het hele spel. Er is geen budget.")
    
    all_riders = sorted(df['Renner'].tolist())
    selected_10 = st.multiselect("Selecteer exact 10 renners:", options=all_riders, default=st.session_state.game_base_team, max_selections=10)
    
    if st.button("Bevestig Basis Team", type="primary"):
        st.session_state.game_base_team = selected_10
        st.success("Basis team opgeslagen!")
        st.rerun()
        
    st.divider()
    st.header("🔁 Transfers (Max 2)")
    st.write("Wissel renners uit je vaste team van 10. Maximaal 2 wissels toegestaan.")
    
    c1, c2, c3, c4 = st.columns([2,2,2,1])
    with c1: t_uit = st.selectbox("Eruit:", options=["-"] + st.session_state.game_base_team)
    with c2: t_in = st.selectbox("Erin:", options=["-"] + [r for r in all_riders if r not in st.session_state.game_base_team])
    with c3: t_mom = st.selectbox("Vanaf welke koers mag hij meedoen?:", options=["-"] + races)
    with c4:
        st.write("")
        if st.button("Voeg wissel toe"):
            if len(st.session_state.game_transfers) >= 2:
                st.error("Maximum van 2 wissels bereikt.")
            elif t_uit != "-" and t_in != "-" and t_mom != "-":
                st.session_state.game_transfers.append({"uit": t_uit, "in": t_in, "moment": t_mom})
                st.rerun()
                
    for i, t in enumerate(st.session_state.game_transfers):
        st.info(f"**Wissel {i+1}:** {t['uit']} ➡️ {t['in']} (Speelt mee vanaf {t['moment']})")

with tab2:
    st.header("Selecties per Koers")
    if len(st.session_state.game_base_team) != 10:
        st.warning("Kies eerst je 10 vaste renners in Tab 1!")
    else:
        race = st.selectbox("Kies de koers om je opstelling te maken:", options=races)
        
        st.divider()
        c_left, c_right = st.columns(2, gap="large")
        
        active_base = get_active_base_team(race)
        starters_race = df[df[race] == 1]['Renner'].tolist()
        
        base_starters = [r for r in active_base if r in starters_race]
        
        with c_left:
            st.subheader(f"🚴 Jouw Vaste Startende Renners")
            if base_starters:
                for r in base_starters: st.success(r)
            else:
                st.write("*Niemand van je vaste team rijdt deze koers...*")
                
        with c_right:
            st.subheader("➕ Kies 3 Extra Renners")
            st.write("Kies 3 renners van de startlijst die je alléén voor deze koers toevoegt.")
            available_extras = [r for r in starters_race if r not in active_base]
            
            cur_extras = st.session_state.game_picks[race]['extras']
            # Zorg dat oude keuzes die nu in basis zitten eruit gehaald worden
            cur_extras = [x for x in cur_extras if x in available_extras] 
            
            new_extras = st.multiselect("Selecteer 3 extra's:", options=available_extras, default=cur_extras, max_selections=3)
            
            st.divider()
            st.subheader("🃏 Kies je Joker (Exp. Rank > 50)")
            st.write("Kies een renner waarvan de verwachte uitslag > 50 is. Eindigt hij in de Top 10? Dan verdien je 50 bonuspunten!")
            
            # Bepaal wie in aanmerking komt voor Joker
            race_ranks = exp_ranks.get(race, {})
            joker_candidates = [r for r in starters_race if race_ranks.get(r, 999) > 50]
            
            # Voeg verwachte rank toe aan de dropdown voor duidelijkheid
            joker_opts = {r: f"{r} (Verwacht: {race_ranks.get(r, 999)})" for r in joker_candidates}
            
            cur_joker = st.session_state.game_picks[race]['joker']
            if cur_joker not in joker_candidates: cur_joker = None
            
            joker_idx = list(joker_opts.keys()).index(cur_joker) + 1 if cur_joker else 0
            
            new_joker = st.selectbox("Selecteer Joker:", options=["Geen Joker"] + list(joker_opts.keys()), index=joker_idx, format_func=lambda x: joker_opts.get(x, x))
            
            if st.button(f"Sla {race} opstelling op", type="primary"):
                st.session_state.game_picks[race]['extras'] = new_extras
                st.session_state.game_picks[race]['joker'] = None if new_joker == "Geen Joker" else new_joker
                st.success("Opgeslagen!")

with tab3:
    st.header("Verwachte Uitslagen (AI Model)")
    st.write("Op basis van de statistieken voorspelt het systeem deze ranking. Renners vanaf plek 51 mogen gekozen worden als Joker.")
    
    r_check = st.selectbox("Bekijk voorspelling voor:", options=races)
    
    pred_df = df[df[r_check] == 1][['Renner', 'Team']].copy()
    pred_df['Expected Rank'] = pred_df['Renner'].map(exp_ranks[r_check])
    pred_df = pred_df.sort_values('Expected Rank')
    
    def highlight_jokers(row):
        return ['background-color: rgba(144, 238, 144, 0.2)'] * len(row) if row['Expected Rank'] > 50 else [''] * len(row)
        
    st.dataframe(pred_df.style.apply(highlight_jokers, axis=1), hide_index=True, use_container_width=True)

with tab4:
    st.header("🏆 Score & Act
