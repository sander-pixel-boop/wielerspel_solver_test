import streamlit as st
import pandas as pd
from thefuzz import process
import re
import plotly.express as px

# --- CONFIGURATIE ---
st.set_page_config(page_title="Scorito Backtester", layout="wide", page_icon="📊")

st.title("📊 Scorito Backtester: Model vs. Realiteit")
st.markdown("""
Plak hier de ruwe uitslag van een koers (bijv. van ProCyclingStats). De app zal:
1. De uitslag automatisch inlezen en koppelen aan de database.
2. Uit jouw geselecteerde team de **beste 3 kopmannen** voorspellen o.b.v. de statistieken.
3. Exact de **Scorito-punten berekenen**, inclusief teampunten voor winnende ploeggenoten!
""")

# --- DATA LADEN ---
@st.cache_data
def load_data():
    df_stats = pd.read_csv("renners_stats.csv", sep='\t')
    df_prog = pd.read_csv("bron_startlijsten.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
    
    # Kolomnamen standaardiseren
    if 'Naam' in df_stats.columns:
        df_stats = df_stats.rename(columns={'Naam': 'Renner'})
    
    return df_stats, df_prog

df_stats, df_prog = load_data()

# Scorito Puntenverdeling Klassiekers
SCORITO_PUNTEN = {
    1: 100, 2: 80, 3: 70, 4: 60, 5: 50, 6: 44, 7: 40, 8: 36, 9: 32, 10: 28,
    11: 24, 12: 20, 13: 16, 14: 14, 15: 12, 16: 10, 17: 8, 18: 6, 19: 4, 20: 2
}
TEAMPUNTEN = {1: 10, 2: 8, 3: 6} # 1e plek = 10pt voor ploeggenoten, etc.

# --- UI: TEAM & KOERS SELECTEREN ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Team & Koers")
    koers = st.selectbox("Welke koers is dit?", ["OHN", "KBK", "SB", "MSR", "E3", "GW", "DDV", "RVV", "PR", "AGR", "WP", "LBL"])
    
    # Koppel de koers aan een stat (simpele mapping)
    stat_mapping = {
        "OHN": "COB", "KBK": "SPR", "SB": "HLL", "MSR": "SPR", "E3": "COB", 
        "GW": "SPR", "DDV": "COB", "RVV": "COB", "PR": "COB", "AGR": "HLL", 
        "WP": "HLL", "LBL": "HLL"
    }
    koers_stat = stat_mapping.get(koers, "COB")
    
    # Gebruiker kiest zijn 20 renners (of 17)
    alle_renners = sorted(df_stats['Renner'].dropna().unique())
    mijn_team = st.multiselect("Selecteer jouw Scorito Team", alle_renners, max_selections=20)

with col2:
    st.subheader("2. Plak Uitslag (ProCyclingStats)")
    raw_uitslag = st.text_area("Plak hier de top-20 uit de PCS uitslag:", height=200, placeholder="1 van der Poel Mathieu Alpecin-Premier Tech 400 225 4:53:55\n2 van Dijke Tim Red Bull - BORA - hansgrohe 320 150 0:22\n...")

# --- BEREKENING ---
if st.button("🚀 Bereken Score & Kopmannen", type="primary"):
    if len(mijn_team) == 0:
        st.error("Selecteer eerst minimaal één renner voor je team!")
    elif not raw_uitslag:
        st.error("Plak een uitslag in het tekstvak!")
    else:
        # 1. Automatisch Kopmannen kiezen uit het eigen team
        team_stats = df_stats[df_stats['Renner'].isin(mijn_team)].copy()
        team_stats = team_stats.sort_values(by=koers_stat, ascending=False).reset_index(drop=True)
        
        kopmannen = team_stats.head(3)['Renner'].tolist()
        # Fallback als het team < 3 renners heeft
        c1 = kopmannen[0] if len(kopmannen) > 0 else None
        c2 = kopmannen[1] if len(kopmannen) > 1 else None
        c3 = kopmannen[2] if len(kopmannen) > 2 else None

        # 2. Ruwe tekst parsen en matchen met de database
        uitslag_parsed = []
        lijnen = raw_uitslag.strip().split('\n')
        
        for lijn in lijnen:
            lijn = lijn.replace('\xa0', ' ').strip()
            # Zoek naar het begincijfer (de positie)
            match = re.match(r'^(\d+)\s+(.+)', lijn)
            if match:
                rank = int(match.group(1))
                if rank > 20: continue # Alleen top 20 krijgt individuele punten
                
                rest_tekst = match.group(2)
                # Gebruik thefuzz om de renner uit de rest_tekst te halen
                beste_match, score = process.extractOne(rest_tekst, alle_renners)
                
                if score > 70: # Alleen betrouwbare matches
                    team_van_renner = df_stats.loc[df_stats['Renner'] == beste_match, 'Team'].values
                    ploeg = team_van_renner[0] if len(team_van_renner) > 0 else "Onbekend"
                    uitslag_parsed.append({"Rank": rank, "Renner": beste_match, "Ploeg": ploeg})

        df_uitslag = pd.DataFrame(uitslag_parsed)
        
        # 3. Scorito Punten Berekenen
        # Zoek op welke ploegen 1e, 2e en 3e zijn geworden voor de teampunten
        winnende_ploegen = {}
        for pos in [1, 2, 3]:
            rij = df_uitslag[df_uitslag['Rank'] == pos]
            if not rij.empty:
                winnende_ploegen[pos] = rij['Ploeg'].values[0]

        resultaten_team = []
        totaal_score = 0

        for renner in mijn_team:
            punten = 0
            uitleg = []
            
            # A) Individuele punten
            finish_rij = df_uitslag[df_uitslag['Renner'] == renner]
            rank = finish_rij['Rank'].values[0] if not finish_rij.empty else None
            
            base_pts = SCORITO_PUNTEN.get(rank, 0) if rank else 0
            
            # B) Kopman Multiplier
            multiplier = 1
            if renner == c1: multiplier = 3
            elif renner == c2: multiplier = 2.5
            elif renner == c3: multiplier = 2
            
            if base_pts > 0:
                punten_individueel = int(base_pts * multiplier)
                punten += punten_individueel
                if multiplier > 1:
                    uitleg.append(f"Top 20 ({base_pts}pt x {multiplier})")
                else:
                    uitleg.append(f"Top 20 ({base_pts}pt)")

            # C) Teampunten
            renner_ploeg = df_stats.loc[df_stats['Renner'] == renner, 'Team'].values
            renner_ploeg = renner_ploeg[0] if len(renner_ploeg) > 0 else ""
            
            if rank not in [1, 2, 3]: # De winnaar krijgt zelf geen teampunten
                for pos, punten_team in TEAMPUNTEN.items():
                    if winnende_ploegen.get(pos) == renner_ploeg and renner_ploeg != "Onbekend":
                        punten += punten_team
                        uitleg.append(f"Teampunten (P{pos}: {punten_team}pt)")

            if punten >
