import streamlit as st
import pandas as pd
from thefuzz import process
import re

# --- CONFIGURATIE ---
st.set_page_config(page_title="Scorito Backtester", layout="wide", page_icon="📊")

st.title("📊 Scorito Backtester: Model vs. Realiteit")
st.markdown("""
Vergelijk de hardcoded AI-modellen met de werkelijke PCS-uitslagen. 
De app weet zelf in welke koers we zitten en activeert automatisch je **wisselstrategie** (voor of na Parijs-Roubaix)!
""")

# --- HARDCODED TEAMS ---
HARDCODED_TEAMS = {
    "Model 1 (Min 3 Renners)": {
        "Basis": [
            "Tadej Pogačar", "Mathieu van der Poel", "Jonathan Milan", "Tim Merlier", 
            "Tim Wellens", "Dylan Groenewegen", "Stefan Küng", "Mattias Skjelmose", 
            "Jasper Stuyven", "João Almeida", "Toms Skujiņš", "Mike Teunissen", 
            "Isaac del Toro", "Jonas Vingegaard", "Jonas Abrahamsen", "Julian Alaphilippe", "Marc Hirschi"
        ],
        "Early": ["Jasper Philipsen", "Mads Pedersen", "Florian Vermeersch"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Model 2 (Min 3 Renners)": {
        "Basis": [
            "Tadej Pogačar", "Mads Pedersen", "Jonathan Milan", "Arnaud De Lie", 
            "Tim Merlier", "Tim Wellens", "Dylan Groenewegen", "Mattias Skjelmose", 
            "Florian Vermeersch", "Toms Skujiņš", "Mike Teunissen", "Marijn van den Berg", 
            "Laurence Pithie", "Jonas Abrahamsen", "Vincenzo Albanese", "Jenno Berckmoes", "Oliver Naesen"
        ],
        "Early": ["Mathieu van der Poel", "Jasper Philipsen", "Jasper Stuyven"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Model 3 (Min 3 Renners)": {
        "Basis": [
            "Tadej Pogačar", "Mathieu van der Poel", "Jasper Philipsen", "Tim Merlier", 
            "Tim Wellens", "Dylan Groenewegen", "Mattias Skjelmose", "Florian Vermeersch", 
            "Toms Skujiņš", "Mike Teunissen", "Isaac del Toro", "Jonas Vingegaard", 
            "Laurence Pithie", "Gianni Vermeersch", "Jonas Abrahamsen", "Julian Alaphilippe", "Quinten Hermans"
        ],
        "Early": ["Mads Pedersen", "Jonathan Milan", "Arnaud De Lie"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Model 4 (Min 3 Renners)": {
        "Basis": [
            "Tadej Pogačar", "Mathieu van der Poel", "Mads Pedersen", "Jonathan Milan", 
            "Tim Wellens", "Paul Magnier", "Dylan Groenewegen", "Mattias Skjelmose", 
            "Jasper Stuyven", "João Almeida", "Toms Skujiņš", "Mike Teunissen", 
            "Jonas Vingegaard", "Giulio Ciccone", "Gianni Vermeersch", "Jonas Abrahamsen", "Marc Hirschi"
        ],
        "Early": ["Jasper Philipsen", "Tim Merlier", "Isaac del Toro"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Mijn Eigen Team": {
        "Basis": [
            "Tadej Pogačar", "Jonathan Milan", "Tom Pidcock", "Christophe Laporte", 
            "Tim Wellens", "Paul Magnier", "Romain Grégoire", "Mattias Skjelmose", 
            "Jasper Stuyven", "Florian Vermeersch", "Milan Fretin", "Jordi Meeus", 
            "Toms Skujiņš", "Mike Teunissen", "Jonas Vingegaard", "Gianni Vermeersch", "Jonas Abrahamsen"
        ],
        "Early": ["Mathieu van der Poel", "Jasper Philipsen", "Laurence Pithie"],
        "Late": ["Remco Evenepoel", "Ben Healy", "Marc Hirschi"]
    }
}

# --- DATA LADEN ---
@st.cache_data
def load_data():
    df_stats = pd.read_csv("renners_stats.csv", sep='\t')
    df_prog = pd.read_csv("bron_startlijsten.csv", sep=None, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
    
    if 'Naam' in df_stats.columns:
        df_stats = df_stats.rename(columns={'Naam': 'Renner'})
    
    return df_stats, df_prog

df_stats, df_prog = load_data()
alle_renners = sorted(df_stats['Renner'].dropna().unique())

# Scorito Puntenverdeling
SCORITO_PUNTEN = {
    1: 100, 2: 80, 3: 70, 4: 60, 5: 50, 6: 44, 7: 40, 8: 36, 9: 32, 10: 28,
    11: 24, 12: 20, 13: 16, 14: 14, 15: 12, 16: 10, 17: 8, 18: 6, 19: 4, 20: 2
}
TEAMPUNTEN = {1: 10, 2: 8, 3: 6}

# --- UI: KOERS & MODEL SELECTEREN ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Team & Koers")
    koers = st.selectbox("Welke koers is dit?", ["OHN", "KBK", "SB", "MSR", "E3", "GW", "DDV", "RVV", "PR", "BP", "AGR", "WP", "LBL", "EF"])
    
    # Bepaal fase voor wissels: na Parijs-Roubaix begint de Heuvel fase
    is_late_season = koers in ["BP", "AGR", "WP", "LBL", "EF"]
    fase_text = "Late Seizoen (Heuvels)" if is_late_season else "Vroege Seizoen (Kasseien)"
    st.info(f"📍 Huidige fase: **{fase_text}**")
    
    stat_mapping = {
        "OHN": "COB", "KBK": "SPR", "SB": "HLL", "MSR": "SPR", "E3": "COB", 
        "GW": "SPR", "DDV": "COB", "RVV": "COB", "PR": "COB", "BP": "HLL",
        "AGR": "HLL", "WP": "HLL", "LBL": "HLL", "EF": "SPR"
    }
    koers_stat = stat_mapping.get(koers, "COB")
    
    model_keuze = st.selectbox("Selecteer Model / Team", list(HARDCODED_TEAMS.keys()) + ["Zelf Samenstellen"])
    
    if model_keuze == "Zelf Samenstellen":
        mijn_team = st.multiselect("Kies 20 renners:", alle_renners, max_selections=20)
    else:
        # Haal het team op uit de hardcoded dict
        model_data = HARDCODED_TEAMS[model_keuze]
        # Bepaal de actieve 20 renners voor deze fase
        mijn_team = model_data["Basis"] + (model_data["Late"] if is_late_season else model_data["Early"])

with col2:
    st.subheader("2. Plak Uitslag (ProCyclingStats)")
    raw_uitslag = st.text_area("Plak hier de top-20 uit de PCS uitslag:", height=250, placeholder="1 van der Poel Mathieu Alpecin-Premier Tech 400 225 4:53:55\n2 van Dijke Tim Red Bull - BORA - hansgrohe 320 150 0:22\n...")

# --- BEREKENING ---
if st.button("🚀 Bereken Score & Kopmannen", type="primary"):
    if len(mijn_team) == 0:
        st.error("Dit team is leeg. Controleer de selectie.")
    elif not raw_uitslag:
        st.error("Plak een uitslag in het tekstvak!")
    else:
        # 1. Automatisch Kopmannen kiezen uit de 20 geselecteerde renners
        team_stats = df_stats[df_stats['Renner'].isin(mijn_team)].copy()
        team_stats = team_stats.sort_values(by=koers_stat, ascending=False).reset_index(drop=True)
        
        kopmannen = team_stats.head(3)['Renner'].tolist()
        c1 = kopmannen[0] if len(kopmannen) > 0 else None
        c2 = kopmannen[1] if len(kopmannen) > 1 else None
        c3 = kopmannen[2] if len(kopmannen) > 2 else None

        # 2. Ruwe tekst inlezen
        uitslag_parsed = []
        lijnen = raw_uitslag.strip().split('\n')
        
        for lijn in lijnen:
            lijn = lijn.replace('\xa0', ' ').strip()
            match = re.match(r'^(\d+)\s+(.+)', lijn)
            if match:
                rank = int(match.group(1))
                if rank > 20: continue 
                
                rest_tekst = match.group(2)
                beste_match, score = process.extractOne(rest_tekst, alle_renners)
                
                if score > 70:
                    team_van_renner = df_stats.loc[df_stats['Renner'] == beste_match, 'Team'].values
                    ploeg = team_van_renner[0] if len(team_van_renner) > 0 else "Onbekend"
                    uitslag_parsed.append({"Rank": rank, "Renner": beste_match, "Ploeg": ploeg})

        df_uitslag = pd.DataFrame(uitslag_parsed)
        
        # 3. Scorito Punten Berekenen
        winnende_ploegen = {}
        for pos in [1, 2, 3]:
            rij = df_uitslag[df_uitslag['Rank'] == pos] if not df_uitslag.empty else pd.DataFrame()
            if not rij.empty:
                winnende_ploegen[pos] = rij['Ploeg'].values[0]

        resultaten_team = []
        totaal_score = 0

        for renner in mijn_team:
            punten = 0
            uitleg = []
            
            finish_rij = df_uitslag[df_uitslag['Renner'] == renner] if not df_uitslag.empty else pd.DataFrame()
            rank = finish_rij['Rank'].values[0] if not finish_rij.empty else None
            base_pts = SCORITO_PUNTEN.get(rank, 0) if rank else 0
            
            multiplier = 1
            if renner == c1: multiplier = 3
            elif renner == c2: multiplier = 2.5
            elif renner == c3: multiplier = 2
            
            if base_pts > 0:
                punten_individueel = int(base_pts * multiplier)
                punten += punten_individueel
                uitleg.append(f"Top 20 ({base_pts}pt x {multiplier})")

            renner_ploeg = df_stats.loc[df_stats['Renner'] == renner, 'Team'].values
            renner_ploeg = renner_ploeg[0] if len(renner_ploeg) > 0 else ""
            
            if rank not in [1, 2, 3]: 
                for pos, punten_team in TEAMPUNTEN.items():
                    if winnende_ploegen.get(pos) == renner_ploeg and renner_ploeg != "Onbekend":
                        punten += punten_team
                        uitleg.append(f"Team (P{pos}: {punten_team}pt)")

            if punten > 0:
                resultaten_team.append({
                    "Renner": renner,
                    "Kopman": "C1" if renner == c1 else ("C2" if renner == c2 else ("C3" if renner == c3 else "")),
                    "Uitslag": f"P{rank}" if rank else "-",
                    "Punten": punten,
                    "Opbouw": " + ".join(uitleg)
                })
                totaal_score += punten

        df_result = pd.DataFrame(resultaten_team)
        
        # --- UI: RESULTATEN TONEN ---
        st.success(f"### 🎉 Totale Scorito Score ({model_keuze}): {int(totaal_score)} punten")
        
        c_res1, c_res2 = st.columns([1, 1])
        
        with c_res1:
            st.write(f"#### 🎯 Automatische Kopmannen op basis van `{koers_stat}`")
            st.info(f"🥇 **C1 (3x):** {c1} \n\n🥈 **C2 (2.5x):** {c2} \n\n🥉 **C3 (2x):** {c3}")
            
            st.write("#### 📝 Jouw Team Scorebord")
            if not df_result.empty:
                df_result = df_result.sort_values(by="Punten", ascending=False)
                st.dataframe(df_result, hide_index=True, use_container_width=True)
            else:
                st.warning("Niemand uit je team heeft punten gescoord of teampunten gepakt.")

        with c_res2:
            st.write("#### 🏁 Ingelezen Uitslag (Top 20)")
            st.dataframe(df_uitslag, hide_index=True, use_container_width=True)
