import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from thefuzz import process

# --- CONFIGURATIE ---
st.set_page_config(page_title="Scorito Backtester", layout="wide", page_icon="📊")

st.title("📊 Scorito Modellen Leaderboard")
st.markdown("Plak de uitslagen, de app werkt `uitslagen.csv` bij en toont de cumulatieve scores per model.")

# --- HARDCODED TEAMS ---
HARDCODED_TEAMS = {
    "Model 1": {
        "Basis": ["Tadej Pogačar", "Mathieu van der Poel", "Jonathan Milan", "Tim Merlier", "Tim Wellens", "Dylan Groenewegen", "Stefan Küng", "Mattias Skjelmose", "Jasper Stuyven", "João Almeida", "Toms Skujiņš", "Mike Teunissen", "Isaac del Toro", "Jonas Vingegaard", "Jonas Abrahamsen", "Julian Alaphilippe", "Marc Hirschi"],
        "Early": ["Jasper Philipsen", "Mads Pedersen", "Florian Vermeersch"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Model 2": {
        "Basis": ["Tadej Pogačar", "Mads Pedersen", "Jonathan Milan", "Arnaud De Lie", "Tim Merlier", "Tim Wellens", "Dylan Groenewegen", "Mattias Skjelmose", "Florian Vermeersch", "Toms Skujiņš", "Mike Teunissen", "Marijn van den Berg", "Laurence Pithie", "Jonas Abrahamsen", "Vincenzo Albanese", "Jenno Berckmoes", "Oliver Naesen"],
        "Early": ["Mathieu van der Poel", "Jasper Philipsen", "Jasper Stuyven"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Model 3": {
        "Basis": ["Tadej Pogačar", "Mathieu van der Poel", "Jasper Philipsen", "Tim Merlier", "Tim Wellens", "Dylan Groenewegen", "Mattias Skjelmose", "Florian Vermeersch", "Toms Skujiņš", "Mike Teunissen", "Isaac del Toro", "Jonas Vingegaard", "Laurence Pithie", "Gianni Vermeersch", "Jonas Abrahamsen", "Julian Alaphilippe", "Quinten Hermans"],
        "Early": ["Mads Pedersen", "Jonathan Milan", "Arnaud De Lie"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Model 4": {
        "Basis": ["Tadej Pogačar", "Mathieu van der Poel", "Mads Pedersen", "Jonathan Milan", "Tim Wellens", "Paul Magnier", "Dylan Groenewegen", "Mattias Skjelmose", "Jasper Stuyven", "João Almeida", "Toms Skujiņš", "Mike Teunissen", "Jonas Vingegaard", "Giulio Ciccone", "Gianni Vermeersch", "Jonas Abrahamsen", "Marc Hirschi"],
        "Early": ["Jasper Philipsen", "Tim Merlier", "Isaac del Toro"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Mijn Eigen Team": {
        "Basis": ["Tadej Pogačar", "Jonathan Milan", "Tom Pidcock", "Christophe Laporte", "Tim Wellens", "Paul Magnier", "Romain Grégoire", "Mattias Skjelmose", "Jasper Stuyven", "Florian Vermeersch", "Milan Fretin", "Jordi Meeus", "Toms Skujiņš", "Mike Teunissen", "Jonas Vingegaard", "Gianni Vermeersch", "Jonas Abrahamsen"],
        "Early": ["Mathieu van der Poel", "Jasper Philipsen", "Laurence Pithie"],
        "Late": ["Remco Evenepoel", "Ben Healy", "Marc Hirschi"]
    }
}

ALLE_KOERSEN = ["OHN", "KBK", "SB", "MSR", "E3", "GW", "DDV", "RVV", "PR", "BP", "AGR", "WP", "LBL", "EF"]
STAT_MAPPING = {"OHN": "COB", "KBK": "SPR", "SB": "HLL", "MSR": "SPR", "E3": "COB", "GW": "SPR", "DDV": "COB", "RVV": "COB", "PR": "COB", "BP": "HLL", "AGR": "HLL", "WP": "HLL", "LBL": "HLL", "EF": "SPR"}
LATE_SEASON_KOERSEN = ["BP", "AGR", "WP", "LBL", "EF"]

# Nieuwe puntenverdeling
SCORITO_PUNTEN = {
    1: 100, 2: 90, 3: 80, 4: 70, 5: 64, 6: 60, 7: 56, 8: 52, 9: 48, 10: 44,
    11: 40, 12: 36, 13: 32, 14: 28, 15: 24, 16: 20, 17: 16, 18: 12, 19: 8, 20: 4
}
# Nieuwe teampunten
TEAMPUNTEN = {1: 30, 2: 20, 3: 10}

# --- DATA LADEN ---
@st.cache_data
def load_data():
    df_stats = pd.read_csv("renners_stats.csv", sep='\t')
    if 'Naam' in df_stats.columns:
        df_stats = df_stats.rename(columns={'Naam': 'Renner'})
    return df_stats

df_stats = load_data()
alle_renners = sorted(df_stats['Renner'].dropna().unique())

# --- UI: UITSLAG TOEVOEGEN ---
with st.expander("➕ Nieuwe Uitslag Toevoegen (Bron: PCS)", expanded=True):
    col1, col2 = st.columns([1, 3])
    with col1:
        koers_input = st.selectbox("Selecteer Koers", ALLE_KOERSEN)
    with col2:
        raw_text = st.text_area("Plak hier de ruwe tekst:", height=150)
    
    if st.button("Sla Uitslag Op", type="primary"):
        if raw_text:
            uitslag_parsed = []
            lijnen = raw_text.strip().split('\n')
            
            for lijn in lijnen:
                lijn = lijn.replace('\xa0', ' ').strip()
                match = re.match(r'^(\d+)\s+(.+)', lijn)
                if match:
                    rank = int(match.group(1))
                    if rank > 20: continue 
                    
                    rest_tekst = match.group(2)
                    beste_match, score = process.extractOne(rest_tekst, alle_renners)
                    
                    if score > 70:
                        uitslag_parsed.append({"Koers": koers_input, "Rank": rank, "Renner": beste_match})
            
            if uitslag_parsed:
                df_new = pd.DataFrame(uitslag_parsed)
                file_path = "uitslagen.csv"
                if os.path.exists(file_path):
                    # Gebruik sep=None voor veilig inlezen
                    df_existing = pd.read_csv(file_path, sep=None, engine='python')
                    # Normaliseer kolomnamen van bestaand bestand (om KeyErrors te voorkomen)
                    df_existing.columns = [str(c).strip().title() for c in df_existing.columns]
                    
                    if 'Koers' in df_existing.columns:
                        df_existing = df_existing[df_existing['Koers'] != koers_input] # Overschrijf bestaande koers
                    
                    df_final = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_final = df_new
                    
                df_final.to_csv(file_path, index=False)
                st.success(f"Top 20 van {koers_input} succesvol opgeslagen in uitslagen.csv!")
            else:
                st.error("Geen geldige renners gevonden in de tekst.")
        else:
            st.error("Plak eerst de tekst.")

st.divider()

# --- GRAFIEK EN BEREKENING ---
if not os.path.exists("uitslagen.csv"):
    st.info("Voeg hierboven een uitslag toe om de grafiek te genereren.")
else:
    # sep=None zoekt zelf uit of het komma's of puntkomma's zijn
    df_uitslagen = pd.read_csv("uitslagen.csv", sep=None, engine='python')
    
    # Forceer kolomnamen naar exact wat we zoeken om KeyErrors te voorkomen
    df_uitslagen.columns = [str(c).strip().title() for c in df_uitslagen.columns]
    
    if 'Koers' not in df_uitslagen.columns or 'Rank' not in df_uitslagen.columns or 'Renner' not in df_uitslagen.columns:
        st.error("Het bestand uitslagen.csv heeft niet de juiste kolommen. Verwijder het bestand en probeer het opnieuw.")
    else:
        verreden_koersen = [k for k in ALLE_KOERSEN if k in df_uitslagen['Koers'].unique()]
        
        if not verreden_koersen:
            st.info("Nog geen geldige koersen in de database.")
        else:
            resultaten_lijst = []

            for koers in verreden_koersen:
                is_late_season = koers in LATE_SEASON_KOERSEN
                koers_stat = STAT_MAPPING.get(koers, "COB")
                
                df_koers_uitslag = df_uitslagen[df_uitslagen['Koers'] == koers]
                
                winnende_ploegen = {}
                for pos in [1, 2, 3]:
                    winnaar = df_koers_uitslag[df_koers_uitslag['Rank'] == pos]
                    if not winnaar.empty:
                        renner_naam = winnaar['Renner'].values[0]
                        ploeg = df_stats.loc[df_stats['Renner'] == renner_naam, 'Team'].values
                        winnende_ploegen[pos] = ploeg[0] if len(ploeg) > 0 else "Onbekend"

                for model_naam, model_data in HARDCODED_TEAMS.items():
                    actieve_selectie = model_data["Basis"] + (model_data["Late"] if is_late_season else model_data["Early"])
                    
                    team_stats = df_stats[df_stats['Renner'].isin(actieve_selectie)].copy()
                    team_stats = team_stats.sort_values(by=koers_stat, ascending=False).reset_index(drop=True)
                    kopmannen = team_stats.head(3)['Renner'].tolist()
                    c1 = kopmannen[0] if len(kopmannen) > 0 else None
                    c2 = kopmannen[1] if len(kopmannen) > 1 else None
                    c3 = kopmannen[2] if len(kopmannen) > 2 else None

                    koers_score = 0
                    
                    for renner in actieve_selectie:
                        punten = 0
                        finish = df_koers_uitslag[df_koers_uitslag['Renner'] == renner]
                        rank = finish['Rank'].values[0] if not finish.empty else None
                        base_pts = SCORITO_PUNTEN.get(rank, 0) if rank else 0
                        
                        multiplier = 1
                        if renner == c1: multiplier = 3
                        elif renner == c2: multiplier = 2.5
                        elif renner == c3: multiplier = 2
                        
                        punten += int(base_pts * multiplier)
                        
                        renner_ploeg = df_stats.loc[df_stats['Renner'] == renner, 'Team'].values
                        renner_ploeg = renner_ploeg[0] if len(renner_ploeg) > 0 else ""
                        
                        if rank not in [1, 2, 3]: 
                            for pos, punten_team in TEAMPUNTEN.items():
                                if winnende_ploegen.get(pos) == renner_ploeg and renner_ploeg != "Onbekend":
                                    punten += punten_team
                                    
                        koers_score += punten
                        
                    resultaten_lijst.append({
                        "Model": model_naam,
                        "Koers": koers,
                        "Punten": koers_score
                    })

            # Data voor grafiek
            df_res = pd.DataFrame(resultaten_lijst)
            df_res['Koers_Index'] = df_res['Koers'].apply(lambda x: verreden_koersen.index(x))
            df_res = df_res.sort_values(by=['Model', 'Koers_Index'])
            df_res['Cumulatieve Punten'] = df_res.groupby('Model')['Punten'].cumsum()

            # Grafiek
            fig = px.line(
                df_res, 
                x="Koers", 
                y="Cumulatieve Punten", 
                color="Model", 
                markers=True,
                title="Cumulatieve Scorito Punten per Model"
            )
            fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=verreden_koersen))
            st.plotly_chart(fig, use_container_width=True)

            # Stand
            c_links, c_rechts = st.columns(2)
            with c_links:
                st.subheader("🏆 Huidige Stand")
                eindstand = df_res.groupby('Model')['Cumulatieve Punten'].max().reset_index().sort_values(by='Cumulatieve Punten', ascending=False)
                eindstand.columns = ['Team', 'Punten']
                st.dataframe(eindstand, hide_index=True)
            
            with c_rechts:
                st.subheader("📋 Ruwe Data")
                df_pivot = df_res.pivot(index='Model', columns='Koers', values='Punten').reindex(columns=verreden_koersen)
                st.dataframe(df_pivot)
