import streamlit as st
import pandas as pd
import plotly.express as px
import os
from thefuzz import process

# --- CONFIGURATIE ---
st.set_page_config(page_title="Scorito Backtester", layout="wide", page_icon="📊")

st.title("📊 Scorito Modellen Leaderboard")
st.markdown("De app berekent automatisch de standen, kopmannen en puntenopbouw. Gestarte renners (inclusief DNF) worden puur bepaald op basis van het gekoppelde `uitslagen.csv` bestand.")

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

SCORITO_PUNTEN = {
    1: 100, 2: 90, 3: 80, 4: 70, 5: 64, 6: 60, 7: 56, 8: 52, 9: 48, 10: 44,
    11: 40, 12: 36, 13: 32, 14: 28, 15: 24, 16: 20, 17: 16, 18: 12, 19: 8, 20: 4
}
TEAMPUNTEN = {1: 30, 2: 20, 3: 10}

# --- DATA LADEN ---
@st.cache_data
def load_data():
    df_stats = pd.read_csv("renners_stats.csv", sep='\t')
    if 'Naam' in df_stats.columns:
        df_stats = df_stats.rename(columns={'Naam': 'Renner'})
    
    alle_renners = sorted(df_stats['Renner'].dropna().unique())
    return df_stats, alle_renners

df_stats, alle_renners = load_data()

st.divider()

# --- GRAFIEK EN BEREKENING ---
if not os.path.exists("uitslagen.csv"):
    st.error("Bestand `uitslagen.csv` niet gevonden. Zorg dat dit bestand in dezelfde map (GitHub repository) staat.")
else:
    try:
        df_raw_uitslagen = pd.read_csv("uitslagen.csv", sep='\t', engine='python')
    except Exception as e:
        try:
             df_raw_uitslagen = pd.read_csv("uitslagen.csv", sep=None, engine='python')
        except Exception as e2:
             st.error(f"Fout bij inlezen van uitslagen.csv: {e2}")
             st.stop()
             
    df_raw_uitslagen.columns = [str(c).strip().title() for c in df_raw_uitslagen.columns]

    if 'Race' not in df_raw_uitslagen.columns or 'Rnk' not in df_raw_uitslagen.columns or 'Rider' not in df_raw_uitslagen.columns:
        st.error("Het bestand uitslagen.csv mist de vereiste kolommen: Race, Rnk, Rider.")
    else:
        uitslag_parsed = []
        for index, row in df_raw_uitslagen.iterrows():
            koers = str(row['Race']).strip()
            rank_str = str(row['Rnk']).strip().upper()
            
            # Negeer DNS of compleet lege velden
            if rank_str in ['DNS', 'NAN', '']:
                continue 
                
            rider_name = str(row['Rider']).strip()
            beste_match, score = process.extractOne(rider_name, alle_renners)
            
            if score > 70:
                # 999 is de code voor DNF, OTL, etc. Ze zijn gestart, dus ze doen mee voor teampunten.
                rank = int(rank_str) if rank_str.isdigit() else 999 
                uitslag_parsed.append({
                    "Koers": koers, 
                    "Rank": rank, 
                    "Renner": beste_match
                })
                        
        df_uitslagen = pd.DataFrame(uitslag_parsed)
        
        if df_uitslagen.empty:
            st.error("Kon geen enkele renner succesvol matchen. Controleer of de namen in uitslagen.csv kloppen.")
            st.stop()

        verreden_koersen = [k for k in ALLE_KOERSEN if k in df_uitslagen['Koers'].unique()]
        
        if not verreden_koersen:
            st.info("Nog geen geldige koersen gevonden in `uitslagen.csv` (bijv. 'OHN', 'KBK').")
        else:
            resultaten_lijst = []
            details_lijst = []

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
                    
                    beschikbare_renners = [r for r in actieve_selectie if r in df_koers_uitslag['Renner'].values]
                    
                    team_stats = df_stats[df_stats['Renner'].isin(beschikbare_renners)].copy()
                    team_stats = team_stats.sort_values(by=koers_stat, ascending=False).reset_index(drop=True)
                    kopmannen = team_stats.head(3)['Renner'].tolist()
                    c1 = kopmannen[0] if len(kopmannen) > 0 else None
                    c2 = kopmannen[1] if len(kopmannen) > 1 else None
                    c3 = kopmannen[2] if len(kopmannen) > 2 else None

                    koers_score = 0
                    
                    for renner in actieve_selectie:
                        if renner not in beschikbare_renners:
                            continue
                            
                        punten = 0
                        uitleg = []
                        
                        finish = df_koers_uitslag[df_koers_uitslag['Renner'] == renner]
                        rank = finish['Rank'].values[0] if not finish.empty else None
                        base_pts = SCORITO_PUNTEN.get(rank, 0) if rank else 0
                        
                        multiplier = 1
                        kopman_label = "-"
                        if renner == c1: 
                            multiplier = 3
                            kopman_label = "C1"
                        elif renner == c2: 
                            multiplier = 2.5
                            kopman_label = "C2"
                        elif renner == c3: 
                            multiplier = 2
                            kopman_label = "C3"
                        
                        if base_pts > 0:
                            pt_ind = int(base_pts * multiplier)
                            punten += pt_ind
                            if multiplier > 1:
                                uitleg.append(f"Top 20 ({base_pts} x {multiplier})")
                            else:
                                uitleg.append(f"Top 20 ({base_pts})")
                        
                        renner_ploeg = df_stats.loc[df_stats['Renner'] == renner, 'Team'].values
                        renner_ploeg = renner_ploeg[0] if len(renner_ploeg) > 0 else ""
                        
                        if rank not in [1, 2, 3]: 
                            for pos, punten_team in TEAMPUNTEN.items():
                                if winnende_ploegen.get(pos) == renner_ploeg and renner_ploeg != "Onbekend":
                                    punten += punten_team
                                    uitleg.append(f"Team P{pos} ({punten_team})")
                                    
                        koers_score += punten
                        
                        if punten > 0:
                            details_lijst.append({
                                "Koers": koers,
                                "Model": model_naam,
                                "Renner": renner,
                                "Kopman": kopman_label,
                                "Uitslag": f"P{rank}" if rank <= 20 else ("DNF (wel teampunten)" if rank == 999 else f"P{rank}"),
                                "Punten": punten,
                                "Opbouw": " + ".join(uitleg)
                            })
                        
                    resultaten_lijst.append({
                        "Model": model_naam,
                        "Koers": koers,
                        "Punten": koers_score,
                        "C1 (3x)": c1,
                        "C2 (2.5x)": c2,
                        "C3 (2x)": c3
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

            # Standen tabellen
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
                
            st.divider()
            
            # Gekozen kopmannen
            st.subheader("🎯 Automatische Kopmannen per Koers")
            df_kopmannen = df_res[['Koers', 'Model', 'C1 (3x)', 'C2 (2.5x)', 'C3 (2x)']]
            df_kopmannen['Koers_Index'] = df_kopmannen['Koers'].apply(lambda x: verreden_koersen.index(x))
            df_kopmannen = df_kopmannen.sort_values(by=['Koers_Index', 'Model']).drop(columns=['Koers_Index'])
            st.dataframe(df_kopmannen, hide_index=True, use_container_width=True)
            
            st.divider()
            
            # Gedetailleerde puntenopbouw
            st.subheader("🔍 Gedetailleerde Puntenopbouw")
            if details_lijst:
                df_details = pd.DataFrame(details_lijst)
                
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    filter_koers = st.selectbox("Filter op Koers", ["Alle"] + verreden_koersen)
                with f_col2:
                    filter_model = st.selectbox("Filter op Model", ["Alle"] + list(HARDCODED_TEAMS.keys()))
                
                if filter_koers != "Alle":
                    df_details = df_details[df_details['Koers'] == filter_koers]
                if filter_model != "Alle":
                    df_details = df_details[df_details['Model'] == filter_model]
                
                df_details = df_details.sort_values(by=['Koers', 'Model', 'Punten'], ascending=[True, True, False])
                st.dataframe(df_details, hide_index=True, use_container_width=True)
            else:
                st.info("Nog geen punten gescoord door de geselecteerde teams in de verreden koersen.")
