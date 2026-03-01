import streamlit as st
import pandas as pd
import plotly.express as px
import os
from thefuzz import process

# --- CONFIGURATIE ---
st.set_page_config(page_title="Model Evaluator", layout="wide", page_icon="📊")

st.title("📊 Scorito Model Evaluator")
st.markdown("""
Welkom bij de **Model Evaluator**. Dit dashboard test en vergelijkt live hoe verschillende wiskundige Scorito-modellen presteren in de praktijk. Deze modellen worden afgezet tegen een door een mens samengestelde referentieselectie (*Sander's Team*).

**Hoe het werkt:**
1. Gestarte renners en daadwerkelijke uitslagen worden automatisch verwerkt via het koersverloop.
2. Voor de rekenmodellen kiest het algoritme per koers de beste drie gestarte kopmannen op basis van data en statistieken.
3. Voor de referentieselectie (*Sander's Team*) worden vooraf vastgestelde kopmannen gebruikt.
4. Na elke race worden de individuele punten en teampunten berekend en toegevoegd aan het algemeen klassement.

**Toelichting Modellen:**
* **Rekenmodel 1:** Scorito ranking (dynamisch)
* **Rekenmodel 2:** Originele Curve (Macht 4)
* **Rekenmodel 3:** Extreme Curve (Macht 10)
* **Rekenmodel 4:** Tiers & Spreiding (Realistisch)
""")

# --- HARDCODED TEAMS & KOPMANNEN ---
HARDCODED_TEAMS = {
    "Rekenmodel 1": {
        "Basis": ["Tadej Pogačar", "Mathieu van der Poel", "Jonathan Milan", "Tim Merlier", "Tim Wellens", "Dylan Groenewegen", "Stefan Küng", "Mattias Skjelmose", "Jasper Stuyven", "João Almeida", "Toms Skujiņš", "Mike Teunissen", "Isaac del Toro", "Jonas Vingegaard", "Jonas Abrahamsen", "Julian Alaphilippe", "Marc Hirschi"],
        "Early": ["Jasper Philipsen", "Mads Pedersen", "Florian Vermeersch"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Rekenmodel 2": {
        "Basis": ["Tadej Pogačar", "Mads Pedersen", "Jonathan Milan", "Arnaud De Lie", "Tim Merlier", "Tim Wellens", "Dylan Groenewegen", "Mattias Skjelmose", "Florian Vermeersch", "Toms Skujiņš", "Mike Teunissen", "Marijn van den Berg", "Laurence Pithie", "Jonas Abrahamsen", "Vincenzo Albanese", "Jenno Berckmoes", "Oliver Naesen"],
        "Early": ["Mathieu van der Poel", "Jasper Philipsen", "Jasper Stuyven"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Rekenmodel 3": {
        "Basis": ["Tadej Pogačar", "Mathieu van der Poel", "Jasper Philipsen", "Tim Merlier", "Tim Wellens", "Dylan Groenewegen", "Mattias Skjelmose", "Florian Vermeersch", "Toms Skujiņš", "Mike Teunissen", "Isaac del Toro", "Jonas Vingegaard", "Laurence Pithie", "Gianni Vermeersch", "Jonas Abrahamsen", "Julian Alaphilippe", "Quinten Hermans"],
        "Early": ["Mads Pedersen", "Jonathan Milan", "Arnaud De Lie"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Rekenmodel 4": {
        "Basis": ["Tadej Pogačar", "Mathieu van der Poel", "Mads Pedersen", "Jonathan Milan", "Tim Wellens", "Paul Magnier", "Dylan Groenewegen", "Mattias Skjelmose", "Jasper Stuyven", "João Almeida", "Toms Skujiņš", "Mike Teunissen", "Jonas Vingegaard", "Giulio Ciccone", "Gianni Vermeersch", "Jonas Abrahamsen", "Marc Hirschi"],
        "Early": ["Jasper Philipsen", "Tim Merlier", "Isaac del Toro"],
        "Late": ["Tom Pidcock", "Remco Evenepoel", "Romain Grégoire"]
    },
    "Sander's Team": {
        "Basis": ["Tadej Pogačar", "Jonathan Milan", "Tom Pidcock", "Christophe Laporte", "Tim Wellens", "Paul Magnier", "Romain Grégoire", "Mattias Skjelmose", "Jasper Stuyven", "Florian Vermeersch", "Milan Fretin", "Jordi Meeus", "Toms Skujiņš", "Mike Teunissen", "Jonas Vingegaard", "Gianni Vermeersch", "Jonas Abrahamsen"],
        "Early": ["Mathieu van der Poel", "Jasper Philipsen", "Laurence Pithie"],
        "Late": ["Remco Evenepoel", "Ben Healy", "Marc Hirschi"]
    }
}

# Vaste kopmannen per koers voor de referentieselectie
MIJN_EIGEN_KOPMANNEN = {
    "OHN": {"C1": "Mathieu van der Poel", "C2": "Tom Pidcock", "C3": "Tim Wellens"},
    "KBK": {"C1": "Jonathan Milan", "C2": "Jasper Philipsen", "C3": "Jordi Meeus"},
}

ALLE_KOERSEN = ["OHN", "KBK", "SB", "PN", "TA", "MSR", "RVB", "E3", "IFF", "DDV", "RVV", "SP", "PR", "BP", "AGR", "WP", "LBL"]
STAT_MAPPING = {"OHN": "COB", "KBK": "SPR", "SB": "HLL", "PN": "HLL/MTN", "TA": "SPR", "MSR": "AVG", "RVB": "SPR", "E3": "COB", "IFF": "SPR", "DDV": "COB", "RVV": "COB", "SP": "SPR", "PR": "COB", "BP": "HLL", "AGR": "HLL", "WP": "HLL", "LBL": "HLL"}
LATE_SEASON_KOERSEN = ["BP", "AGR", "WP", "LBL"]

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
    st.error("Bestand `uitslagen.csv` niet gevonden. Zorg dat dit bestand in de hoofddirectory staat.")
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
            
            if rank_str in ['DNS', 'NAN', '']:
                continue 
                
            rider_name = str(row['Rider']).strip()
            beste_match, score = process.extractOne(rider_name, alle_renners)
            
            if score > 70:
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
            st.info("Nog geen geldige koersen gevonden in de dataset.")
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
                    
                    c1, c2, c3 = None, None, None
                    
                    if model_naam == "Sander's Team":
                        geplande_kopmannen = MIJN_EIGEN_KOPMANNEN.get(koers, {})
                        c1_intended = geplande_kopmannen.get("C1")
                        c2_intended = geplande_kopmannen.get("C2")
                        c3_intended = geplande_kopmannen.get("C3")
                        
                        if c1_intended in beschikbare_renners: c1 = c1_intended
                        if c2_intended in beschikbare_renners: c2 = c2_intended
                        if c3_intended in beschikbare_renners: c3 = c3_intended
                            
                    team_stats = df_stats[df_stats['Renner'].isin(beschikbare_renners)].copy()
                    team_stats = team_stats.sort_values(by=koers_stat, ascending=False).reset_index(drop=True)
                    
                    reeds_kopman = [x for x in [c1, c2, c3] if x is not None]
                    
                    for r in team_stats['Renner'].tolist():
                        if c1 is not None and c2 is not None and c3 is not None:
                            break
                        if r in reeds_kopman:
                            continue
                            
                        if c1 is None:
                            c1 = r
                            reeds_kopman.append(r)
                        elif c2 is None:
                            c2 = r
                            reeds_kopman.append(r)
                        elif c3 is None:
                            c3 = r
                            reeds_kopman.append(r)

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

            # Data prep voor grafiek en tabel
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
                title="Verloop Cumulatieve Scorito Punten"
            )
            fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=verreden_koersen))
            st.plotly_chart(fig, use_container_width=True)

            # Gecombineerde Standen Tabel
            st.subheader("🏆 Klassement & Punten per Koers")
            df_pivot = df_res.pivot(index='Model', columns='Koers', values='Punten')
            
            for k in ALLE_KOERSEN:
                if k not in df_pivot.columns:
                    df_pivot[k] = pd.NA
                    
            totaal_punten = df_res.groupby('Model')['Cumulatieve Punten'].max()
            df_combined = df_pivot.copy()
            df_combined.insert(0, 'Totaal', totaal_punten)
            df_combined = df_combined.reset_index().sort_values(by='Totaal', ascending=False)
            
            cols = ['Model', 'Totaal'] + ALLE_KOERSEN
            df_combined = df_combined[cols]
            
            st.dataframe(df_combined, hide_index=True, use_container_width=True)
            
            st.divider()
            
            # Detail Analyse Sectie
            st.subheader("🔍 Inzoomen per Koers")
            geselecteerde_koers = st.selectbox("Selecteer een koers om kopmannen en puntenopbouw in te zien:", verreden_koersen)
            
            if geselecteerde_koers:
                st.markdown(f"**🎯 Kopmannen ({geselecteerde_koers})**")
                df_kop_koers = df_res[df_res['Koers'] == geselecteerde_koers][['Model', 'C1 (3x)', 'C2 (2.5x)', 'C3 (2x)']]
                st.dataframe(df_kop_koers, hide_index=True, use_container_width=True)
                
                st.markdown(f"**📝 Puntenopbouw per Selectie ({geselecteerde_koers})**")
                if details_lijst:
                    df_details = pd.DataFrame(details_lijst)
                    df_det_koers = df_details[df_details['Koers'] == geselecteerde_koers]
                    
                    model_namen = df_kop_koers['Model'].tolist()
                    tabs = st.tabs(model_namen)
                    
                    for i, m_naam in enumerate(model_namen):
                        with tabs[i]:
                            df_m = df_det_koers[df_det_koers['Model'] == m_naam]
                            if not df_m.empty:
                                df_m = df_m[['Renner', 'Kopman', 'Uitslag', 'Punten', 'Opbouw']].sort_values(by='Punten', ascending=False)
                                st.dataframe(df_m, hide_index=True, use_container_width=True)
                                st.markdown(f"*Totaal score in {geselecteerde_koers}: **{df_m['Punten'].sum()}***")
                            else:
                                st.info("Geen punten gescoord in deze koers.")

# --- VOLLEDIGE SELECTIES WEERGAVE ---
st.divider()
st.subheader("📋 Volledige Selecties per Model (Alle 23 renners)")

selectie_data = {}
for m_name, m_data in HARDCODED_TEAMS.items():
    basis = m_data['Basis']
    early = [f"{r} (Voorjaar)" for r in m_data['Early']]
    late = [f"{r} (Heuvels)" for r in m_data['Late']]
    alle_renners_team = basis + early + late
    selectie_data[m_name] = alle_renners_team

df_selecties = pd.DataFrame(selectie_data)
df_selecties.index = range(1, len(df_selecties) + 1)

st.dataframe(df_selecties, use_container_width=True)
