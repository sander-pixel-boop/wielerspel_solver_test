import streamlit as st

# --- CONFIGURATIE ---
st.set_page_config(page_title="Sporza Wielermanager (WIP)", layout="wide", page_icon="🦁")

st.title("🦁 Sporza Wielermanager - Team Builder")

# --- WIP MELDING ---
st.warning("🚧 **WORK IN PROGRESS (WIP) - NOG NIET BESCHIKBAAR** 🚧\n\nDeze module is momenteel in aanbouw en nog niet bruikbaar. We wachten op de dataset met de officiële Sporza-prijzen en de juiste ploegindelingen voor dit seizoen.")

st.info("Zodra de actuele prijzen en startlijsten bekend zijn, wordt de rekenmodule hier geactiveerd. Je kunt dan je optimale team van 120 miljoen laten berekenen.")

st.divider()

st.markdown("""
### Wat je straks hier kunt verwachten:
De Sporza Wielermanager werkt wiskundig heel anders dan Scorito. Zodra de tool live gaat, houdt het algoritme automatisch rekening met de volgende Sporza-regels:

* **Budget:** Je bouwt een team met **€120 miljoen** in plaats van €45 miljoen.
* **Teamgrootte:** Exact **20 renners** in je selectie (12 starters, 8 in de bus).
* **Ploegenlimiet:** Maximaal **4 renners per ploeg** (dit is een harde eis in de solver).
* **Puntentelling:** Koersen hebben een unieke weging (Monument = 125pt, WT = 100pt, etc.).
* **Teampunten:** De tool berekent de impact van de 10 bonuspunten voor ploegmaats van een winnaar.
* **Transfers:** Het trapsgewijze transfersysteem (eerst gratis, daarna miljoenen aftrek van je budget) wordt wiskundig geoptimaliseerd.
""")
