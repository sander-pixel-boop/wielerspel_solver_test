import streamlit as st

# 1. Paginaconfiguratie (MOET als eerste)
st.set_page_config(page_title="Wieler Spellen Solver", page_icon="🚴‍♂️")

# 2. Inlog Systeem
def check_password():
    def password_entered():
        user = st.session_state["username_input"].strip()
        pwd = st.session_state["password_input"].strip()
        
        # Controleer of de gebruiker bestaat en het wachtwoord klopt via de Streamlit Secrets
        if user in st.secrets.get("passwords", {}) and pwd == str(st.secrets["passwords"][user]):
            st.session_state["password_correct"] = True
            st.session_state["ingelogde_speler"] = user
            del st.session_state["password_input"]  # Wis wachtwoord uit geheugen voor veiligheid
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.write("# 🔒 Log in om verder te gaan")
    st.text_input("Gebruikersnaam", key="username_input")
    st.text_input("Wachtwoord", type="password", key="password_input")
    st.button("Inloggen", on_click=password_entered, type="primary")

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("❌ Gebruikersnaam of wachtwoord onjuist.")
    return False

# Stop het script als de speler niet is ingelogd
if not check_password():
    st.stop()

# 3. De Homepagina (Alleen zichtbaar na inloggen)
def home_page():
    speler = st.session_state["ingelogde_speler"].capitalize()
    st.write(f"# Welkom bij de Wieler Spellen Solver, {speler}! 🚴‍♂️")

    st.markdown(
        """
        Dit is jouw centrale dashboard voor het berekenen van de ultieme selecties voor wielerspellen.
        
        👈 **Kies een spel in het menu aan de linkerkant om te beginnen!**
        
        ### Beschikbare Solvers & Modules:
        
        **Cycling Fantasy**
        * **CF Dashboard:** Bereken het optimale dagteam per koers op basis van ingeladen PCS-startlijsten en de actuele credits.
        
        **Scorito**
        * **Klassiekers:** Optimaliseer je selectie met het 45M budget en bereken de perfecte wisselstrategie na Parijs-Roubaix.
        * **Grand Tour:** *(Binnenkort)* Bereken je ideale selectie voor de grote rondes.
        * **Evaluator:** Test en vergelijk live hoe verschillende wiskundige modellen presteren ten opzichte van jouw eigen selectie.

        **Sporza**
        * **Klassiekers:** Bouw je team binnen de limieten van 120M, 20 renners en maximaal 4 per ploeg, inclusief de 12-starters regel.
        * **Grand Tour:** *(Binnenkort)* Optimaliseer je Sporza-team voor de grote rondes.
        * **Evaluator:** *(Binnenkort)* Test en vergelijk Sporza modellen in de praktijk.
        
        **Vriendencompetitie**
        * **Eigen Spel (Custom):** 🎮 Speel je eigen custom spel met vrienden! Kies 10 vaste renners, doe 2 transfers en selecteer per koers je 3 extra's en je Joker.
        
        ---
        
        ### 🙏 Credits & Databronnen
        Deze applicatie is gebouwd op de schouders van de fantastische wielercommunity. Veel dank aan (bronvermelding):
        * **[Wielerorakel.nl](https://www.cyclingoracle.com/):** Voor het leveren van de Stats van de renners.
        * **[Kopmanpuzzel](https://kopmanpuzzel.up.railway.app/):** Voor het uitstekende voorwerk rondom de startlijsten en de actuele Scorito-prijzen.
        """
    )

# 4. Pagina Navigatie
home = st.Page(home_page, title="Home", icon="🏠", default=True)

cf_pagina = st.Page("pages/Cycling_Fantasy.py", title="CF Dashboard", icon="🚴")

scorito_klassiekers = st.Page("pages/Klassiekers - Scorito.py", title="Klassiekers", icon="🏆")
scorito_grand_tour = st.Page("pages/Scorito_Grand_Tour.py", title="[Binnenkort] Grand Tour", icon="⛰️")
scorito_evaluator = st.Page("pages/Model_Evaluator_(Scorito).py", title="Evaluator", icon="📊")

sporza_klassiekers = st.Page("pages/Klassiekers - Sporza.py", title="Klassiekers", icon="🏁")
sporza_grand_tour = st.Page("pages/Sporza_Grand_Tour.py", title="[Binnenkort] Grand Tour", icon="⛰️")
sporza_evaluator = st.Page("pages/Sporza_Evaluator.py", title="[Binnenkort] Evaluator", icon="📊")

eigen_spel = st.Page("pages/Het_Spel.py", title="Custom Klassiekers Spel", icon="🎮")

pg = st.navigation({
    "Info": [home],
    "Cycling Fantasy": [cf_pagina],
    "Scorito": [scorito_klassiekers, scorito_grand_tour, scorito_evaluator],
    "Sporza": [sporza_klassiekers, sporza_grand_tour, sporza_evaluator],
    "Eigen Competitie": [eigen_spel]
})

pg.run()
