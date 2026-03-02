import streamlit as st

# 1. Definieer de inhoud van de homepagina in een functie
def home_page():
    st.set_page_config(
        page_title="Wieler Spellen Solver",
        page_icon="🚴‍♂️",
    )

    st.write("# Welkom bij de Wieler Spellen Solver! 🚴‍♂️")

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
        * **Klassiekers:** 🚧 *(WIP)* Bouw je team binnen de limieten van 120M, 20 renners en maximaal 4 per ploeg.
        * **Grand Tour:** *(Binnenkort)* Optimaliseer je Sporza-team voor de grote rondes.
        * **Evaluator:** *(Binnenkort)* Test en vergelijk Sporza modellen in de praktijk.
        
        ---
        
        ### 🙏 Credits & Databronnen
        Deze applicatie is gebouwd op de schouders van de fantastische wielercommunity. Veel dank aan:
        * **[Wielerorakel.nl](https://www.cyclingoracle.com/):** Voor het leveren van de Stats van de renners.
        * **[Kopmanpuzzel](https://kopmanpuzzel.up.railway.app/):** Voor het uitstekende voorwerk rondom de startlijsten en de actuele Scorito-prijzen.
        """
    )

# 2. Definieer alle pagina's met de exacte, eenduidige bestandsnamen en titels
home = st.Page(home_page, title="Home", icon="🏠", default=True)

cf_pagina = st.Page("pages/Cycling_Fantasy.py", title="CF Dashboard", icon="🚴")

scorito_klassiekers = st.Page("pages/Klassiekers - Scorito.py", title="Klassiekers", icon="🏆")
scorito_grand_tour = st.Page("pages/Scorito_Grand_Tour.py", title="Grand Tour", icon="⛰️")
scorito_evaluator = st.Page("pages/Model_Evaluator_(Scorito).py", title="Evaluator", icon="📊")

sporza_klassiekers = st.Page("pages/Klassiekers - Sporza.py", title="Klassiekers", icon="🏁")
sporza_grand_tour = st.Page("pages/Sporza_Grand_Tour.py", title="Grand Tour", icon="⛰️")
sporza_evaluator = st.Page("pages/Sporza_Evaluator.py", title="Evaluator", icon="📊")

# 3. Groepeer de navigatie voor de sidebar
pg = st.navigation({
    "Info": [home],
    "Cycling Fantasy": [cf_pagina],
    "Scorito": [scorito_klassiekers, scorito_grand_tour, scorito_evaluator],
    "Sporza": [sporza_klassiekers, sporza_grand_tour, sporza_evaluator]
})

# 4. Voer de applicatie uit
pg.run()
