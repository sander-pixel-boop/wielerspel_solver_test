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
        Dit is jouw centrale dashboard voor het berekenen van de ultieme selecties voor de voorjaarsklassiekers en grote ronden.
        
        👈 **Kies een spel in het menu aan de linkerkant om te beginnen!**
        
        ### Beschikbare Solvers:
        * **Scorito Klassiekers:** Optimaliseer je selectie met het 45M budget en bereken de perfecte wisselstrategie na Parijs-Roubaix.
        * **Cycling Fantasy:** Bereken het optimale dagteam per koers op basis van ingeladen PCS-startlijsten en de vaste actuele credits.
        * **🚧 WORK IN PROGRESS (WIP) 🚧 Sporza Wielermanager:** Bouw je team binnen de limieten van 120M, 20 renners en maximaal 4 per ploeg.
        * *Binnenkort: Scorito en Sporza voor de Grote Ronden!*
        
        ---
        
        ### 🙏 Credits & Databronnen
        Deze applicatie is gebouwd op de schouders van de fantastische wielercommunity. Veel dank aan:
        * **[Wielerorakel.nl](https://www.cyclingoracle.com/):** Voor het leveren van de Stats van de renners.
        * **[Kopmanpuzzel](https://kopmanpuzzel.up.railway.app/):** Voor het uitstekende voorwerk rondom de startlijsten en de actuele Scorito-prijzen.
        """
    )

# 2. Definieer alle pagina's
home = st.Page(home_page, title="Home", icon="🏠", default=True)

# Let op: vul hieronder jouw eigen bestandsnamen in!
cf_pagina = st.Page("pages/naam_van_cf_bestand.py", title="CF Dashboard", icon="🚴")
scorito_klassiekers = st.Page("pages/naam_van_klassieker_bestand.py", title="Klassieker App", icon="🏆")
scorito_evaluator = st.Page("pages/Model_Evaluator_(Scorito).py", title="Model Evaluator", icon="📊")
sporza_pagina = st.Page("pages/naam_van_sporza_bestand.py", title="Sporza Dashboard", icon="🏁")

# 3. Groepeer de navigatie voor de sidebar
pg = st.navigation({
    "Info": [home],
    "Cycling Fantasy": [cf_pagina],
    "Scorito": [scorito_klassiekers, scorito_evaluator],
    "Sporza": [sporza_pagina]
})

# 4. Voer de applicatie uit
pg.run()
