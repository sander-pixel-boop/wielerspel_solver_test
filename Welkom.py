import streamlit as st

# 1. Paginaconfiguratie (MOET als eerste regel code)
st.set_page_config(page_title="Wieler Spellen Solver", page_icon="🚴‍♂️")

# 2. Inlog Systeem
def check_password():
    def password_entered():
        user = st.session_state["username_input"].strip()
        pwd = st.session_state["password_input"].strip()
        
        if user in st.secrets.get("passwords", {}) and pwd == str(st.secrets["passwords"][user]):
            st.session_state["password_correct"] = True
            st.session_state["ingelogde_speler"] = user
            if "password_input" in st.session_state:
                del st.session_state["password_input"]
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

if not check_password():
    st.stop()

# 3. Homepagina Definitie
def home_page():
    speler = st.session_state.get("ingelogde_speler", "bezoeker").capitalize()
    st.write(f"# Welkom bij de Wieler Spellen Solver, {speler}! 🚴‍♂️")
    st.markdown("👈 **Kies een spel in het menu aan de linkerkant om te beginnen!**")

# 4. Pagina Navigatie
home = st.Page(home_page, title="Home", icon="🏠", default=True)
cf_pagina = st.Page("pages/Cycling_Fantasy.py", title="CF Dashboard", icon="🚴")
scorito_klassiekers = st.Page("pages/Klassiekers - Scorito.py", title="Klassiekers", icon="🏆")
sporza_klassiekers = st.Page("pages/Klassiekers - Sporza.py", title="Klassiekers", icon="🏁")
eigen_spel = st.Page("pages/Het_Spel.py", title="Custom Klassiekers Spel", icon="🎮")

pg = st.navigation({
    "Info": [home],
    "Cycling Fantasy": [cf_pagina],
    "Scorito": [scorito_klassiekers],
    "Sporza": [sporza_klassiekers],
    "Eigen Competitie": [eigen_spel]
})

pg.run()
