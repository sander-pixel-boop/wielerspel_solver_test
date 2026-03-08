import streamlit as st
import hashlib
from supabase import create_client

st.set_page_config(page_title="Wieler Spellen Solver", page_icon="🚴‍♂️")

# --- DATABASE CONNECTIE ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()
TABEL_NAAM = "gebruikers_data_test"

def hash_wachtwoord(wachtwoord):
    return hashlib.sha256(wachtwoord.encode()).hexdigest()

# --- AUTHENTICATIE ---
if "ingelogde_speler" not in st.session_state:
    st.title("🔒 Welkom! Log in of maak een account")
    tab1, tab2 = st.tabs(["Inloggen", "Account Aanmaken"])
    
    with tab1:
        inlog_naam = st.text_input("Gebruikersnaam", key="inlog_naam")
        inlog_ww = st.text_input("Wachtwoord", type="password", key="inlog_ww")
        if st.button("Inloggen", type="primary"):
            if inlog_naam and inlog_ww:
                res = supabase.table(TABEL_NAAM).select("password").eq("username", inlog_naam.lower()).execute()
                if res.data and res.data[0].get("password") == hash_wachtwoord(inlog_ww):
                    st.session_state["ingelogde_speler"] = inlog_naam.lower()
                    st.rerun()
                else:
                    st.error("❌ Onjuiste gebruikersnaam of wachtwoord.")
            else:
                st.warning("Vul beide velden in.")
                
    with tab2:
        nieuw_naam = st.text_input("Kies een Gebruikersnaam", key="nieuw_naam")
        nieuw_ww = st.text_input("Kies een Wachtwoord", type="password", key="nieuw_ww")
        if st.button("Maak account aan"):
            if nieuw_naam and nieuw_ww:
                bestaat_al = supabase.table(TABEL_NAAM).select("username").eq("username", nieuw_naam.lower()).execute()
                if bestaat_al.data:
                    st.error("❌ Deze gebruikersnaam is al in gebruik. Kies een andere.")
                else:
                    try:
                        supabase.table(TABEL_NAAM).insert({
                            "username": nieuw_naam.lower(),
                            "password": hash_wachtwoord(nieuw_ww)
                        }).execute()
                        st.success("✅ Account succesvol aangemaakt! Je kunt nu inloggen.")
                    except Exception as e:
                        st.error(f"Fout bij aanmaken account: {e}")
            else:
                st.warning("Vul beide velden in.")
    
    st.divider()
    if st.button("Doorgaan als gast (zonder account)"):
        st.session_state["ingelogde_speler"] = "gast"
        st.rerun()
        
    st.stop()

# --- HOME PAGINA & NAVIGATIE ---
def home_page():
    speler = st.session_state.get("ingelogde_speler", "bezoeker").capitalize()
    st.write(f"# Welkom bij de Wieler Spellen Solver, {speler}! 🚴‍♂️")
    st.markdown("👈 **Kies een spel in het menu aan de linkerkant om te beginnen!**")
    
    if st.button("Uitloggen"):
        del st.session_state["ingelogde_speler"]
        st.rerun()

home = st.Page(home_page, title="Home", icon="🏠", default=True)
cf_pagina = st.Page("pages/Cycling_Fantasy.py", title="CF Dashboard", icon="🚴")

# Scorito pagina's
scorito_klassiekers = st.Page("pages/Klassiekers - Scorito.py", title="Klassiekers", icon="🏆")
scorito_evaluator = st.Page("pages/Model_Evaluator_(Scorito).py", title="Evaluator", icon="📊")
scorito_giro = st.Page("pages/Scorito_Grand_Tour.py", title="Giro d'Italia", icon="🇮🇹")

# Sporza pagina's
sporza_klassiekers = st.Page("pages/Klassiekers - Sporza.py", title="Klassiekers", icon="🏁")
sporza_evaluator = st.Page("pages/Sporza_Evaluator.py", title="Evaluator", icon="📊")
# LET OP: Het Giro bestand moet exact 'Sporza_Giro.py' heten in je map 'pages'!
sporza_giro = st.Page("pages/Sporza_Giro.py", title="Giro d'Italia", icon="🇮🇹")

eigen_spel = st.Page("pages/Het_Spel.py", title="Custom Klassiekers Spel", icon="🎮")

pg = st.navigation({
    "Info": [home],
    "Cycling Fantasy": [cf_pagina],
    "Scorito - Klassiekers": [scorito_klassiekers, scorito_evaluator],
    "Scorito - Grand Tours": [scorito_giro],
    "Sporza - Klassiekers": [sporza_klassiekers, sporza_evaluator],
    "Sporza - Grand Tours": [sporza_giro],
    "Eigen Competitie": [eigen_spel]
})

pg.run()
