import streamlit as st
import hashlib
from supabase import create_client

st.set_page_config(page_title="Wieler Spellen Solver", page_icon="🚴‍♂️")

# Bron: Supabase Python SDK documentatie
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()
TABEL_NAAM = "gebruikers_data_test"

def hash_wachtwoord(wachtwoord):
    return hashlib.sha256(wachtwoord.encode()).hexdigest()

if "ingelogde_speler" not in st.session_state:
    st.title("🔒 Welkom! Log in of maak een account")
    tab1, tab2 = st.tabs(["Inloggen", "Account Aanmaken"])
    
    with tab1:
        inlog_naam = st.text_input("Gebruikersnaam", key="inlog_naam")
        inlog_ww = st.text_input("Wachtwoord", type="password", key="inlog_ww")
        if st.button("Log in", type="primary"):
            if inlog_naam and inlog_ww:
                res = supabase.table(TABEL_NAAM).select("password").eq("username", inlog_naam.lower()).execute()
                if res.data and res.data[0].get("password") == hash_wachtwoord(inlog_ww):
                    st.session_state["ingelogde_speler"] = inlog_naam.lower()
                    st.rerun()
                else:
                    st.error("Onjuiste gebruikersnaam of wachtwoord.")
            else:
                st.warning("Vul beide velden in.")
                
    with tab2:
        nieuw_naam = st.text_input("Kies een Gebruikersnaam", key="nieuw_naam")
        nieuw_ww = st.text_input("Kies een Wachtwoord", type="password", key="nieuw_ww")
        if st.button("Maak account aan"):
            if nieuw_naam and nieuw_ww:
                bestaat_al = supabase.table(TABEL_NAAM).select("username").eq("username", nieuw_naam.lower()).execute()
                if bestaat_al.data:
                    st.error("Deze gebruikersnaam is al in gebruik. Kies een andere.")
                else:
                    try:
                        supabase.table(TABEL_NAAM).insert({
                            "username": nieuw_naam.lower(),
                            "password": hash_wachtwoord(nieuw_ww)
                        }).execute()
                        st.success("Account succesvol aangemaakt! Je kunt nu inloggen via het andere tabblad.")
                    except Exception as e:
                        st.error(f"Fout bij aanmaken account: {e}")
            else:
                st.warning("Vul beide velden in.")
                
    st.stop()

st.write(f"# Welkom, {st.session_state['ingelogde_speler'].capitalize()}! 🚴‍♂️")

if st.button("Uitloggen"):
    del st.session_state["ingelogde_speler"]
    st.rerun()

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
    * **[Wielerorakel.nl](https://www.cyclingoracle.com/):** Voor het leveren van de AI-gebaseerde Skill-scores van de renners.
    * **[Kopmanpuzzel](https://kopmanpuzzel.up.railway.app/):** Voor het uitstekende voorwerk rondom de startlijsten en de actuele Scorito-prijzen.
    """
)
