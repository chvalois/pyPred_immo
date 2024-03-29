from collections import OrderedDict

import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import _intro, iii_prix_features, iv_annonce, iv_demo


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (_intro.sidebar_name, _intro),
        (iv_annonce.sidebar_name, iv_annonce),
        (iii_prix_features.sidebar_name, iii_prix_features),          
        (iv_demo.sidebar_name, iv_demo)
    ]
)


def run():

    st.sidebar.markdown("pyPredImmo")
    tab_name = st.sidebar.radio("", list(TABS.keys()), index = 0)
    st.sidebar.markdown("---")

    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
