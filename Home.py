import streamlit as st
import insert_logo


st.set_page_config(
    page_title="WithBrother-AI Home",
    page_icon = "🤖"
)

insert_logo.add_logo("withbrother_logo.png")


st.title("WithBrother-AI Home")

st.markdown(
    """
# Hello!
            
Welcome to WithBrother-AI PLATFORM!
            
Here are the apps We can use:
            
- [x] [T&D Assistant](/T&D_Assistant)
- [x] [Script Maker](/Script_Maker)
- [ ] [CardNews Reporter](/CardNews_Reporter)
- [ ] [Combination](/Combination)
- [ ] [Q&A](/Q&A)
- [ ] [Review-Answer](/Review-Answer)
- [ ] [Email Assistant](/Email_Assistant)
"""
)
