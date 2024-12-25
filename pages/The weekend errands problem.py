import streamlit as st
st.set_page_config(layout="wide")
st.title("Finding the shortest path to complete all my weekend errands")
st.markdown("""Some of you may recognize this as a shameless rehash of the traveling salesman problem... and to that, I say: shut up, nerd.
            
            \nAnyways, I made the following app to find the shortest path to hit up X number of locations on a map. It relies
            heavily on the Google Maps API and was originally intended for personal use, so please don't use up all of my API points...""")