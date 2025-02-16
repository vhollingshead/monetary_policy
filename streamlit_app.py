import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Monetary Policy Dashboard_hellooo',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

import streamlit as st

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "About", "Our Solution", "Demo (MVP)", "Use Case"])
    
    if selection == "Our Product goes here":
        demo()
    elif selection == "Home":
        home()
    elif selection == "About":
        about()
    elif selection == "Our Solution":
        our_solution()
    
    elif selection == "Use Case":
        use_case()

def home():
    st.title("Home")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")

def about():
    st.title("About")
    st.write("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

def our_solution():
    st.title("Our Solution")
    st.write("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.")
    
    st.subheader("High Level Overview")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    
    st.subheader("Methodology")
    st.write("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")
    
    st.subheader("Data pipeline")
    st.write("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.")
    
    st.subheader("Training dataset (if applicable)")
    st.write("Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
    
    st.subheader("Data pre-processing")
    st.write("Curabitur pretium tincidunt lacus. Nulla gravida orci a odio.")
    
    st.subheader("Modelling")
    st.write("Nullam varius, turpis et commodo pharetra, est eros bibendum elit.")
    
    st.subheader("Model Evaluation")
    st.write("Donec odio tempus molestie, porttitor ut, iaculis quis, sem.")
    
    st.subheader("Key Learnings & Impact")
    st.write("Maecenas libero. Curabitur suscipit suscipit tellus.")
    
    st.subheader("Key Contributions")
    st.write("Sed lectus. Integer euismod lacus luctus magna.")
    
    st.subheader("Future Work")
    st.write("Vivamus quis mi. Phasellus a est.")
    
    st.subheader("Acknowledgements")
    st.write("Pellentesque dapibus hendrerit tortor. Praesent egestas tristique nibh.")

def demo():
    st.title("Demo (MVP)")
    st.write("Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

def use_case():
    st.title("Use Case")
    st.write("Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit.")

if __name__ == "__main__":
    main()
