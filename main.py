import streamlit as st
import warnings
import os
import numpy as np
import pandas as pd

from analyzer import FootballAnalyzer
from models import VideoInfo, KEY_ANALYSIS_COMPLETE, KEY_VIDEO_INFO

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the football analyzer application
    """
    # Initialize and run the football analyzer
    analyzer = FootballAnalyzer(st)
    analyzer.run()

if __name__ == "__main__":
    # Set page config before any other Streamlit commands
    st.set_page_config(
        page_title="Football Match Analysis", 
        page_icon="⚽", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set the page title and description
    main()