import pandas as pd
import streamlit as st

from recommender import MusicRecommender

# Page configuration
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Design
st.markdown("""
    <style>
        html, body {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        header {
            display: none !important;
        }
        
        .stAppHeader {
            display: none !important;
        }
        
        [data-testid="stHeader"] {
            display: none !important;
        }
        
        .css-1y4p8pa {
            display: none !important;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #f4f7fe 0%, #e0e7ff 100%);
            font-family: 'Inter', 'Roboto', -apple-system, sans-serif;
        }
        
        .main .block-container {
            background: transparent;
            padding: 3rem 4rem;
            max-width: 1000px;
            margin: 0 auto;
        }
        
        /* Header Styling */
        .header-container {
            text-align: center;
            margin-bottom: 3.5rem;
            padding: 1rem 0;
        }
        
        .header-container h1 {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -1.5px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.8rem;
        }
        
        .header-container p {
            font-size: 1.2rem;
            color: #4b5563;
            font-weight: 400;
            letter-spacing: 0.2px;
        }
        
        /* Input Styling */
        .stTextInput > div > div > input {
            border-radius: 16px;
            border: 2px solid white;
            padding: 1rem 1.5rem;
            font-size: 1.15rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            background-color: white;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #818cf8;
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.2);
            outline: none;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 16px !important;
            padding: 1rem 2rem !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            cursor: pointer !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
            height: 100% !important;
            width: 100% !important;
            min-height: 52px !important;
        }
        
        .stButton > button:hover {
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
            transform: translateY(-2px) !important;
            background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
            color: white !important;
        }
        
        .stButton > button:active {
            transform: translateY(1px) !important;
        }
        
        /* Subheader Styling */
        .subheader {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1f2937;
            margin-top: 3rem;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 0.8rem;
        }
        
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1.5rem;
            }
            .header-container h1 {
                font-size: 2.5rem;
            }
            .header-container p {
                font-size: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def _load_spotify_songs() -> pd.DataFrame:
    """Load songs from Spotify CSV file and prepare data for recommendation."""
    df = pd.read_csv("spotify_songs.csv")
    # Select required columns and rename for consistency
    songs_df = df[["track_name", "tempo", "energy", "valence"]].copy()
    songs_df.columns = ["name", "tempo", "energy", "valence"]
    # Remove any rows with missing values
    songs_df = songs_df.dropna()
    return songs_df


# Initialize session state
if "songs_df" not in st.session_state:
    st.session_state.songs_df = _load_spotify_songs()
    st.session_state.recommender = MusicRecommender(st.session_state.songs_df)

songs_df = st.session_state.songs_df
recommender = st.session_state.recommender

# Header
st.markdown("""
    <div class="header-container">
        <h1>🎵 Music Recommender</h1>
        <p>Discover songs similar to your favorites based on audio features</p>
    </div>
""", unsafe_allow_html=True)

# Search Input Section
search_col, btn_col = st.columns([4, 1])

with search_col:
    song_name = st.text_input(
        "Song Name",
        value="",
        placeholder="Enter a song title (e.g., 'Shape of You')...",
        label_visibility="collapsed"
    )

with btn_col:
    recommend_btn = st.button("Search", use_container_width=True)

# Process search
if recommend_btn or song_name:
    if song_name.strip():
        song_query = song_name.strip()
        top_n_value = 5  # Fixed number instead of slider
        
        try:
            # Try exact match first
            recs = recommender.get_similar_songs(song_query, top_n=top_n_value)
            
            # Subheader
            st.markdown(
                f'<h2 class="subheader">✨ Recommended Songs for "<strong>{song_query}</strong>"</h2>',
                unsafe_allow_html=True
            )
            
            # Helper to return formatted styles for tags
            def get_match_quality(score):
                if score >= 0.90: 
                    return "🔥 Best Match", "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)", "#b91c1c"
                elif score >= 0.80: 
                    return "✨ Great Match", "linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)", "#4338ca"
                elif score >= 0.65: 
                    return "👍 Good Match", "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)", "#15803d"
                else: 
                    return "🎵 Fair Match", "linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)", "#374151"
            
            html_content = '<div style="display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem;">'
            
            for idx, row in recs.iterrows():
                song = row["name"]
                score = row["similarity"]
                tag, bg, color = get_match_quality(score)
                
                # We construct the HTML without any leading whitespace to prevent 
                # Streamlit's markdown parser from treating it as a code block.
                html_content += f"""
<div style="background: white; border-radius: 16px; padding: 1.5rem 2rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid #f3f4f6; transition: all 0.2s ease;"
     onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.08)';"
     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.03)';">
    <div style="font-size: 1.25rem; font-weight: 600; color: #1f2937; display: flex; align-items: center; gap: 1rem;">
        <span style="background: #f3f4f6; padding: 0.5rem; border-radius: 50%; font-size: 1.2rem;">🎶</span>
        {song}
    </div>
    <div style="background: {bg}; border-radius: 30px; padding: 0.5rem 1.2rem; font-size: 0.9rem; font-weight: 700; color: {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05); letter-spacing: 0.3px;">
        {tag}
    </div>
</div>"""
                
            html_content += '</div>'
            st.markdown(html_content, unsafe_allow_html=True)
            
        except ValueError:
            # Show user-friendly error message without empty stream containers
            st.markdown(f"""
                <div style="background: white; border-radius: 16px; padding: 3rem 2rem; text-align: center; border: 1px solid #fecaca; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-top: 2rem;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">😕</div>
                    <h3 style="color: #ef4444; margin-bottom: 1rem; font-size: 1.8rem; font-weight: 700;">Song Not Found</h3>
                    <p style="font-size: 1.15rem; color: #4b5563; margin-bottom: 0.5rem;">
                        We couldn't find "<strong>{song_query}</strong>" in our database.
                    </p>
                    <p style="color: #6b7280; margin-top: 1rem; font-size: 1rem; background: #f9fafb; display: inline-block; padding: 0.8rem 1.5rem; border-radius: 12px;">
                        💡 <strong>Tip:</strong> Try searching for popular artists or well-known tracks. Check the exact spelling!
                    </p>
                </div>
            """, unsafe_allow_html=True)
