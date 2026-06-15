'''
Author: Kabir Ahmed
Email: beingkabirahmed@gmail.com
Date: 2026-April-18
'''

import pickle
import streamlit as st
import numpy as np

# ─────────────────────────────────────────────
#  Page Config & Theme
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="InkLogic – Smart Recommender",
    page_icon="📚",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Session State Initialisation
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "All"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# ─────────────────────────────────────────────
#  Theme CSS
# ─────────────────────────────────────────────
def inject_theme(dark: bool):
    if dark:
        bg, surface, text, accent, card = "#0f0f14", "#1c1c26", "#e8e6f0", "#7c6af7", "#252533"
    else:
        bg, surface, text, accent, card = "#f5f3ff", "#ffffff", "#1a1535", "#5b48e8", "#ede9ff"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {bg} !important;
            color: {text} !important;
            font-family: 'Inter', sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background-color: {surface} !important;
        }}
        h1, h2, h3 {{
            font-family: 'Syne', sans-serif !important;
            color: {text} !important;
        }}
        .stButton > button {{
            background: {accent};
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1.4rem;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            transition: opacity 0.2s;
        }}
        .stButton > button:hover {{ opacity: 0.85; }}
        .stSelectbox > div > div, .stTextInput > div > div > input {{
            background-color: {card} !important;
            color: {text} !important;
            border-radius: 8px !important;
        }}
        .book-card {{
            background: {card};
            border-radius: 14px;
            padding: 14px 10px 18px;
            text-align: center;
            transition: transform 0.2s;
            height: 100%;
        }}
        .book-card:hover {{ transform: translateY(-4px); }}
        .book-title {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.82rem;
            color: {text};
            margin-top: 10px;
            line-height: 1.35;
        }}
        .book-meta {{
            font-size: 0.75rem;
            color: {accent};
            margin-top: 4px;
        }}
        .fav-badge {{
            background: {accent};
            color: #fff;
            border-radius: 20px;
            padding: 2px 10px;
            font-size: 0.72rem;
            font-weight: 600;
            display: inline-block;
            margin-top: 6px;
        }}
        .section-title {{
            font-family: 'Syne', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            margin: 1.8rem 0 0.8rem;
            color: {text};
        }}
        div[data-testid="stImage"] img {{
            border-radius: 10px;
            width: 100%;
            height: 180px;
            object-fit: cover;
        }}
        .pill {{
            display: inline-block;
            background: {accent}22;
            color: {accent};
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 2px;
        }}
    </style>
    """, unsafe_allow_html=True)

inject_theme(st.session_state.dark_mode)

# ─────────────────────────────────────────────
#  Load Artifacts
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model       = pickle.load(open('artifacts/model.pkl', 'rb'))
    book_names  = pickle.load(open('artifacts/book_names.pkl', 'rb'))
    final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
    book_pivot  = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
    return model, book_names, final_rating, book_pivot

model, book_names, final_rating, book_pivot = load_artifacts()

# ─────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────
def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion:
        name = book_pivot.index[book_id]
        ids  = np.where(final_rating['title'] == name)[0][0]
        poster_url.append(final_rating.iloc[ids]['image_url'])
    return poster_url


def fetch_book_details(book_name: str) -> dict:
    """Return author, year, rating for a given title (if columns exist)."""
    row = final_rating[final_rating['title'] == book_name]
    if row.empty:
        return {}
    row = row.iloc[0]
    details = {}
    for col, key in [('book_author', 'Author'), ('year_of_publication', 'Year'),
                     ('avg_rating', 'Avg Rating'), ('genre', 'Genre')]:
        if col in final_rating.columns:
            details[key] = row[col]
    return details


def recommend_book(book_name: str):
    book_id  = np.where(book_pivot.index == book_name)[0][0]
    _, suggestion = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6
    )
    books_list, poster_urls = [], []
    for i in range(len(suggestion)):
        for title in book_pivot.index[suggestion[i]]:
            books_list.append(title)
    # fetch posters per title (flat suggestion indices)
    for book_id_inner in suggestion[0]:
        name = book_pivot.index[book_id_inner]
        ids  = np.where(final_rating['title'] == name)[0][0]
        poster_urls.append(final_rating.iloc[ids]['image_url'])
    return books_list, poster_urls


def get_genres() -> list:
    if 'genre' in final_rating.columns:
        genres = sorted(final_rating['genre'].dropna().unique().tolist())
        return ["All"] + genres
    return ["All"]


def filter_books_by_genre(names: list, genre: str) -> list:
    if genre == "All" or 'genre' not in final_rating.columns:
        return names
    valid = set(final_rating[final_rating['genre'] == genre]['title'].tolist())
    return [b for b in names if b in valid]


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Dark / Light Toggle
    mode_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(mode_label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.divider()

    # Genre Filter
    st.markdown("### 🏷️ Genre Filter")
    genres = get_genres()
    st.session_state.selected_genre = st.selectbox(
        "Filter books by genre", genres,
        index=genres.index(st.session_state.selected_genre)
              if st.session_state.selected_genre in genres else 0,
        key="genre_select"
    )

    st.divider()

    # Favorites Panel
    st.markdown("### ❤️ Favourites")
    if st.session_state.favorites:
        for fav in st.session_state.favorites:
            col_f1, col_f2 = st.columns([4, 1])
            col_f1.markdown(f"<span style='font-size:0.82rem'>{fav}</span>",
                            unsafe_allow_html=True)
            if col_f2.button("✕", key=f"rm_{fav}"):
                st.session_state.favorites.remove(fav)
                st.rerun()
    else:
        st.caption("No favourites yet. Click ♡ on any card to save.")

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown("# 📚 BookLens")
st.markdown("*Discover your next great read with ML-powered recommendations*")
st.divider()

# ─────────────────────────────────────────────
#  Search + Dropdown
# ─────────────────────────────────────────────
col_search, col_genre_badge = st.columns([3, 1])

with col_search:
    search_query = st.text_input(
        "🔍 Search books", placeholder="Type to filter the dropdown…",
        value=st.session_state.search_query, key="search_input"
    )
    st.session_state.search_query = search_query

# Filter book list by genre + search text
filtered_names = filter_books_by_genre(list(book_names), st.session_state.selected_genre)
if search_query:
    filtered_names = [b for b in filtered_names if search_query.lower() in b.lower()]

with col_genre_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<span class='pill'>🏷️ {st.session_state.selected_genre}</span> "
        f"<span class='pill'>{len(filtered_names)} books</span>",
        unsafe_allow_html=True
    )

if not filtered_names:
    st.warning("No books match your search / genre filter. Try adjusting them.")
    st.stop()

selected_book = st.selectbox(
    "Select or type a book title", filtered_names, key="book_select"
)

# ─────────────────────────────────────────────
#  Book Detail Expander for Selected Book
# ─────────────────────────────────────────────
details = fetch_book_details(selected_book)
if details:
    with st.expander(f"ℹ️ Details for **{selected_book}**", expanded=False):
        d_cols = st.columns(len(details))
        for i, (k, v) in enumerate(details.items()):
            d_cols[i].metric(k, v)

# ─────────────────────────────────────────────
#  Recommend Button
# ─────────────────────────────────────────────
if st.button("✨ Show Recommendations", use_container_width=False):
    if selected_book not in book_pivot.index:
        st.error("Selected book not found in the model index. Please choose another.")
    else:
        recommended_books, poster_urls = recommend_book(selected_book)
        # skip index 0 (the selected book itself)
        recs = list(zip(recommended_books[1:6], poster_urls[1:6]))

        st.markdown("<div class='section-title'>Recommended for you</div>",
                    unsafe_allow_html=True)
        cols = st.columns(5)
        for idx, (col, (title, url)) in enumerate(zip(cols, recs)):
            with col:
                is_fav = title in st.session_state.favorites
                st.markdown(f"""
                    <div class='book-card'>
                        <img src='{url}' style='width:100%;height:180px;
                             object-fit:cover;border-radius:10px;'/>
                        <div class='book-title'>{title}</div>
                """, unsafe_allow_html=True)

                # Book detail inline
                rec_details = fetch_book_details(title)
                if rec_details.get("Author"):
                    st.markdown(
                        f"<div class='book-meta'>✍️ {rec_details['Author']}</div>",
                        unsafe_allow_html=True
                    )
                if rec_details.get("Avg Rating"):
                    st.markdown(
                        f"<div class='book-meta'>⭐ {rec_details['Avg Rating']}</div>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                # Favourite toggle button
                fav_label = "❤️ Saved" if is_fav else "🤍 Save"
                if st.button(fav_label, key=f"fav_{idx}_{title}"):
                    if is_fav:
                        st.session_state.favorites.remove(title)
                    else:
                        if title not in st.session_state.favorites:
                            st.session_state.favorites.append(title)
                    st.rerun()