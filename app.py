import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InkLogic – Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── CSS Variables ── */
    :root {
        --ink: #1a1a2e;
        --parchment: #faf6ef;
        --gold: #c9a84c;
        --gold-light: #f0d98a;
        --crimson: #8b1a1a;
        --sage: #4a7c6f;
        --muted: #6b6b6b;
        --card-bg: #ffffff;
        --border: #e8e0d0;
    }

    /* ── Base Reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--parchment) !important;
        color: var(--ink);
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; }

    /* ── Hero Banner ── */
    .hero {
        background: linear-gradient(135deg, var(--ink) 0%, #2d2b55 60%, #1a1a2e 100%);
        padding: 4rem 3rem 3rem;
        border-radius: 0 0 32px 32px;
        margin: -1rem -1rem 2rem -1rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "📖";
        font-size: 18rem;
        position: absolute;
        right: -2rem;
        top: -3rem;
        opacity: 0.04;
        line-height: 1;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(201,168,76,0.15);
        border: 1px solid var(--gold);
        color: var(--gold-light);
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        font-weight: 900;
        color: #fff;
        line-height: 1.1;
        margin: 0 0 0.6rem 0;
    }
    .hero-title span { color: var(--gold); }
    .hero-subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        font-weight: 300;
        max-width: 520px;
        line-height: 1.6;
        margin: 0;
    }
    .hero-stats {
        display: flex;
        gap: 2.5rem;
        margin-top: 2rem;
        flex-wrap: wrap;
    }
    .stat-item { text-align: left; }
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--gold);
        display: block;
    }
    .stat-label {
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.45);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--ink) !important;
    }
    [data-testid="stSidebar"] * {
        color: #fff !important;
    }
    [data-testid="stSidebar"] .sidebar-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--gold) !important;
        padding: 1rem 0 0.5rem;
        border-bottom: 1px solid rgba(201,168,76,0.3);
        margin-bottom: 1.5rem;
    }
    [data-testid="stSidebar"] .sidebar-section {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4) !important;
        margin: 1.5rem 0 0.6rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* ── Section Headings ── */
    .section-label {
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.4rem;
    }
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0 0 1.5rem 0;
    }
    .section-title span { color: var(--crimson); }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background: var(--card-bg) !important;
        border: 2px solid var(--border) !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        color: var(--ink) !important;
        padding: 0.5rem 1rem !important;
        transition: border-color 0.2s;
    }
    [data-testid="stSelectbox"] > div > div:focus-within {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 3px rgba(201,168,76,0.15) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--ink), #2d2b55) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.8rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.03em !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(26,26,46,0.25) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(26,26,46,0.35) !important;
        background: linear-gradient(135deg, var(--gold), #a07830) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Book Cards ── */
    .book-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }
    .book-card {
        background: var(--card-bg);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(26,26,46,0.07);
        border: 1px solid var(--border);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        cursor: pointer;
    }
    .book-card:hover {
        transform: translateY(-6px) rotate(0.5deg);
        box-shadow: 0 12px 30px rgba(26,26,46,0.15);
    }
    .book-card-img {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        display: block;
        background: linear-gradient(135deg, #e8e0d0, #d4c9b5);
    }
    .book-card-body {
        padding: 0.7rem 0.8rem;
    }
    .book-card-title {
        font-family: 'Playfair Display', serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .book-card-badge {
        display: inline-block;
        background: linear-gradient(90deg, var(--gold), #a07830);
        color: #fff;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-top: 0.4rem;
        text-transform: uppercase;
    }

    /* ── Info Boxes ── */
    .info-box {
        background: linear-gradient(135deg, rgba(26,26,46,0.03), rgba(201,168,76,0.05));
        border: 1px solid var(--border);
        border-left: 4px solid var(--gold);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: var(--muted);
        line-height: 1.6;
    }
    .warning-box {
        background: rgba(139,26,26,0.05);
        border: 1px solid rgba(139,26,26,0.2);
        border-left: 4px solid var(--crimson);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: var(--crimson);
        line-height: 1.6;
    }
    .success-box {
        background: rgba(74,124,111,0.08);
        border: 1px solid rgba(74,124,111,0.25);
        border-left: 4px solid var(--sage);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: var(--sage);
        line-height: 1.6;
    }

    /* ── Divider ── */
    .ink-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 2rem 0;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2.5rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.8;
    }
    .footer a { color: var(--gold); text-decoration: none; }
    .footer-logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 900;
        color: var(--ink);
        margin-bottom: 0.3rem;
    }
    .footer-logo span { color: var(--gold); }

    /* ── How It Works ── */
    .how-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        height: 100%;
    }
    .how-icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
        display: block;
    }
    .how-step {
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--gold);
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .how-title {
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.5rem;
    }
    .how-desc {
        font-size: 0.8rem;
        color: var(--muted);
        line-height: 1.5;
    }

    /* ── Spinner override ── */
    .stSpinner > div {
        border-top-color: var(--gold) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Recommendation Class ───────────────────────────────────────────────────────
class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]:
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(
                book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6
            )
            poster_url = self.fetch_poster(suggestion)

            for i in range(len(suggestion)):
                books = book_pivot.index[suggestion[i]]
                for j in books:
                    books_list.append(j)
            return books_list, poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            logging.info("Training completed successfully!")
        except Exception as e:
            raise AppException(e, sys) from e


# ─── Render Recommendations ────────────────────────────────────────────────────
def render_recommendations(books_list, poster_url):
    """Render book cards in a responsive grid."""
    st.markdown('<div class="book-grid">', unsafe_allow_html=True)
    for i in range(1, 6):
        title = books_list[i] if i < len(books_list) else "Unknown"
        img = poster_url[i] if i < len(poster_url) else ""
        img_tag = (
            f'<img class="book-card-img" src="{img}" alt="{title}" onerror="this.style.display=\'none\'">'
            if img else
            '<div class="book-card-img" style="height:180px;background:linear-gradient(135deg,#e8e0d0,#c9b99a);display:flex;align-items:center;justify-content:center;font-size:3rem;">📖</div>'
        )
        st.markdown(f"""
            <div class="book-card">
                {img_tag}
                <div class="book-card-body">
                    <div class="book-card-title">{title}</div>
                    <div class="book-card-badge">Recommended</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">📚 InkLogic</div>', unsafe_allow_html=True)
    st.markdown(
        "Discover your next great read with collaborative filtering — powered by 270,000+ real reader ratings.",
        unsafe_allow_html=True
    )
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">About the Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:rgba(255,255,255,0.55); line-height:1.7;">
        📦 <b style="color:rgba(255,255,255,0.8)">BX-Books</b> — 271,360 titles<br>
        👤 <b style="color:rgba(255,255,255,0.8)">BX-Users</b> — 278,858 readers<br>
        ⭐ <b style="color:rgba(255,255,255,0.8)">BX-Ratings</b> — 1,149,780 ratings
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Algorithm</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:rgba(255,255,255,0.55); line-height:1.7;">
        🤖 <b style="color:rgba(255,255,255,0.8)">KNN Collaborative Filtering</b><br>
        Finds books liked by users with similar reading taste — no content analysis required.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Links</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; line-height:2;">
        <a href="https://github.com/beingkabirahmed/InkLogic" target="_blank"
           style="color:var(--gold-light);text-decoration:none;">
           🔗 GitHub Repository
        </a><br>
        <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset" target="_blank"
           style="color:var(--gold-light);text-decoration:none;">
           📊 Kaggle Dataset
        </a>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ ML-Powered · Collaborative Filtering</div>
    <h1 class="hero-title">Ink<span>Logic</span></h1>
    <p class="hero-subtitle">
        Your personal reading oracle. We analyse patterns from millions of reader
        ratings to surface books you'll actually love — not just bestsellers.
    </p>
    <div class="hero-stats">
        <div class="stat-item">
            <span class="stat-number">1.1M+</span>
            <span class="stat-label">Book Ratings</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">271K</span>
            <span class="stat-label">Books Indexed</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">278K</span>
            <span class="stat-label">Readers Analysed</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_recommendation_engine():
    return Recommendation()

@st.cache_data(show_spinner=False)
def load_book_names():
    path = os.path.join('templates', 'book_names.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    return []


# ─── Main App Body ─────────────────────────────────────────────────────────────
tab_recommend, tab_train, tab_howto = st.tabs(["🔍 Get Recommendations", "⚙️ Train Model", "📖 How It Works"])


# ── Tab 1: Recommendations ────────────────────────────────────────────────────
with tab_recommend:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Pick a book you <span>love</span></h2>', unsafe_allow_html=True)

    book_names = load_book_names()

    if not book_names:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <b>Model not trained yet.</b> Switch to the <em>Train Model</em> tab to build 
            the recommendation engine first, then come back here.
        </div>
        """, unsafe_allow_html=True)
    else:
        selected_book = st.selectbox(
            "Search or scroll to pick a book",
            book_names,
            label_visibility="collapsed"
        )

        st.markdown('<br>', unsafe_allow_html=True)
        col_btn, col_spacer = st.columns([1, 3])
        with col_btn:
            show_recs = st.button("✨ Show Recommendations", key="recommend_btn")

        if show_recs:
            st.markdown('<hr class="ink-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
            st.markdown(f'<h2 class="section-title">Readers of <span>"{selected_book[:40]}{"..." if len(selected_book)>40 else ""}"</span> also loved</h2>',
                        unsafe_allow_html=True)

            try:
                obj = load_recommendation_engine()
                with st.spinner("Finding your next favourites…"):
                    recommended_books, poster_url = obj.recommend_book(selected_book)

                render_recommendations(recommended_books, poster_url)

                st.markdown("""
                <div class="info-box">
                    💡 <b>How these are chosen:</b> The engine found readers with a similar
                    rating history to fans of your chosen book, then surfaced titles they rated
                    highly that you haven't seen yet.
                </div>
                """, unsafe_allow_html=True)
                logging.info(f"Recommendations served for: {selected_book}")

            except Exception as e:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <b>Could not generate recommendations.</b><br>
                    Make sure the model is trained first. Error: <code>{str(e)[:120]}</code>
                </div>
                """, unsafe_allow_html=True)


# ── Tab 2: Train Model ────────────────────────────────────────────────────────
with tab_train:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Training</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Build the <span>engine</span></h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        ℹ️ Run this once before requesting recommendations. The pipeline will:<br><br>
        &nbsp;&nbsp;① Ingest &amp; validate the BX-Books, BX-Users and BX-Ratings CSVs<br>
        &nbsp;&nbsp;② Filter and pivot the rating matrix<br>
        &nbsp;&nbsp;③ Train a K-Nearest Neighbours collaborative filtering model<br>
        &nbsp;&nbsp;④ Serialise model artefacts to <code>artifacts/</code>
    </div>
    """, unsafe_allow_html=True)

    col_train, col_spacer = st.columns([1, 3])
    with col_train:
        run_training = st.button("🚀 Start Training Pipeline", key="train_btn")

    if run_training:
        try:
            obj = Recommendation()
            with st.spinner("Training in progress — this may take a few minutes…"):
                obj.train_engine()
            st.markdown("""
            <div class="success-box">
                ✅ <b>Training complete!</b> Your recommendation engine is ready.
                Switch to the <em>Get Recommendations</em> tab to explore.
            </div>
            """, unsafe_allow_html=True)
            # Clear cached resources so next call picks up fresh model
            load_recommendation_engine.clear()
            load_book_names.clear()

        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
                ❌ <b>Training failed.</b> Please check your data files and configuration.<br>
                Error: <code>{str(e)[:200]}</code>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 3: How It Works ───────────────────────────────────────────────────────
with tab_howto:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">The science of <span>serendipity</span></h2>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📥", "Step 01", "Data Ingestion", "Three CSVs are loaded — Books, Users, and Ratings — then validated for schema and quality."),
        ("🔧", "Step 02", "Transformation", "A user–book rating matrix is built. Books with fewer than 50 ratings are filtered out to reduce noise."),
        ("🤖", "Step 03", "KNN Training", "A K-Nearest Neighbours model (k=5) finds books in the same neighbourhood of the rating space."),
        ("✨", "Step 04", "Inference", "Given your book, we locate it in the matrix and return its 5 nearest neighbours as recommendations."),
    ]
    for col, (icon, step, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="how-card">
                <span class="how-icon">{icon}</span>
                <div class="how-step">{step}</div>
                <div class="how-title">{title}</div>
                <div class="how-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="ink-divider">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title" style="font-size:1.2rem;">About the <span>Dataset</span></h3>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
            <b>Book-Crossing Dataset</b><br><br>
            Collected by Cai-Nicolas Ziegler in a 4-week crawl from the Book-Crossing community.
            It contains explicit ratings (1–10) and implicit ratings (0) from real users,
            covering titles across every genre imaginable.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-box">
            <b>Collaborative Filtering</b><br><br>
            Unlike content-based methods, collaborative filtering never reads a single word
            of a book. It recommends purely from <em>who else liked what you liked</em>
            — which often surfaces surprising, cross-genre gems.
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="ink-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <div class="footer-logo">Ink<span>Logic</span></div>
    A collaborative filtering book recommender · Built with Streamlit &amp; scikit-learn<br>
    Dataset: <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset" target="_blank">Book-Crossing (Kaggle)</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/beingkabirahmed/InkLogic" target="_blank">View on GitHub</a>
    &nbsp;·&nbsp; MIT License
</div>
""", unsafe_allow_html=True)
