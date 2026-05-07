import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InkLogic · Book Recommender",
    page_icon="🖊️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --forest:   #1e3a2f;
    --moss:     #2d5a3d;
    --fern:     #3d7a52;
    --sage:     #8fbc8f;
    --mint:     #c8e6c9;
    --cream:    #faf8f2;
    --paper:    #f2ede3;
    --tan:      #e8dfc8;
    --text:     #1a2318;
    --muted:    #5a6b5a;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--cream) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--forest) !important;
    border-right: 1px solid var(--moss);
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }
.sb-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #000 !important;
    letter-spacing: -0.01em;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.sb-logo em { color:#1b4535  !important; font-style: italic; }
.sb-tagline {
    font-size: 0.68rem;
    color: #1b4535 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.sb-divider { height: 1px; background: #1b4535; margin: 1.2rem 0; }
.sb-section {
    font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #1b4535 !important; margin-bottom: 0.7rem; font-weight: 600;
}
.sb-stat {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.4rem 0; border-bottom: 1px solid rgba(143,18,143); font-size: 0.78rem;
}
.sb-stat-lbl { color: #1b4535 !important; }
.sb-stat-val { color: #1b4535 !important; font-weight: 600; }
.sb-link {
    display: block; color: rgba(200,230,201) !important; text-decoration: none;
    font-size: 0.8rem; padding: 0.3rem 0; transition: color 0.18s;
}
.sb-link:hover { color: #1b4535 !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(160deg, var(--forest) 0%, #235240 55%, #1b4535 100%);
    padding: 3.5rem 3.5rem 3rem;
    margin: -1rem -1rem 2.5rem -1rem;
    border-radius: 0 0 28px 28px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='52' height='52' viewBox='0 0 52 52' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.02' fill-rule='evenodd'%3E%3Cpath d='M0 0h26v26H0zm26 26h26v26H26z'/%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}
.hero-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(143,188,143,0.12);
    border: 1px solid rgba(143,188,143,0.35);
    color: var(--sage) !important;
    font-size: 0.66rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.28rem 0.8rem; border-radius: 30px; margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 700; color: #fff !important; line-height: 1.05;
    margin: 0 0 0.9rem; letter-spacing: -0.02em;
}
.hero-title em { font-style: italic; color: var(--sage) !important; }
.hero-sub {
    color: rgba(200,230,201,0.6) !important; font-size: 0.96rem; font-weight: 300;
    max-width: 480px; line-height: 1.7; margin: 0 0 2rem;
}
.hero-metrics { display: flex; }
.metric {
    padding: 0 2.5rem 0 0;
    border-right: 1px solid rgba(143,188,143,0.22);
    margin-right: 2.5rem;
}
.metric:last-child { border-right: none; margin-right: 0; }
.metric-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem; font-weight: 700; color: var(--sage) !important;
    line-height: 1; display: block;
}
.metric-lbl {
    font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: rgba(200,230,201,0.4) !important; margin-top: 0.2rem;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important; font-size: 0.84rem !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important; padding-bottom: 0.55rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--forest) !important; border-bottom-color: var(--fern) !important;
}
[data-testid="stTabPanel"] { padding-top: 1.5rem !important; }

/* ── Section Labels ── */
.sec-eyebrow {
    font-size: 0.63rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--fern); font-weight: 600; margin-bottom: 0.4rem;
}
.sec-heading {
    font-family: 'Cormorant Garamond', serif; font-size: 1.85rem; font-weight: 700;
    color: var(--forest); margin: 0 0 1.4rem; line-height: 1.15;
}
.sec-heading em { font-style: italic; color: var(--fern); }

/* ── Selectbox ── */
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stSelectbox"] > div > div {
    background: #fff !important; border: 1.5px solid var(--tan) !important;
    border-radius: 10px !important; font-size: 0.9rem !important;
    padding: 0.5rem 1rem !important; color: var(--text) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--fern) !important;
    box-shadow: 0 0 0 3px rgba(61,122,82,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--forest) !important; color: var(--cream) !important;
    border: none !important; border-radius: 9px !important;
    padding: 0.62rem 2rem !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 500 !important;
    font-size: 0.87rem !important; letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 12px rgba(30,58,47,0.22) !important;
}
.stButton > button:hover {
    background: var(--fern) !important; transform: translateY(-2px) !important;
    box-shadow: 0 7px 18px rgba(30,58,47,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Book Card Grid ── */
.book-row {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.2rem; margin-top: 0.5rem;
}
.bcard {
    background: #fff; border-radius: 12px; border: 1px solid var(--tan);
    overflow: hidden; transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.bcard:hover { transform: translateY(-5px); box-shadow: 0 14px 32px rgba(30,58,47,0.14); }
.bcard-img-wrap {
    width: 100%; padding-top: 148%; position: relative;
    background: var(--paper); overflow: hidden;
}
.bcard-img-wrap img {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover; display: block;
}
.bcard-fallback {
    position: absolute; inset: 0; display: flex;
    align-items: center; justify-content: center; font-size: 3rem;
    background: linear-gradient(145deg, var(--paper), var(--tan));
}
.bcard-body { padding: 0.7rem 0.8rem 0.9rem; }
.bcard-num {
    font-size: 0.58rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--fern); margin-bottom: 0.22rem;
}
.bcard-title {
    font-family: 'Cormorant Garamond', serif; font-size: 0.88rem;
    font-weight: 600; color: var(--forest); line-height: 1.3;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}

/* ── Alert Boxes ── */
.box-info {
    background: rgba(61,122,82,0.06); border: 1px solid rgba(61,122,82,0.2);
    border-left: 3px solid var(--fern); border-radius: 10px;
    padding: 0.9rem 1.1rem; font-size: 0.84rem; color: var(--muted); line-height: 1.65; margin: 1.2rem 0;
}
.box-warn {
    background: rgba(170,55,35,0.05); border: 1px solid rgba(170,55,35,0.18);
    border-left: 3px solid #b03c25; border-radius: 10px;
    padding: 0.9rem 1.1rem; font-size: 0.84rem; color: #7a2b18; line-height: 1.65; margin: 1.2rem 0;
}
.box-success {
    background: rgba(61,122,82,0.08); border: 1px solid rgba(61,122,82,0.25);
    border-left: 3px solid var(--fern); border-radius: 10px;
    padding: 0.9rem 1.1rem; font-size: 0.84rem; color: var(--moss); line-height: 1.65; margin: 1.2rem 0;
}

/* ── Pipeline Cards ── */
.pipe-card {
    background: #fff; border: 1px solid var(--tan); border-radius: 14px;
    padding: 1.4rem 1.2rem; height: 100%; position: relative; overflow: hidden;
}
.pipe-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, var(--fern), var(--sage));
}
.pipe-icon { font-size: 1.7rem; margin-bottom: 0.7rem; display: block; }
.pipe-step { font-size: 0.58rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--fern); margin-bottom: 0.28rem; }
.pipe-title { font-family: 'Cormorant Garamond', serif; font-size: 1rem; font-weight: 700; color: var(--forest); margin-bottom: 0.45rem; }
.pipe-desc { font-size: 0.78rem; color: var(--muted); line-height: 1.6; }

/* ── Divider ── */
.divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, var(--tan), transparent);
    margin: 2.5rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center; padding: 2.5rem 1rem 2rem; margin-top: 3rem;
    border-top: 1px solid var(--tan); font-size: 0.77rem; color: var(--muted); line-height: 2;
}
.footer-brand {
    font-family: 'Cormorant Garamond', serif; font-size: 1.3rem; font-weight: 700;
    color: var(--forest); display: block; margin-bottom: 0.2rem;
}
.footer-brand em { font-style: italic; color: var(--fern); }
.footer a { color: var(--fern); text-decoration: none; }
.stSpinner > div { border-top-color: var(--fern) !important; }
</style>
""", unsafe_allow_html=True)


# ── Core Class ─────────────────────────────────────────────────────────────────
class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_name, ids_index, poster_url = [], [], []
            book_pivot   = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))
            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])
            for name in book_name[0]:
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)
            for idx in ids_index:
                poster_url.append(final_rating.iloc[idx]['image_url'])
            return poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            books_list = []
            model      = pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            book_id    = np.where(book_pivot.index == book_name)[0][0]
            _, suggestion = model.kneighbors(
                book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6
            )
            poster_url = self.fetch_poster(suggestion)
            for i in range(len(suggestion)):
                for j in book_pivot.index[suggestion[i]]:
                    books_list.append(j)
            return books_list, poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            TrainingPipeline().start_training_pipeline()
            logging.info("Training complete.")
        except Exception as e:
            raise AppException(e, sys) from e


# ── Cached Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine():
    return Recommendation()

@st.cache_data(show_spinner=False)
def get_book_names():
    """Returns a plain Python list — safe for len() and bool checks."""
    path = os.path.join('templates', 'book_names.pkl')
    if os.path.exists(path):
        raw = pickle.load(open(path, 'rb'))
        return list(raw)          # converts pandas Index / numpy array → list
    return []


# ── Book Cards ─────────────────────────────────────────────────────────────────
def render_cards(books_list, poster_url):
    html = '<div class="book-row">'
    for i in range(1, 6):
        title = books_list[i] if i < len(books_list) else "Unknown"
        url   = poster_url[i]  if i < len(poster_url) else ""
        label = (title[:44] + "…") if len(title) > 44 else title

        if url:
            img = (
                f'<div class="bcard-img-wrap">'
                f'<img src="{url}" alt="{label}" '
                f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'">'
                f'<div class="bcard-fallback" style="display:none">📗</div>'
                f'</div>'
            )
        else:
            img = '<div class="bcard-img-wrap"><div class="bcard-fallback">📗</div></div>'

        html += f"""
        <div class="bcard">
            {img}
            <div class="bcard-body">
                <div class="bcard-num">Pick #{i}</div>
                <div class="bcard-title">{label}</div>
            </div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">Ink<em>Logic</em></div>
    <div class="sb-tagline">Book Recommender · ML</div>
    <p style="font-size:0.78rem;color:rgba(200,230,201,0.5);line-height:1.65;margin:0 0 0.5rem">
        Collaborative filtering on real reader data — surface books loved by
        people who share your exact taste.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Dataset</div>', unsafe_allow_html=True)
    for lbl, val in [("Books", "271,360"), ("Users", "278,858"), ("Ratings", "1,149,780")]:
        st.markdown(
            f'<div class="sb-stat">'
            f'<span class="sb-stat-lbl">{lbl}</span>'
            f'<span class="sb-stat-val">{val}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Algorithm</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.76rem;color:rgba(200,230,201,0.48);line-height:1.65;margin:0">
        <b style="color:rgba(200,230,201,0.8)">KNN Collaborative Filtering</b><br>
        k = 5 · Cosine similarity · User–book matrix
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Links</div>', unsafe_allow_html=True)
    st.markdown("""
    <a class="sb-link" href="https://github.com/beingkabirahmed/InkLogic" target="_blank">🔗 GitHub Repository</a>
    <a class="sb-link" href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset" target="_blank">📊 Kaggle Dataset</a>
    """, unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-pill">🖊️ Collaborative Filtering · ML-Powered</div>
    <h1 class="hero-title">Ink<em>Logic</em></h1>
    <p class="hero-sub">
        Your personal reading oracle. We surface books you'll actually love
        by studying patterns from a quarter-million real readers — not algorithmic bestsellers.
    </p>
    <div class="hero-metrics">
        <div class="metric">
            <span class="metric-val">1.1M+</span>
            <span class="metric-lbl">Book Ratings</span>
        </div>
        <div class="metric">
            <span class="metric-val">271K</span>
            <span class="metric-lbl">Books Indexed</span>
        </div>
        <div class="metric">
            <span class="metric-val">278K</span>
            <span class="metric-lbl">Readers Analysed</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_rec, tab_train, tab_how = st.tabs(["🔍 Recommendations", "⚙️ Train Model", "📖 How It Works"])


# ── Tab 1 · Recommendations ───────────────────────────────────────────────────
with tab_rec:
    st.markdown('<div class="sec-eyebrow">Step 1 of 2</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sec-heading">Choose a book you <em>love</em></h2>', unsafe_allow_html=True)

    book_names  = get_book_names()
    model_ready = len(book_names) > 0    # ✅ safe: no pandas ambiguity

    if not model_ready:
        st.markdown("""
        <div class="box-warn">
            ⚠️ <b>No trained model found.</b> Head to the <em>Train Model</em> tab,
            run the pipeline, then come back here.
        </div>
        """, unsafe_allow_html=True)
    else:
        selected = st.selectbox("book-selector", book_names, label_visibility="collapsed")

        st.markdown('<br>', unsafe_allow_html=True)
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            go = st.button("✦ Show My Recommendations")

        if go:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            short = (selected[:50] + "…") if len(selected) > 50 else selected
            st.markdown('<div class="sec-eyebrow">Step 2 of 2 · Results</div>', unsafe_allow_html=True)
            st.markdown(
                f'<h2 class="sec-heading">Readers of <em>"{short}"</em> also loved</h2>',
                unsafe_allow_html=True
            )
            try:
                engine = get_engine()
                with st.spinner("Finding your next favourites…"):
                    rec_books, rec_posters = engine.recommend_book(selected)
                render_cards(rec_books, rec_posters)
                st.markdown("""
                <div class="box-info" style="margin-top:1.5rem">
                    💡 <b>Why these books?</b> The engine found other readers who rated your
                    chosen title highly, then surfaced titles <em>they</em> loved —
                    no genre tags, no keywords, pure collective reader intelligence.
                </div>
                """, unsafe_allow_html=True)
                logging.info(f"Recommendations served for: {selected}")
            except Exception as e:
                st.markdown(f"""
                <div class="box-warn">
                    ❌ <b>Recommendation failed.</b> Ensure the model is trained
                    and artifacts exist in the <code>artifacts/</code> folder.<br><br>
                    <code>{str(e)[:220]}</code>
                </div>
                """, unsafe_allow_html=True)


# ── Tab 2 · Train ─────────────────────────────────────────────────────────────
with tab_train:
    st.markdown('<div class="sec-eyebrow">ML Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sec-heading">Build the <em>engine</em></h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="box-info">
        Run this <b>once</b> before requesting recommendations. The pipeline will:<br><br>
        &nbsp;&nbsp;① Ingest &amp; validate <code>BX-Books</code>, <code>BX-Users</code>,
            <code>BX-Book-Ratings</code> CSVs<br>
        &nbsp;&nbsp;② Build a user–book pivot matrix and filter sparse entries<br>
        &nbsp;&nbsp;③ Train a KNN collaborative filtering model<br>
        &nbsp;&nbsp;④ Serialise artefacts to <code>artifacts/</code>
    </div>
    """, unsafe_allow_html=True)

    col_t, _ = st.columns([1, 3])
    with col_t:
        run = st.button("🚀 Start Training Pipeline")

    if run:
        try:
            with st.spinner("Training in progress — this may take a few minutes…"):
                Recommendation().train_engine()
            st.markdown("""
            <div class="box-success">
                ✅ <b>Training complete!</b> Your recommendation engine is ready.
                Switch to the <em>Recommendations</em> tab to explore.
            </div>
            """, unsafe_allow_html=True)
            get_engine.clear()
            get_book_names.clear()
        except Exception as e:
            st.markdown(f"""
            <div class="box-warn">
                ❌ <b>Training failed.</b> Check your data files and config.<br><br>
                <code>{str(e)[:260]}</code>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 3 · How It Works ──────────────────────────────────────────────────────
with tab_how:
    st.markdown('<div class="sec-eyebrow">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sec-heading">The science of <em>serendipity</em></h2>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("📥", "Stage 00",    "Data Ingestion",  "Books, Users and Ratings CSVs are loaded then validated for schema integrity and missing values."),
        ("🔧", "Stage 01–02", "Transformation",  "Books with fewer than 50 ratings are removed. A sparse user–book pivot matrix is constructed from the rest."),
        ("🤖", "Stage 03",    "KNN Training",    "A K-Nearest Neighbours model (k=5, cosine distance) is fitted on the pivot matrix and pickled to disk."),
        ("✨", "Inference",   "Recommendations", "Your book is located in the matrix; the 5 closest neighbours by cosine similarity are returned."),
    ]
    for col, (icon, step, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="pipe-card">
                <span class="pipe-icon">{icon}</span>
                <div class="pipe-step">{step}</div>
                <div class="pipe-title">{title}</div>
                <div class="pipe-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
        <div class="box-info">
            <b>Book-Crossing Dataset</b><br><br>
            Collected from the Book-Crossing community. Contains explicit ratings (1–10)
            and implicit signals (0) spanning every genre from real-world readers.
        </div>
        """, unsafe_allow_html=True)
    with cb:
        st.markdown("""
        <div class="box-info">
            <b>Why Collaborative Filtering?</b><br><br>
            This approach never reads a word of any book. It recommends purely from
            <em>who else liked what you liked</em> — surfacing cross-genre discoveries
            that no keyword or content model could find.
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <span class="footer-brand">Ink<em>Logic</em></span>
    End-to-end ML book recommender · Streamlit &amp; scikit-learn<br>
    Dataset: <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset" target="_blank">Book-Crossing (Kaggle)</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/beingkabirahmed/InkLogic" target="_blank">GitHub</a>
    &nbsp;·&nbsp; MIT License · Kabir Ahmed
</div>
""", unsafe_allow_html=True)