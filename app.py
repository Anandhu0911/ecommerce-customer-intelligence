import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="E-Commerce Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL THEME
# ============================================================
CUSTOM_CSS = """
<style>
:root {
    --bg: #0b0f14;
    --panel: #121821;
    --card: #171f2b;
    --card2: #1c2633;
    --accent: #e8a33d;
    --accent2: #ff6f61;
    --text: #f5f7fa;
    --muted: #9aa5b1;
    --border: #293442;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(232,163,61,.08), transparent 30%),
        radial-gradient(circle at 85% 10%, rgba(255,111,97,.06), transparent 25%),
        var(--bg);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10161f 0%, #0c1118 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

.sidebar-brand {
    padding: 8px 4px 22px 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}

.sidebar-brand .logo {
    font-size: 30px;
}

.sidebar-brand .name {
    font-size: 20px;
    font-weight: 800;
    margin-top: 5px;
}

.sidebar-brand .sub {
    color: var(--muted);
    font-size: 12px;
    margin-top: 3px;
}

.hero {
    padding: 34px 38px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background:
        linear-gradient(135deg, rgba(232,163,61,.12), rgba(255,111,97,.04) 45%, rgba(23,31,43,.95));
    margin-bottom: 26px;
    box-shadow: 0 18px 55px rgba(0,0,0,.22);
}

.hero-kicker {
    color: var(--accent);
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-size: 12px;
}

.hero-title {
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1.02;
    font-weight: 850;
    margin: 10px 0;
    color: var(--text);
}

.hero-copy {
    color: var(--muted);
    max-width: 800px;
    font-size: 16px;
    line-height: 1.65;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(232,163,61,.12);
    color: var(--accent);
    border: 1px solid rgba(232,163,61,.25);
    font-size: 12px;
    font-weight: 700;
    margin-right: 7px;
    margin-top: 10px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: var(--text);
    margin: 28px 0 13px;
}

.metric-card {
    background: linear-gradient(145deg, var(--card), #111821);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 19px 20px;
    min-height: 118px;
    box-shadow: 0 10px 25px rgba(0,0,0,.16);
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.metric-value {
    color: var(--accent);
    font-size: 29px;
    font-weight: 850;
    margin-top: 9px;
}

.metric-note {
    color: #7f8a98;
    font-size: 11px;
    margin-top: 4px;
}

.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    height: 100%;
}

.info-card h3 {
    color: var(--text);
    margin-top: 0;
}

.info-card p, .info-card li {
    color: var(--muted);
    line-height: 1.6;
}

.reco-card {
    background: linear-gradient(145deg, #17202b, #111720);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 17px;
    margin-bottom: 12px;
    transition: .2s ease;
}

.reco-card:hover {
    border-color: rgba(232,163,61,.55);
    transform: translateY(-2px);
}

.reco-id {
    color: var(--accent);
    font-size: 12px;
    font-weight: 800;
}

.reco-title {
    color: var(--text);
    font-size: 16px;
    font-weight: 750;
    margin-top: 5px;
}

.reco-meta {
    color: var(--muted);
    font-size: 12px;
    margin-top: 6px;
}

.prediction-box {
    padding: 26px;
    border-radius: 18px;
    border: 1px solid var(--border);
    background: linear-gradient(145deg, #171f2a, #111720);
    text-align: center;
    margin-top: 20px;
}

.prediction-label {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-result {
    font-size: 32px;
    font-weight: 850;
    color: var(--accent);
    margin-top: 8px;
}

div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 13px;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #f0b45d);
    color: #111 !important;
    border: none;
    border-radius: 9px;
    font-weight: 800;
    padding: .6rem 1.4rem;
}

.stButton > button:hover {
    filter: brightness(1.08);
    box-shadow: 0 0 18px rgba(232,163,61,.22);
}

div[data-baseweb="select"] > div,
input, textarea {
    background-color: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
}

hr {
    border-color: var(--border);
}

.small-muted {
    color: var(--muted);
    font-size: 12px;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# DATA / MODEL LOADING
# ============================================================
BASE = Path(__file__).resolve().parent

def load_csv(name):
    return pd.read_csv(BASE / name)

@st.cache_data
def get_data():
    master = load_csv("master_dataset.csv")
    customer = load_csv("customer_features.csv")
    segment = load_csv("customer_segmentation.csv")
    return master, customer, segment

@st.cache_resource
def get_models():
    tfidf = joblib.load(BASE / "tfidf_vectorizer.pkl")
    products = joblib.load(BASE / "products_df.pkl")
    model = joblib.load(BASE / "customer_behavior_model.pkl")
    return tfidf, products, model

try:
    master, customer, segment = get_data()
    tfidf, products, model = get_models()
    DATA_READY = True
except Exception as e:
    DATA_READY = False
    LOAD_ERROR = str(e)

# ============================================================
# HELPERS
# ============================================================
def money(value):
    return f"₹{value:,.2f}"

def safe_value(df, col, default=0):
    return df[col] if col in df.columns else pd.Series([default] * len(df))

def recommend(product_id, top_n=6):
    if "product_id" not in products.columns:
        return pd.DataFrame()

    selected = products[products["product_id"] == product_id]
    if selected.empty:
        return pd.DataFrame()

    category_col = "product_category_name_english"
    if category_col not in products.columns:
        return pd.DataFrame()

    selected_text = selected[category_col].fillna("").astype(str)
    all_text = products[category_col].fillna("").astype(str)

    selected_vec = tfidf.transform(selected_text)
    all_vec = tfidf.transform(all_text)

    sim_scores = cosine_similarity(selected_vec, all_vec).flatten()
    top_indices = sim_scores.argsort()[::-1][1:top_n + 1]

    result = products.iloc[top_indices].copy()
    result["similarity_score"] = sim_scores[top_indices]
    return result

def chart_style(ax, title):
    ax.set_title(title, color="#f5f7fa", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#9aa5b1")
    ax.grid(axis="y", alpha=.12)
    for spine in ax.spines.values():
        spine.set_color("#293442")
    ax.set_facecolor("#121821")
    ax.figure.patch.set_facecolor("#121821")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo">🛍️</div>
        <div class="name">E-Commerce IQ</div>
        <div class="sub">Customer intelligence platform</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "WORKSPACE",
        [
            "Home",
            "Dashboard",
            "Customer Segmentation",
            "Recommendation",
            "Customer Prediction",
        ],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown(
        '<div class="small-muted">Analytics • Recommendations • ML Prediction</div>',
        unsafe_allow_html=True
    )

# ============================================================
# ERROR STATE
# ============================================================
if not DATA_READY:
    st.error("The dashboard could not load its data/model files.")
    st.code(LOAD_ERROR)
    st.info(
        "Place the CSV and PKL files in the same folder as app.py, then restart Streamlit."
    )
    st.stop()

# ============================================================
# HOME
# ============================================================
if menu == "Home":
    customers = master["customer_unique_id"].nunique()
    orders = master["order_id"].nunique()
    revenue = master["payment_value"].sum()
    products_count = products["product_id"].nunique() if "product_id" in products else len(products)

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">E-Commerce Intelligence</div>
        <div class="hero-title">Turn customer data into smarter decisions.</div>
        <div class="hero-copy">
            An interactive analytics and machine-learning workspace for understanding
            customer behaviour, discovering segments, recommending products, and
            identifying high-value customers.
        </div>
        <span class="badge">Customer Analytics</span>
        <span class="badge">Recommendation Engine</span>
        <span class="badge">ML Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Platform snapshot</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    metrics = [
        ("Customers", f"{customers:,}", "Unique customers"),
        ("Orders", f"{orders:,}", "Recorded orders"),
        ("Revenue", money(revenue), "Total payment value"),
        ("Products", f"{products_count:,}", "Recommendation catalogue"),
    ]

    for col, (label, value, note) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-title">What you can explore</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("📊", "Business Dashboard",
         "Track customers, orders and revenue while exploring monthly performance."),
        ("🎯", "Customer Segmentation",
         "Understand customer groups and compare their distribution."),
        ("✨", "Recommendations",
         "Select a product and discover similar products using TF-IDF similarity."),
    ]

    for col, (icon, title, text) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'<div class="info-card"><h3>{icon} {title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-title">ML prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h3>🤖 Customer Value Prediction</h3>
        <p>
        Use behavioural inputs such as total orders, average order value,
        purchase frequency, review score and delivery days to classify a customer
        as a high-value or low-value customer using the trained model.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DASHBOARD
# ============================================================
elif menu == "Dashboard":
    st.title("📊 Business Dashboard")
    st.caption("A high-level view of marketplace performance and customer activity.")

    customers = master["customer_unique_id"].nunique()
    orders = master["order_id"].nunique()
    revenue = master["payment_value"].sum()
    avg_order = master["payment_value"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{customers:,}")
    c2.metric("Orders", f"{orders:,}")
    c3.metric("Revenue", money(revenue))
    c4.metric("Avg. Order Value", money(avg_order))

    st.markdown('<div class="section-title">Revenue performance</div>', unsafe_allow_html=True)

    if "purchase_month" in master.columns:
        monthly = master.groupby("purchase_month", sort=True)["payment_value"].sum()
        fig, ax = plt.subplots(figsize=(12, 4.5))
        monthly.plot(kind="line", marker="o", linewidth=2.5, ax=ax, color="#e8a33d")
        chart_style(ax, "Monthly Revenue Trend")
        ax.set_xlabel("")
        ax.set_ylabel("Revenue")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("purchase_month is not available in the dataset.")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Order value distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        master["payment_value"].dropna().plot(kind="hist", bins=30, ax=ax, alpha=.85)
        chart_style(ax, "Payment Value Distribution")
        ax.set_xlabel("Payment Value")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown('<div class="section-title">Top product categories</div>', unsafe_allow_html=True)
        cat_col = "product_category_name_english"
        if cat_col in master.columns:
            top_cat = master[cat_col].fillna("Unknown").value_counts().head(10).sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            top_cat.plot(kind="barh", ax=ax)
            chart_style(ax, "Top 10 Categories by Orders")
            ax.set_xlabel("Orders")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Product category data is not available.")

    st.markdown('<div class="section-title">Dataset preview</div>', unsafe_allow_html=True)
    st.dataframe(master.head(12), use_container_width=True, hide_index=True)

# ============================================================
# SEGMENTATION
# ============================================================
elif menu == "Customer Segmentation":
    st.title("🎯 Customer Segmentation")
    st.caption("Explore how customers are distributed across behavioural segments.")

    if "Customer_Type" in segment.columns:
        counts = segment["Customer_Type"].fillna("Unknown").value_counts()
        total = counts.sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Segments", f"{counts.shape[0]:,}")
        c2.metric("Customers in file", f"{len(segment):,}")
        c3.metric("Largest segment", str(counts.index[0]))

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown('<div class="section-title">Segment distribution</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4.8))
            counts.sort_values().plot(kind="barh", ax=ax)
            chart_style(ax, "Customers by Segment")
            ax.set_xlabel("Customers")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown('<div class="section-title">Segment share</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4.8))
            counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, startangle=90)
            ax.set_ylabel("")
            ax.set_title("Customer Mix", color="#f5f7fa", fontweight="bold")
            fig.patch.set_facecolor("#121821")
            ax.set_facecolor("#121821")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown('<div class="section-title">Segment records</div>', unsafe_allow_html=True)
        st.dataframe(segment, use_container_width=True, hide_index=True)
    else:
        st.warning("Customer_Type column was not found in customer_segmentation.csv.")
        st.dataframe(segment.head(20), use_container_width=True)

# ============================================================
# RECOMMENDATION
# ============================================================
elif menu == "Recommendation":
    st.title("✨ Product Recommendation Engine")
    st.caption("Find products with similar category representations using your trained TF-IDF vectorizer.")

    if "product_id" not in products.columns:
        st.error("product_id column is missing from products_df.pkl.")
        st.stop()

    product_ids = products["product_id"].dropna().astype(str).unique().tolist()

    selected = st.selectbox(
        "Choose a product",
        product_ids,
        index=0,
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f'<div class="info-card"><h3>Selected product</h3>'
            f'<p class="small-muted">Product ID</p><h2>{selected}</h2></div>',
            unsafe_allow_html=True
        )
    with c2:
        top_n = st.slider("Recommendations", 3, 10, 6)

    if st.button("Generate Recommendations", use_container_width=True):
        result = recommend(selected, top_n)

        if result.empty:
            st.warning("No recommendations could be generated for this product.")
        else:
            st.markdown('<div class="section-title">Recommended products</div>', unsafe_allow_html=True)

            cols = st.columns(2)
            for i, (_, row) in enumerate(result.iterrows()):
                product_id = row.get("product_id", "N/A")
                category = row.get("product_category_name_english", "Category unavailable")
                score = row.get("similarity_score", 0)

                with cols[i % 2]:
                    st.markdown(
                        f'<div class="reco-card">'
                        f'<div class="reco-id">RECOMMENDATION #{i+1}</div>'
                        f'<div class="reco-title">{category}</div>'
                        f'<div class="reco-meta">Product ID: {product_id}</div>'
                        f'<div class="reco-meta">Similarity score: {score:.3f}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with st.expander("View recommendation data"):
                st.dataframe(result, use_container_width=True, hide_index=True)

# ============================================================
# PREDICTION
# ============================================================
elif menu == "Customer Prediction":
    st.title("🤖 Customer Value Prediction")
    st.caption("Estimate whether a customer belongs to the high-value or low-value class.")

    st.markdown('<div class="section-title">Customer behaviour inputs</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        total_orders = st.number_input(
            "Total Orders",
            min_value=1,
            max_value=1000,
            value=5,
            step=1,
        )
        avg_order = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=500.0,
            step=50.0,
        )
        purchase = st.number_input(
            "Purchase Frequency",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )

    with c2:
        review = st.slider(
            "Average Review Score",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
        )
        delivery = st.number_input(
            "Average Delivery Days",
            min_value=0.0,
            value=3.0,
            step=1.0,
        )

    st.markdown("---")

    if st.button("Predict Customer Value", use_container_width=True):
        sample = [[
            total_orders,
            avg_order,
            purchase,
            review,
            delivery,
        ]]

        try:
            pred = model.predict(sample)

            if pred[0] == 1:
                st.markdown("""
                <div class="prediction-box">
                    <div class="prediction-label">Model prediction</div>
                    <div class="prediction-result">⭐ High Value Customer</div>
                    <p class="small-muted">
                    The trained model classified this customer as high value.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box">
                    <div class="prediction-label">Model prediction</div>
                    <div class="prediction-result">Low Value Customer</div>
                    <p class="small-muted">
                    The trained model classified this customer as low value.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("View submitted inputs"):
                st.dataframe(
                    pd.DataFrame([{
                        "Total Orders": total_orders,
                        "Average Order Value": avg_order,
                        "Purchase Frequency": purchase,
                        "Average Review": review,
                        "Delivery Days": delivery,
                    }]),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
