import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    silhouette_score
)
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS — Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0a0e1a; color: #e2e8f0; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f1526;
    border-right: 1px solid #1e2d50;
  }
  section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label { color: #64748b !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #0f1e3d 0%, #112244 50%, #0a1628 100%);
    border: 1px solid #1e3a6e;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      90deg, transparent, transparent 40px,
      rgba(30,60,120,0.08) 40px, rgba(30,60,120,0.08) 41px
    );
    pointer-events: none;
  }
  .hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #38bdf8; margin: 0 0 0.4rem;
    text-shadow: 0 0 30px rgba(56,189,248,0.4);
  }
  .hero p { color: #94a3b8; margin: 0; font-size: 0.95rem; }
  .hero .badge {
    display: inline-block; background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3); border-radius: 20px;
    padding: 3px 12px; font-size: 0.72rem; color: #38bdf8;
    font-family: 'Space Mono', monospace; margin-right: 6px; margin-top: 10px;
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 130px;
    background: #0f1a30; border: 1px solid #1e3a6e;
    border-radius: 10px; padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700; color: #38bdf8;
  }
  .metric-card .lbl {
    font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em;
    text-transform: uppercase; margin-top: 4px;
  }
  .metric-card.danger .val { color: #f87171; }
  .metric-card.warning .val { color: #fb923c; }
  .metric-card.success .val { color: #4ade80; }

  /* Section header */
  .sec-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: #38bdf8;
    border-left: 3px solid #38bdf8; padding-left: 10px;
    margin: 1.5rem 0 0.8rem;
  }

  /* Info box */
  .info-box {
    background: #0f1a30; border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem;
    font-size: 0.88rem; color: #94a3b8; margin: 0.5rem 0 1rem;
  }

  /* Alert boxes */
  .alert-danger {
    background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3);
    border-radius: 8px; padding: 0.8rem 1rem; color: #fca5a5; font-size: 0.88rem; margin: 0.5rem 0;
  }
  .alert-success {
    background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.3);
    border-radius: 8px; padding: 0.8rem 1rem; color: #86efac; font-size: 0.88rem; margin: 0.5rem 0;
  }

  /* Cluster risk badge */
  .risk-high   { color: #f87171; font-weight: 700; }
  .risk-medium { color: #fb923c; font-weight: 700; }
  .risk-low    { color: #4ade80; font-weight: 700; }

  /* DataFrame styling */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0f1526; gap: 2px; }
  .stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; padding: 8px 18px; border-radius: 6px 6px 0 0; }
  .stTabs [aria-selected="true"] { background: #0f1a30 !important; color: #38bdf8 !important; }

  /* Button */
  .stButton > button {
    background: linear-gradient(135deg, #1e3a6e, #1a4a8a);
    color: #38bdf8; border: 1px solid #2563eb;
    border-radius: 8px; font-family: 'Space Mono', monospace;
    font-size: 0.82rem; padding: 0.55rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: #fff; border-color: #38bdf8;
    box-shadow: 0 0 20px rgba(56,189,248,0.3);
  }

  /* Spinner */
  .stSpinner > div { border-top-color: #38bdf8 !important; }

  /* Hide default streamlit chrome */
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper – matplotlib dark theme
# ─────────────────────────────────────────────
def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0f1a30")
    ax.set_facecolor("#0a1220")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a6e")
    ax.tick_params(colors="#64748b")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#38bdf8")
    return fig, ax

PALETTE = ["#38bdf8", "#818cf8", "#f472b6", "#4ade80", "#fb923c"]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ NIDS Control Panel")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"],
        help="Upload the CICIDS network traffic CSV file.",
    )

    st.markdown("#### ⚙️ Model Parameters")
    n_clusters   = st.slider("KMeans Clusters",        2, 10, 5)
    n_components = st.slider("PCA Components",         2, 20, 10)
    contamination= st.slider("IF Contamination",       0.01, 0.20, 0.05, 0.01,
                              help="Expected anomaly fraction for Isolation Forest.")
    svm_nu       = st.slider("One-Class SVM ν",        0.01, 0.50, 0.05, 0.01)
    sample_size  = st.selectbox("Sample Size (rows)",
                                [5000, 10000, 25000, 50000, "All"],
                                index=2)

    run_btn = st.button("▶  Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#334155;'>NIDS · Unsupervised ML<br>PCA · KMeans · IForest · OCSVM</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛡️ Network Intrusion Detection System</h1>
  <p>Unsupervised anomaly detection using PCA, MiniBatch K-Means, Isolation Forest &amp; One-Class SVM</p>
  <span class="badge">Unsupervised ML</span>
  <span class="badge">PCA</span>
  <span class="badge">Clustering</span>
  <span class="badge">Anomaly Detection</span>
  <span class="badge">Cybersecurity</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────
# If no file uploaded – show instructions
# ─────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="info-box">
      📂 <b>Upload your CSV dataset</b> using the sidebar panel, configure parameters, then click <b>Run Analysis</b>.<br><br>
      Compatible with CICIDS datasets e.g. <code>Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv</code>.
      The dataset must contain a <code>Label</code> column for evaluation.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="val">PCA</div>
          <div class="lbl">Dimensionality Reduction</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="val" style="color:#818cf8">KMeans</div>
          <div class="lbl">Traffic Clustering</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="val" style="color:#f472b6">IF + SVM</div>
          <div class="lbl">Anomaly Detection</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes, n_clusters, n_components, contamination, svm_nu, sample_size):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))

    # ── Cleaning ──
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # ── Sample ──
    if sample_size != "All" and len(df) > int(sample_size):
        df = df.sample(int(sample_size), random_state=42).reset_index(drop=True)

    # ── Label ──
    label_col = "Label"
    if label_col not in df.columns:
        raise ValueError("Dataset must contain a 'Label' column.")

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df[label_col])
    df["is_attack"] = (df[label_col].str.strip().str.upper() != "BENIGN").astype(int)

    # ── Features ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = ["label_enc", "is_attack"]
    feature_cols = [c for c in numeric_cols if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df["is_attack"].values

    # ── Scale ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── PCA ──
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # ── KMeans ──
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    kmeans.fit(X_pca)
    cluster_labels = kmeans.labels_

    # silhouette on a subset
    sil_sample = min(10000, len(X_pca))
    sil_idx = np.random.choice(len(X_pca), sil_sample, replace=False)
    sil_score = silhouette_score(X_pca[sil_idx], cluster_labels[sil_idx])

    # ── KMeans metrics ──
    km_pred = (cluster_labels == -1).astype(int)   # placeholder; clusters don't yield binary directly
    # map cluster with highest attack ratio as "anomaly"
    attack_ratios = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        attack_ratios[c] = y[mask].mean() if mask.sum() > 0 else 0.0
    best_cluster = max(attack_ratios, key=attack_ratios.get)

    km_pred_binary = (cluster_labels == best_cluster).astype(int)
    km_metrics = {
        "accuracy":  accuracy_score(y, km_pred_binary),
        "precision": precision_score(y, km_pred_binary, zero_division=0),
        "recall":    recall_score(y, km_pred_binary, zero_division=0),
        "f1":        f1_score(y, km_pred_binary, zero_division=0),
    }

    # ── Isolation Forest ──
    iforest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    if_raw = iforest.fit_predict(X_pca)
    if_pred = (if_raw == -1).astype(int)
    if_metrics = {
        "accuracy":  accuracy_score(y, if_pred),
        "precision": precision_score(y, if_pred, zero_division=0),
        "recall":    recall_score(y, if_pred, zero_division=0),
        "f1":        f1_score(y, if_pred, zero_division=0),
    }

    # ── One-Class SVM ──
    ocsvm = OneClassSVM(nu=svm_nu, kernel="rbf", gamma="scale")
    svm_raw = ocsvm.fit_predict(X_pca)
    svm_pred = (svm_raw == -1).astype(int)
    svm_metrics = {
        "accuracy":  accuracy_score(y, svm_pred),
        "precision": precision_score(y, svm_pred, zero_division=0),
        "recall":    recall_score(y, svm_pred, zero_division=0),
        "f1":        f1_score(y, svm_pred, zero_division=0),
    }

    # ── Hybrid ──
    hybrid_score = if_pred + svm_pred + km_pred_binary
    hybrid_pred  = (hybrid_score >= 2).astype(int)
    hybrid_metrics = {
        "accuracy":  accuracy_score(y, hybrid_pred),
        "precision": precision_score(y, hybrid_pred, zero_division=0),
        "recall":    recall_score(y, hybrid_pred, zero_division=0),
        "f1":        f1_score(y, hybrid_pred, zero_division=0),
    }

    return dict(
        df=df, X_pca=X_pca, y=y,
        cluster_labels=cluster_labels, attack_ratios=attack_ratios,
        best_cluster=best_cluster, sil_score=sil_score,
        explained_variance=pca.explained_variance_ratio_,
        km_metrics=km_metrics, if_metrics=if_metrics,
        svm_metrics=svm_metrics, hybrid_metrics=hybrid_metrics,
        feature_cols=feature_cols,
        if_pred=if_pred, svm_pred=svm_pred, km_pred_binary=km_pred_binary,
        hybrid_pred=hybrid_pred,
        n_clusters=n_clusters,
    )

# ─────────────────────────────────────────────
# Run on button press
# ─────────────────────────────────────────────
if run_btn or st.session_state.results is not None:
    if run_btn:
        with st.spinner("Running pipeline… this may take a minute for large datasets."):
            try:
                res = run_pipeline(
                    uploaded.getvalue(),
                    n_clusters, n_components, contamination, svm_nu, sample_size
                )
                st.session_state.results = res
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

    res = st.session_state.results
    if res is None:
        st.stop()

    df            = res["df"]
    X_pca         = res["X_pca"]
    y             = res["y"]
    cl            = res["cluster_labels"]
    attack_ratios = res["attack_ratios"]
    best_cluster  = res["best_cluster"]
    sil           = res["sil_score"]
    ev            = res["explained_variance"]
    n_clust       = res["n_clusters"]

    # ─── Top metrics strip ───
    n_normal = int((y == 0).sum())
    n_attack = int((y == 1).sum())
    pct_attack = 100 * n_attack / len(y)
    max_ar = max(attack_ratios.values()) * 100

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="val">{len(df):,}</div><div class="lbl">Total Records</div></div>
      <div class="metric-card success"><div class="val">{n_normal:,}</div><div class="lbl">Normal Traffic</div></div>
      <div class="metric-card danger"><div class="val">{n_attack:,}</div><div class="lbl">Attack Traffic</div></div>
      <div class="metric-card warning"><div class="val">{pct_attack:.1f}%</div><div class="lbl">Attack Ratio</div></div>
      <div class="metric-card"><div class="val">{sil:.3f}</div><div class="lbl">Silhouette Score</div></div>
      <div class="metric-card danger"><div class="val">{max_ar:.0f}%</div><div class="lbl">Max Cluster Attack%</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # Tabs
    # ─────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 EDA",
        "🔬 PCA & Clustering",
        "🚨 Anomaly Detection",
        "📈 Model Comparison",
        "🔍 Cluster Analysis",
    ])

    # ═══════════════════════════════════════
    # TAB 1 – EDA
    # ═══════════════════════════════════════
    with tab1:
        st.markdown('<div class="sec-header">Attack Distribution</div>', unsafe_allow_html=True)
        label_counts = df["Label"].str.strip().value_counts()

        fig, ax = dark_fig(10, 4)
        colors = [PALETTE[1] if v.upper() == "BENIGN" else PALETTE[3]
                  for v in label_counts.index]
        bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor="#0a1220", linewidth=0.5)
        ax.set_title("Traffic Label Distribution", pad=12)
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right", fontsize=8)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 50,
                    f'{int(b.get_height()):,}', ha='center', va='bottom',
                    fontsize=7, color='#94a3b8')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="sec-header">Correlation Heatmap (Top 20 Features)</div>', unsafe_allow_html=True)
        top20 = df[res["feature_cols"]].iloc[:, :20]
        corr = top20.corr()

        fig2, ax2 = plt.subplots(figsize=(12, 7))
        fig2.patch.set_facecolor("#0f1a30")
        ax2.set_facecolor("#0a1220")
        sns.heatmap(corr, ax=ax2, cmap="coolwarm", center=0,
                    linewidths=0.3, linecolor="#0a1220",
                    annot=False, square=False,
                    cbar_kws={"shrink": 0.8})
        ax2.set_title("Feature Correlation Matrix (Top 20)", color="#38bdf8", pad=12)
        plt.xticks(rotation=45, ha="right", fontsize=7, color="#64748b")
        plt.yticks(fontsize=7, color="#64748b")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown('<div class="sec-header">Feature Distributions (Sample)</div>', unsafe_allow_html=True)
        sample_feats = res["feature_cols"][:9]
        fig3, axes = plt.subplots(3, 3, figsize=(14, 8))
        fig3.patch.set_facecolor("#0f1a30")
        for i, feat in enumerate(sample_feats):
            ax = axes[i // 3][i % 3]
            ax.set_facecolor("#0a1220")
            data = df[feat].dropna()
            ax.hist(data, bins=50, color=PALETTE[i % len(PALETTE)], alpha=0.75, edgecolor="none")
            ax.set_title(feat[:28], fontsize=8, color="#38bdf8")
            ax.tick_params(colors="#64748b", labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e3a6e")
        fig3.tight_layout(pad=1.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        st.markdown('<div class="sec-header">Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True, height=260)

    # ═══════════════════════════════════════
    # TAB 2 – PCA & Clustering
    # ═══════════════════════════════════════
    with tab2:
        st.markdown('<div class="sec-header">PCA Explained Variance</div>', unsafe_allow_html=True)
        cumvar = np.cumsum(ev)
        fig, ax = dark_fig(10, 4)
        ax.bar(range(1, len(ev)+1), ev*100, color=PALETTE[0], alpha=0.7, label="Per Component")
        ax2b = ax.twinx()
        ax2b.plot(range(1, len(cumvar)+1), cumvar*100, color=PALETTE[2],
                  marker='o', markersize=4, linewidth=2, label="Cumulative")
        ax2b.set_ylabel("Cumulative %", color=PALETTE[2])
        ax2b.tick_params(colors="#64748b")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance %")
        ax.set_title("PCA Explained Variance per Component")
        ax.legend(loc="upper right", facecolor="#0f1a30", edgecolor="#1e3a6e", labelcolor="#94a3b8")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-header">2D PCA Cluster Scatter</div>', unsafe_allow_html=True)
            fig, ax = dark_fig(7, 6)
            for k in range(n_clust):
                mask = cl == k
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           s=4, alpha=0.4, color=PALETTE[k % len(PALETTE)],
                           label=f"Cluster {k}")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title("2D PCA Clusters")
            ax.legend(markerscale=3, fontsize=8, facecolor="#0f1a30",
                      edgecolor="#1e3a6e", labelcolor="#94a3b8")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            st.markdown('<div class="sec-header">3D PCA Cluster Scatter</div>', unsafe_allow_html=True)
            fig3d = plt.figure(figsize=(7, 6))
            fig3d.patch.set_facecolor("#0f1a30")
            ax3d = fig3d.add_subplot(111, projection='3d')
            ax3d.set_facecolor("#0a1220")
            for k in range(n_clust):
                mask = cl == k
                ax3d.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                             s=2, alpha=0.3, color=PALETTE[k % len(PALETTE)],
                             label=f"Cluster {k}")
            ax3d.set_title("3D PCA Network Traffic Clusters", color="#38bdf8", pad=6)
            ax3d.tick_params(colors="#64748b", labelsize=6)
            ax3d.xaxis.pane.fill = False
            ax3d.yaxis.pane.fill = False
            ax3d.zaxis.pane.fill = False
            st.pyplot(fig3d, use_container_width=True)
            plt.close()

        st.markdown('<div class="sec-header">Attack Concentration per Cluster</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(10, 4)
        cluster_ids = list(attack_ratios.keys())
        ratios = [attack_ratios[c]*100 for c in cluster_ids]
        bar_colors = [PALETTE[3] if r < 20 else PALETTE[4] if r < 60 else PALETTE[3] for r in ratios]
        bar_colors[best_cluster] = "#f87171"
        bars = ax.bar([f"Cluster {c}" for c in cluster_ids], ratios,
                      color=bar_colors, edgecolor="#0a1220")
        ax.set_title("Attack % per Cluster")
        ax.set_ylabel("Attack Percentage (%)")
        ax.axhline(50, linestyle="--", color="#64748b", linewidth=1, alpha=0.5)
        for b, r in zip(bars, ratios):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f'{r:.1f}%', ha='center', va='bottom', fontsize=8, color='#94a3b8')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ═══════════════════════════════════════
    # TAB 3 – Anomaly Detection
    # ═══════════════════════════════════════
    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="sec-header">Isolation Forest Anomalies</div>', unsafe_allow_html=True)
            fig, ax = dark_fig(7, 5)
            ax.scatter(X_pca[res["if_pred"]==0, 0], X_pca[res["if_pred"]==0, 1],
                       s=3, alpha=0.3, color=PALETTE[0], label="Normal")
            ax.scatter(X_pca[res["if_pred"]==1, 0], X_pca[res["if_pred"]==1, 1],
                       s=5, alpha=0.6, color=PALETTE[3], label="Anomaly")
            ax.set_title("Isolation Forest — PCA Space")
            ax.legend(markerscale=3, facecolor="#0f1a30", edgecolor="#1e3a6e", labelcolor="#94a3b8")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown('<div class="sec-header">One-Class SVM Anomalies</div>', unsafe_allow_html=True)
            fig, ax = dark_fig(7, 5)
            ax.scatter(X_pca[res["svm_pred"]==0, 0], X_pca[res["svm_pred"]==0, 1],
                       s=3, alpha=0.3, color=PALETTE[1], label="Normal")
            ax.scatter(X_pca[res["svm_pred"]==1, 0], X_pca[res["svm_pred"]==1, 1],
                       s=5, alpha=0.6, color=PALETTE[2], label="Anomaly")
            ax.set_title("One-Class SVM — PCA Space")
            ax.legend(markerscale=3, facecolor="#0f1a30", edgecolor="#1e3a6e", labelcolor="#94a3b8")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown('<div class="sec-header">Hybrid Ensemble Anomalies</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(10, 5)
        ax.scatter(X_pca[res["hybrid_pred"]==0, 0], X_pca[res["hybrid_pred"]==0, 1],
                   s=3, alpha=0.25, color=PALETTE[0], label="Normal")
        ax.scatter(X_pca[res["hybrid_pred"]==1, 0], X_pca[res["hybrid_pred"]==1, 1],
                   s=6, alpha=0.7, color=PALETTE[3], label="Hybrid Anomaly")
        ax.set_title("Hybrid Model (IF + OCSVM + KMeans) — PCA Space")
        ax.legend(markerscale=3, facecolor="#0f1a30", edgecolor="#1e3a6e", labelcolor="#94a3b8")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="sec-header">Ground Truth (Actual Labels)</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(10, 5)
        ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1],
                   s=3, alpha=0.2, color=PALETTE[0], label="Actual Normal")
        ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1],
                   s=5, alpha=0.6, color=PALETTE[3], label="Actual Attack")
        ax.set_title("Ground Truth Labels — PCA Space")
        ax.legend(markerscale=3, facecolor="#0f1a30", edgecolor="#1e3a6e", labelcolor="#94a3b8")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ═══════════════════════════════════════
    # TAB 4 – Model Comparison
    # ═══════════════════════════════════════
    with tab4:
        st.markdown('<div class="sec-header">Performance Metrics</div>', unsafe_allow_html=True)

        models_metrics = {
            "MiniBatch K-Means": res["km_metrics"],
            "Isolation Forest":  res["if_metrics"],
            "One-Class SVM":     res["svm_metrics"],
            "Hybrid Ensemble":   res["hybrid_metrics"],
        }

        metrics_df = pd.DataFrame(models_metrics).T.reset_index()
        metrics_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.4f}")
        st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

        metric_names = ["accuracy", "precision", "recall", "f1"]
        model_names  = list(models_metrics.keys())

        fig, axes = plt.subplots(1, 4, figsize=(14, 5))
        fig.patch.set_facecolor("#0f1a30")
        for i, m in enumerate(metric_names):
            ax = axes[i]
            ax.set_facecolor("#0a1220")
            vals   = [models_metrics[mn][m] for mn in model_names]
            colors = [PALETTE[j % len(PALETTE)] for j in range(len(model_names))]
            bars = ax.bar(model_names, vals, color=colors, edgecolor="#0a1220")
            ax.set_title(m.capitalize(), color="#38bdf8", fontsize=10)
            ax.set_ylim(0, max(max(vals) * 1.25, 0.05))
            ax.tick_params(colors="#64748b", labelsize=7)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
            for sp in ax.spines.values(): sp.set_edgecolor("#1e3a6e")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7, color='#94a3b8')
        fig.suptitle("Model Performance Comparison", color="#38bdf8", fontsize=13, y=1.02)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class="info-box">
          <b>Why is accuracy misleading?</b><br>
          The dataset is highly imbalanced — normal traffic dominates. A model predicting "normal" for
          everything can still score ~53% accuracy. <b>Recall</b> and <b>F1-Score</b> are more meaningful
          for intrusion detection, where missing attacks is costly.
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # TAB 5 – Cluster Analysis
    # ═══════════════════════════════════════
    with tab5:
        st.markdown('<div class="sec-header">Cluster Risk Summary</div>', unsafe_allow_html=True)

        cluster_data = []
        for c in sorted(attack_ratios.keys()):
            ratio = attack_ratios[c]
            count = int((cl == c).sum())
            attacks = int(y[cl == c].sum())
            if ratio > 0.5:
                risk = "🔴 HIGH"
            elif ratio > 0.1:
                risk = "🟠 MEDIUM"
            else:
                risk = "🟢 LOW"
            cluster_data.append({
                "Cluster": f"Cluster {c}",
                "Total Traffic": count,
                "Attack Count": attacks,
                "Normal Count": count - attacks,
                "Attack Ratio": f"{ratio*100:.2f}%",
                "Risk Level": risk,
            })

        cluster_df = pd.DataFrame(cluster_data)
        st.dataframe(cluster_df.set_index("Cluster"), use_container_width=True)

        st.markdown(f"""
        <div class="alert-danger">
          🚨 <b>Cluster {best_cluster}</b> has the highest attack concentration at
          <b>{attack_ratios[best_cluster]*100:.1f}%</b>.
          SOC teams should prioritize investigation of traffic in this cluster — it likely represents
          port scanning, DDoS, or other malicious behaviour.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header">Traffic Composition per Cluster</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(10, 5)
        cluster_ids = sorted(attack_ratios.keys())
        normal_counts = [int((cl == c).sum()) - int(y[cl == c].sum()) for c in cluster_ids]
        attack_counts = [int(y[cl == c].sum()) for c in cluster_ids]
        x_pos = np.arange(len(cluster_ids))
        bars1 = ax.bar(x_pos, normal_counts, color=PALETTE[0], label="Normal", alpha=0.85)
        bars2 = ax.bar(x_pos, attack_counts, bottom=normal_counts, color=PALETTE[3],
                       label="Attack", alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Cluster {c}" for c in cluster_ids])
        ax.set_title("Stacked Traffic Composition per Cluster")
        ax.set_ylabel("Record Count")
        ax.legend(facecolor="#0f1a30", edgecolor="#1e3a6e", labelcolor="#94a3b8")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="sec-header">Silhouette Score Interpretation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
          <b>Silhouette Score: {sil:.4f}</b><br>
          {"✅ Good cluster separation (> 0.5) — PCA + MiniBatch K-Means produced meaningful traffic groups." if sil > 0.5
          else "⚠️ Moderate cluster separation — consider increasing PCA components or adjusting cluster count."}
          <br><br>
          Score ranges: &nbsp; 0.7–1.0 = Strong &nbsp;|&nbsp; 0.5–0.7 = Moderate &nbsp;|&nbsp; 0.25–0.5 = Weak &nbsp;|&nbsp; &lt; 0.25 = Poor
        </div>
        """, unsafe_allow_html=True)
