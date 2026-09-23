"""
VoltGrid interactive visualiser.

Built for teaching ON STREAM: every control changes ONE thing, and the
screen shows what that thing did. Nothing here is decorative.

    streamlit run apps/visualizer.py

Tabs:
  1  Decision boundary   - watch a model carve up two features   (chapters 10-11)
  2  Threshold           - precision vs recall, made physical     (chapters 3, 14)
  3  Scaling             - why chapter 4 insisted on it           (chapter 4)
  4  Leakage             - the trap from chapter 17, live         (chapter 17)
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from voltgrid import NUMERIC_FEATURES, load_starter, plot_confusion, plot_roc
from voltgrid.plotting import INK, INK_SOFT, SERIES, STYLE, use_style

warnings.filterwarnings("ignore")

st.set_page_config(page_title="VoltGrid | scikit-learn visualiser",
                   page_icon="⚡", layout="wide")
use_style()


@st.cache_data
def get_data():
    return load_starter()


df = get_data()
y_all = df["failed"]

st.title("⚡ VoltGrid — scikit-learn visualiser")
st.caption(
    f"{len(df):,} charging sessions · {y_all.mean():.1%} failed · "
    "synthetic data, no real people"
)

MODELS = {
    "LogisticRegression": lambda p: LogisticRegression(C=p["C"], max_iter=2000),
    "DecisionTree": lambda p: DecisionTreeClassifier(
        max_depth=p["depth"], min_samples_leaf=p["leaf"], random_state=0),
    "RandomForest": lambda p: RandomForestClassifier(
        n_estimators=p["trees"], max_depth=p["depth"], random_state=0, n_jobs=-1),
    "KNeighbors": lambda p: KNeighborsClassifier(n_neighbors=p["k"]),
    "HistGradientBoosting": lambda p: HistGradientBoostingClassifier(
        max_depth=p["depth"], learning_rate=p["lr"], random_state=0),
}

tab1, tab2, tab3, tab4 = st.tabs(
    ["Decision boundary", "Threshold", "Scaling", "Leakage"]
)

# ════════════════════════════════════════════════ 1. decision boundary
with tab1:
    left, right = st.columns([1, 2.1])

    with left:
        st.subheader("Controls")
        f1 = st.selectbox("x-axis feature", NUMERIC_FEATURES,
                          index=NUMERIC_FEATURES.index("grid_load_index"))
        f2 = st.selectbox("y-axis feature", NUMERIC_FEATURES,
                          index=NUMERIC_FEATURES.index("station_age_years"))
        name = st.selectbox("model", list(MODELS))

        p = {}
        if name == "LogisticRegression":
            p["C"] = st.select_slider(
                "C  (low = simpler, high = wigglier)",
                [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        if name in ("DecisionTree", "RandomForest", "HistGradientBoosting"):
            p["depth"] = st.slider("max_depth", 1, 14, 4)
        if name == "DecisionTree":
            p["leaf"] = st.slider("min_samples_leaf", 1, 80, 10)
        if name == "RandomForest":
            p["trees"] = st.slider("n_estimators", 10, 300, 100, step=10)
        if name == "KNeighbors":
            p["k"] = st.slider("n_neighbors", 1, 60, 15)
        if name == "HistGradientBoosting":
            p["lr"] = st.select_slider("learning_rate",
                                       [0.01, 0.05, 0.1, 0.3], value=0.1)

        st.divider()
        st.markdown(
            "**Predict before you move the slider.** Overfitting looks like "
            "islands of colour around single points."
        )

    with right:
        if f1 == f2:
            st.warning("Pick two different features.")
        else:
            X2 = df[[f1, f2]]
            Xtr, Xte, ytr, yte = train_test_split(
                X2, y_all, test_size=0.25, random_state=42, stratify=y_all)

            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            model = MODELS[name](p).fit(Xtr_s, ytr)

            tr_auc = roc_auc_score(ytr, model.predict_proba(Xtr_s)[:, 1])
            te_auc = roc_auc_score(yte, model.predict_proba(Xte_s)[:, 1])

            c1, c2, c3 = st.columns(3)
            c1.metric("train ROC-AUC", f"{tr_auc:.3f}")
            c2.metric("test ROC-AUC", f"{te_auc:.3f}")
            c3.metric("overfit gap", f"{tr_auc - te_auc:+.3f}",
                      delta=f"{tr_auc - te_auc:+.3f}", delta_color="inverse")

            if tr_auc - te_auc > 0.12:
                st.error(
                    f"Train beats test by {tr_auc - te_auc:.3f}. The model is "
                    "memorising these rows, not learning the pattern."
                )

            # ---- boundary drawn on the ORIGINAL units so the axes read sensibly
            fig, ax = plt.subplots(figsize=(7.4, 5.4))
            pad1 = 0.06 * (X2[f1].max() - X2[f1].min())
            pad2 = 0.06 * (X2[f2].max() - X2[f2].min())
            gx, gy = np.meshgrid(
                np.linspace(X2[f1].min() - pad1, X2[f1].max() + pad1, 240),
                np.linspace(X2[f2].min() - pad2, X2[f2].max() + pad2, 240),
            )
            grid = sc.transform(np.c_[gx.ravel(), gy.ravel()])
            zz = model.predict_proba(grid)[:, 1].reshape(gx.shape)

            from voltgrid.plotting import CMAP_DIV, CMAP_SEQ
            crosses = zz.min() < 0.5 < zz.max()
            if crosses:
                # a real 0.5 boundary exists: diverging ramp around it
                im = ax.contourf(gx, gy, zz, levels=np.linspace(0, 1, 25), cmap=CMAP_DIV, alpha=0.85)
                ax.contour(gx, gy, zz, levels=[0.5], colors=[INK],
                           linewidths=1.4, linestyles="--")
            else:
                # no boundary anywhere: a diverging ramp would imply one, so use sequential
                im = ax.contourf(gx, gy, zz, levels=np.linspace(0, max(zz.max(), 0.05), 22),
                                 cmap=CMAP_SEQ, alpha=0.9)

            show = Xte.copy()
            show["failed"] = yte.values
            ok = show[show.failed == 0].sample(min(420, (yte == 0).sum()),
                                               random_state=1)
            bad = show[show.failed == 1]
            ax.scatter(ok[f1], ok[f2], s=20, c=SERIES[0] if crosses else INK_SOFT,
                       alpha=0.55 if crosses else 0.3,
                       edgecolor="#fcfcfb", linewidth=0.8, label="completed")
            ax.scatter(bad[f1], bad[f2], s=42, c=SERIES[1],
                       edgecolor="#fcfcfb", linewidth=1.0, label="failed",
                       zorder=3)
            ax.legend(loc="upper right")
            ax.set_xlabel(f1)
            ax.set_ylabel(f2)
            ax.set_title(f"{name} — predicted probability of failure",
                         loc="left", pad=10)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("P(failure)", color=INK_SOFT)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

            if crosses:
                st.caption("Blue = predicted safe, red = predicted failure, grey = "
                           "the model is unsure. The dashed line is the 0.5 boundary.")
            else:
                st.caption(f"Darker = higher risk. The highest predicted risk anywhere "
                           f"on this plane is {zz.max():.2f}, so at a 0.5 threshold this "
                           "model flags nothing - the chapter 2 and 3 lesson, live.")

# ════════════════════════════════════════════════════════ 2. threshold
with tab2:
    st.subheader("0.5 is a choice, not a law")

    X = df[NUMERIC_FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_all, test_size=0.25, random_state=42, stratify=y_all)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))[:, 1]

    thr = st.slider("decision threshold", 0.01, 0.95, 0.50, 0.01)
    pred = (proba >= thr).astype(int)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("accuracy", f"{accuracy_score(yte, pred):.3f}")
    m2.metric("recall (failures caught)",
              f"{recall_score(yte, pred, zero_division=0):.3f}")
    m3.metric("precision", f"{precision_score(yte, pred, zero_division=0):.3f}")
    m4.metric("flagged", f"{pred.sum()} / {len(pred)}")

    cm = confusion_matrix(yte, pred)
    cL, cR = st.columns(2)
    with cL:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        plot_confusion(cm, labels=("completed", "failed"), ax=ax,
                       title=f"Confusion matrix @ {thr:.2f}")
        st.pyplot(fig, width='stretch')
        plt.close(fig)
    with cR:
        fpr, tpr, _ = roc_curve(yte, proba)
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        plot_roc({"LogisticRegression": (fpr, tpr, roc_auc_score(yte, proba))},
                 ax=ax, title="ROC — every threshold at once")
        ax.scatter([(pred[yte == 0] == 1).mean()],
                   [(pred[yte == 1] == 1).mean()],
                   s=90, c=SERIES[1], zorder=5, edgecolor="#fcfcfb",
                   linewidth=1.4, label=f"you are here ({thr:.2f})")
        ax.legend(loc="lower right")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    missed = int(((pred == 0) & (yte == 1)).sum())
    st.info(
        f"At a threshold of **{thr:.2f}** this model misses **{missed}** of the "
        f"**{int(yte.sum())}** real failures and flags **{int(pred.sum())}** "
        "sessions for inspection. Drag the slider: the ROC curve does not "
        "move, only your position on it does."
    )

# ══════════════════════════════════════════════════════════ 3. scaling
with tab3:
    st.subheader("Chapter 4, live")
    cap = st.slider("max_iter", 20, 800, 100, 20)

    X = df[NUMERIC_FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_all, test_size=0.25, random_state=42, stratify=y_all)

    raw = LogisticRegression(max_iter=cap).fit(Xtr, ytr)
    raw_auc = roc_auc_score(yte, raw.predict_proba(Xte)[:, 1])

    s = StandardScaler().fit(Xtr)
    sca = LogisticRegression(max_iter=cap).fit(s.transform(Xtr), ytr)
    sca_auc = roc_auc_score(yte, sca.predict_proba(s.transform(Xte))[:, 1])

    a, b = st.columns(2)
    a.metric("RAW columns — ROC-AUC", f"{raw_auc:.4f}",
             delta=f"{int(raw.n_iter_[0])} iterations used")
    b.metric("SCALED columns — ROC-AUC", f"{sca_auc:.4f}",
             delta=f"{int(sca.n_iter_[0])} iterations used")

    if raw.n_iter_[0] >= cap:
        st.error(
            f"The raw model hit the {cap}-iteration cap and stopped early. "
            "scikit-learn only warns — it does not raise."
        )
    else:
        st.success(f"Both converged. Raw needed {int(raw.n_iter_[0])} "
                   f"iterations, scaled needed {int(sca.n_iter_[0])}.")

    sd = Xtr.std().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(range(len(sd)), sd.values, color=SERIES[0], width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sd)), sd.index, rotation=38, ha="right")
    ax.set_ylabel("standard deviation (log scale)")
    ax.set_title("The columns are not on the same scale", loc="left", pad=10)
    st.pyplot(fig, width='stretch')
    plt.close(fig)

# ══════════════════════════════════════════════════════════ 4. leakage
with tab4:
    st.subheader("An unexpected perfect score is a leakage alarm")
    st.markdown(
        "`error_code` is blank for completed sessions and filled in for "
        "failed ones. It is written **after** the outcome — so a model that "
        "uses it is reading the answer, not predicting it."
    )

    from voltgrid import load_ml_sessions, load_sessions_raw

    @st.cache_data
    def leak_frame():
        """Join the operations table onto the modelling table.

        This is how leakage actually arrives in real projects: nobody adds a
        column called `the_answer`. Somebody joins in another table.
        """
        ml = load_ml_sessions()
        ops = load_sessions_raw()[["session_id", "error_code"]]
        return ml.merge(ops, on="session_id", how="left")

    full = leak_frame()
    st.caption(
        f"Joined `error_code` from the operations table onto "
        f"{len(full):,} modelling rows."
    )
    use_leak = st.toggle("include `error_code` as a feature", value=False)

    feats = NUMERIC_FEATURES.copy()
    Xl = full[feats].copy()
    if use_leak:
        Xl["error_code_present"] = (
            full["error_code"].fillna("").astype(str).str.strip().str.len() > 0
        ).astype(int)

    a, b, c, d = train_test_split(
        Xl, full["failed"], test_size=0.25, random_state=42,
        stratify=full["failed"])
    sc = StandardScaler().fit(a)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(a), c)
    auc = roc_auc_score(d, m.predict_proba(sc.transform(b))[:, 1])

    st.metric("ROC-AUC", f"{auc:.4f}")
    if auc > 0.99:
        st.error(
            "**1.0000. Investigate before you celebrate.** On a noisy problem "
            "like this one, a perfect score is a leakage alarm: the target has "
            "leaked into the features. This "
            "model cannot run in production, because at prediction time the "
            "error code has not been written yet."
        )
    else:
        st.success(
            f"{auc:.4f} — a believable score from honest features. "
            "Turn the toggle on to see what leakage looks like."
        )

    st.caption(
        "Rule of thumb: ask of every feature, *would I actually have this "
        "value at the moment I need the prediction?* If no, drop it."
    )
