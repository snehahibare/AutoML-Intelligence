import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                              mean_squared_error, mean_absolute_error,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, auc)

COLORS = {'best': '#2563EB', 'others': '#94A3B8', 'accent': '#F59E0B', 'text': '#1E293B'}
plt.rcParams.update({'figure.facecolor': '#F8FAFC', 'axes.facecolor': '#F8FAFC',
                     'axes.spines.top': False, 'axes.spines.right': False})

# ── Helper Functions ──────────────────────────────────────────────

def detect_problem(y):
    unique, total = y.nunique(), len(y)
    if not pd.api.types.is_numeric_dtype(y) or unique < 20 or (unique/total) < 0.05:
        return 'classification', f'{unique} unique values → low cardinality'
    return 'regression', f'{unique} unique values → high cardinality'

def build_preprocessor(X):
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc',  StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('enc', OneHotEncoder(handle_unknown='ignore',
                                                sparse_output=False))]), cat_cols)
    ], remainder='drop')
    return preprocessor, num_cols, cat_cols

def get_candidates(problem, n):
    if problem == 'classification':
        cands = [
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42),
             {'clf__C': [0.01, 0.1, 1, 10]}),
            ('Random Forest', RandomForestClassifier(random_state=42),
             {'clf__n_estimators': [50,100], 'clf__max_depth': [None,5,10]}),
            ('Gradient Boosting', GradientBoostingClassifier(random_state=42),
             {'clf__n_estimators': [50,100], 'clf__learning_rate': [0.05,0.1,0.2]}),
        ]
        if n < 10_000:
            cands.append(('SVM', SVC(probability=True, random_state=42),
                          {'clf__C': [0.1,1,10], 'clf__kernel': ['rbf','linear']}))
    else:
        cands = [
            ('Linear Regression', LinearRegression(), {}),
            ('Ridge', Ridge(), {'clf__alpha': [0.1,1,10,100]}),
            ('Random Forest', RandomForestRegressor(random_state=42),
             {'clf__n_estimators': [50,100], 'clf__max_depth': [None,5,10]}),
            ('Gradient Boosting', GradientBoostingRegressor(random_state=42),
             {'clf__n_estimators': [50,100], 'clf__learning_rate': [0.05,0.1]}),
        ]
    return cands

def train_models(candidates, preprocessor, X_train, X_test, y_train, y_test, problem):
    scoring = 'accuracy' if problem == 'classification' else 'r2'
    results = []
    for name, estimator, params in candidates:
        pipe = Pipeline([('pre', preprocessor), ('clf', estimator)])
        if params:
            search = RandomizedSearchCV(pipe, params, n_iter=6, cv=3,
                                        scoring=scoring, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
        else:
            pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        if problem == 'classification':
            score = accuracy_score(y_test, y_pred)
            extra = {'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)}
        else:
            score = r2_score(y_test, y_pred)
            extra = {'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                     'MAE' : mean_absolute_error(y_test, y_pred)}
        results.append({'name': name, 'pipe': pipe, 'score': score, 'extra': extra})
    return sorted(results, key=lambda r: r['score'], reverse=True)

# ── Streamlit UI ──────────────────────────────────────────────────

st.set_page_config(page_title='AutoML System', page_icon='🤖', layout='wide')

st.title('🤖 AutoML System')
st.caption('Upload any CSV → auto-detects problem → trains models → picks best → explains why')
st.divider()

# Sidebar
with st.sidebar:
    st.header('⚙️ Settings')
    tune = st.toggle('Hyperparameter Tuning', value=True)
    st.divider()
    st.markdown('**Steps**')
    st.markdown('1. Upload CSV\n2. Select target\n3. Click Run\n4. See results!')

# Upload
uploaded = st.file_uploader('📂 Upload your CSV', type='csv')

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f'✅ Loaded — {df.shape[0]} rows × {df.shape[1]} columns')
    st.dataframe(df.head(3), use_container_width=True)
    st.divider()

    # Target select
    target = st.selectbox('🎯 Select Target Column', df.columns.tolist(),
                           index=len(df.columns)-1)

    if st.button('🚀 Run AutoML', type='primary', use_container_width=True):

        # ── Setup
        y_raw = df[target].copy()
        X     = df.drop(columns=[target])
        drop_cols = [c for c in X.columns if c.lower() in {'id','index','unnamed: 0'}]
        X.drop(columns=drop_cols, inplace=True, errors='ignore')

        PROBLEM, reason = detect_problem(y_raw)
        le = None
        if PROBLEM == 'classification':
            le = LabelEncoder()
            y  = pd.Series(le.fit_transform(y_raw.astype(str)), name=target)
        else:
            y  = y_raw.astype(float)

        preprocessor, num_cols, cat_cols = build_preprocessor(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if PROBLEM=='classification' else None)

        # ── Problem Detection
        st.subheader('🔍 Problem Detection')
        col1, col2, col3 = st.columns(3)
        col1.metric('Problem Type', PROBLEM.upper())
        col2.metric('Dataset Size', f'{len(df):,} rows')
        col3.metric('Features', X.shape[1])
        st.info(f'**Reason:** {reason}')
        st.divider()

        # ── Training
        st.subheader('🏋️ Training Models...')
        candidates = get_candidates(PROBLEM, len(df))
        progress   = st.progress(0)
        results    = []
        scoring    = 'accuracy' if PROBLEM == 'classification' else 'r2'

        for i, (name, estimator, params) in enumerate(candidates):
            with st.spinner(f'Training {name}...'):
                pipe = Pipeline([('pre', preprocessor), ('clf', estimator)])
                if tune and params:
                    search = RandomizedSearchCV(pipe, params, n_iter=6, cv=3,
                                                scoring=scoring, random_state=42, n_jobs=-1)
                    search.fit(X_train, y_train)
                    pipe = search.best_estimator_
                else:
                    pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                if PROBLEM == 'classification':
                    score = accuracy_score(y_test, y_pred)
                    extra = {'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)}
                else:
                    score = r2_score(y_test, y_pred)
                    extra = {'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                             'MAE' : mean_absolute_error(y_test, y_pred)}
                results.append({'name': name, 'pipe': pipe, 'score': score, 'extra': extra})
                progress.progress((i+1) / len(candidates))

        results.sort(key=lambda r: r['score'], reverse=True)
        best = results[0]
        st.divider()

        # ── Best Model
        st.subheader('🏆 Best Model')
        m1, m2, m3 = st.columns(3)
        metric_label = 'Accuracy' if PROBLEM == 'classification' else 'R² Score'
        m1.metric('🥇 Best Model', best['name'])
        m2.metric(metric_label, f'{best["score"]:.4f}')
        for k, v in best['extra'].items():
            m3.metric(k, f'{v:.4f}')
        st.divider()

        # ── Model Comparison Plot
        st.subheader('📊 Model Comparison')
        names  = [r['name']  for r in results]
        scores = [r['score'] for r in results]
        colors = [COLORS['best'] if n == best['name'] else COLORS['others'] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        bars = axes[0].barh(names[::-1], scores[::-1], color=colors[::-1],
                            height=0.5, edgecolor='none')
        for bar, sc in zip(bars, scores[::-1]):
            axes[0].text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                         f'{sc:.3f}', va='center', fontsize=10,
                         fontweight='bold', color=COLORS['text'])
        axes[0].set_xlim(0, min(1.0, max(scores)*1.2))
        axes[0].set_title('Model Leaderboard', fontsize=12, fontweight='bold')
        axes[0].axvline(best['score'], color=COLORS['accent'], lw=1.5, ls='--')

        baseline = scores[-1]
        deltas   = [s - baseline for s in scores]
        dcols    = [COLORS['best'] if d==max(deltas) else '#CBD5E1' for d in deltas]
        axes[1].bar(names, deltas, color=dcols, edgecolor='none', width=0.5)
        axes[1].set_title('Improvement over Baseline', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        st.divider()

        # ── Confusion Matrix + ROC
        if PROBLEM == 'classification':
            st.subheader('📈 Evaluation Plots')
            y_pred_best = best['pipe'].predict(X_test)
            col_cm, col_roc = st.columns(2)

            with col_cm:
                cm   = confusion_matrix(y_test, y_pred_best)
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                ConfusionMatrixDisplay(cm, display_labels=le.classes_ if le else None).plot(
                    ax=ax2, colorbar=False, cmap='Blues')
                ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig2)

            with col_roc:
                y_proba      = best['pipe'].predict_proba(X_test)[:, 1]
                fpr, tpr, _  = roc_curve(y_test, y_proba)
                roc_auc      = auc(fpr, tpr)
                fig3, ax3    = plt.subplots(figsize=(5, 4))
                ax3.plot(fpr, tpr, color=COLORS['best'], lw=2,
                         label=f'AUC = {roc_auc:.3f}')
                ax3.plot([0,1],[0,1], color=COLORS['others'], ls='--', lw=1.5)
                ax3.fill_between(fpr, tpr, alpha=0.08, color=COLORS['best'])
                ax3.set_xlabel('False Positive Rate')
                ax3.set_ylabel('True Positive Rate')
                ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
                ax3.legend()
                plt.tight_layout()
                st.pyplot(fig3)
            st.divider()

        # ── Feature Importance
        st.subheader('🔍 Feature Importance')
        estimator  = best['pipe'].named_steps['clf']
        pre        = best['pipe'].named_steps['pre']
        feat_names = list(num_cols)
        try:
            enc = pre.named_transformers_['cat'].named_steps['enc']
            feat_names += list(enc.get_feature_names_out(cat_cols))
        except Exception:
            feat_names += cat_cols

        if hasattr(estimator, 'feature_importances_'):
            imp = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            c   = estimator.coef_
            imp = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)
        else:
            imp = None

        if imp is not None:
            n  = min(len(feat_names), len(imp))
            fi = (pd.DataFrame({'feature': feat_names[:n], 'importance': imp[:n]})
                    .sort_values('importance', ascending=False).head(10))
            fig4, ax4 = plt.subplots(figsize=(9, 4))
            bar_colors = [COLORS['best'] if i==0 else '#7C3AED' if i<3
                          else COLORS['others'] for i in range(len(fi))]
            ax4.barh(fi['feature'][::-1], fi['importance'][::-1],
                     color=bar_colors[::-1], height=0.55, edgecolor='none')
            ax4.set_title(f'Top Features — {best["name"]}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig4)
            st.dataframe(fi.reset_index(drop=True), use_container_width=True)
        st.divider()

        # ── Save + Download
        st.subheader('💾 Download Best Model')
        joblib.dump(best['pipe'], 'best_model.pkl')
        with open('best_model.pkl', 'rb') as f:
            st.download_button('⬇️ Download best_model.pkl', f,
                               file_name='best_model.pkl')

        # ── Final Report
        st.subheader('📋 Final Report')
        report = f"""
{'='*45}
        🤖 AutoML SYSTEM — FINAL REPORT
{'='*45}
Dataset   : {len(df)} rows × {X.shape[1]} features
Target    : {target}
Problem   : {PROBLEM.upper()}
Best Model: {best['name']}
{metric_label:<10}: {best['score']:.4f}
{'='*45}
All Models:
"""
        for i, r in enumerate(results):
            marker = '🏆' if i == 0 else '  '
            report += f"\n{marker} {r['name']:<25} {metric_label} = {r['score']:.4f}"
        st.code(report)
