# -*- coding: utf-8 -*-
"""
SmartThesis UMPO — Enterprise Academic Research Discovery & Hybrid RAG-Lite Thesis Recommendation System
Universitas Muhammadiyah Ponorogo | Program Studi Teknik Informatika
Tim Pengembang:
1. Prasetya Riski Wa'afan (23533743) — https://prasetyarizki.my.id
2. Agya Gilar Cahyono (23533753)
3. Muhammad Taufik Hidayat (23533739)
4. Galang Finto Anergi (23533835)
5. Muhammad Nur Arifin (23533792)
"""

import os
import re
import json
import time
import pickle
import urllib.request
import urllib.error
import html as html_lib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. KONFIGURASI HALAMAN & META SEO UNTUK HOSTING SERVER PRODUKSI
# ==============================================================================
st.set_page_config(
    page_title="SmartThesis UMPO | Sistem Cerdas Rekomendasi Skripsi & Asisten Riset AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://prasetyarizki.my.id",
        "Report a bug": "mailto:pr0165341@gmail.com",
        "About": "SmartThesis UMPO — Sistem Pendukung Pemilihan Judul Skripsi Lintas Prodi Menggunakan TF-IDF N-Gram, Cosine Similarity & Hybrid LLM Assistant."
    }
)

# ==============================================================================
# 2. DESAIN UI/UX MODERN EDITORIAL AKADEMIK (FLUID & ORGANIC — ANTI KAKU)
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;1,9..144,500&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ==========================================================================
   DESIGN TOKENS — "Espresso & Warm Crema" (Luxury Academic Editorial & Archive)
   Deep roast espresso + warm alabaster linen + solar honey amber
   ========================================================================== */
:root {
    --bg-canvas: #F7F5F0;
    --bg-surface: #FFFFFF;
    --bg-elevated: rgba(255, 255, 255, 0.92);
    --bg-subtle: #EFECE4;
    --bg-hover: #E5E0D5;

    --text-primary: #1C1613;
    --text-secondary: #4A3E37;
    --text-muted: #6B5F57;

    --brand: #8A4F1D;
    --brand-hover: #6D3D14;
    --brand-light: #FBF4EC;
    --brand-border: #E8D3BD;
    --on-brand: #FFFFFF;

    --gold: #C27803;
    --gold-fill: #D97706;
    --gold-light: #FDF6E2;
    --gold-border: #F7E2A8;

    --rel-high: #2D6A4F;
    --rel-high-bg: #D8F3DC;
    --rel-high-bd: #95D5B2;
    --rel-med: #C27803;
    --rel-med-bg: #FEF3C7;
    --rel-med-bd: #FCD34D;
    --rel-low: #6E635C;
    --rel-low-bg: #EFECE4;
    --rel-low-bd: #D4C9BF;

    --border-hairline: rgba(28, 22, 19, 0.08);
    --border-strong: rgba(28, 22, 19, 0.16);

    --shadow-soft: 0 1px 2px rgba(28, 22, 19, .04), 0 8px 24px -10px rgba(28, 22, 19, .08);
    --shadow-hover: 0 2px 4px rgba(28, 22, 19, .05), 0 22px 44px -14px rgba(138, 79, 29, .24);
    --glow: 0 0 0 4px rgba(138, 79, 29, .18);

    --hero-a: rgba(138, 79, 29, .14);
    --hero-b: rgba(194, 120, 3, .12);
    --highlight-bg: #FDE68A;
    --highlight-fg: #1C1613;

    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 22px;
    --radius-xl: 28px;
    --radius-full: 9999px;

    --ease-out: cubic-bezier(.22, 1, .36, 1);
    --transition-spring: .35s var(--ease-out);
    --transition-base: .18s ease;

    --font-display: 'Fraunces', Georgia, serif;
    --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-canvas: #130E0C;
        --bg-surface: #1C1613;
        --bg-elevated: rgba(28, 22, 19, 0.88);
        --bg-subtle: #27201C;
        --bg-hover: #332B26;

        --text-primary: #F7F5F0;
        --text-secondary: #D4C9BF;
        --text-muted: #B3A599;

        --brand: #E09852;
        --brand-hover: #EBB074;
        --brand-light: rgba(224, 152, 82, .16);
        --brand-border: rgba(224, 152, 82, .35);
        --on-brand: #130E0C;

        --gold: #F5B041;
        --gold-fill: #F5B041;
        --gold-light: rgba(245, 176, 65, .16);
        --gold-border: rgba(245, 176, 65, .35);

        --rel-high: #52B788;
        --rel-high-bg: rgba(45, 106, 79, .25);
        --rel-high-bd: rgba(82, 183, 136, .45);
        --rel-med: #F5B041;
        --rel-med-bg: rgba(245, 176, 65, .20);
        --rel-med-bd: rgba(245, 176, 65, .45);
        --rel-low: #B8ABA0;
        --rel-low-bg: rgba(184, 171, 160, .20);
        --rel-low-bd: rgba(184, 171, 160, .40);

        --border-hairline: rgba(247, 245, 240, 0.08);
        --border-strong: rgba(247, 245, 240, 0.18);

        --shadow-soft: 0 1px 2px rgba(0, 0, 0, .45), 0 10px 28px -10px rgba(0, 0, 0, .6);
        --shadow-hover: 0 2px 4px rgba(0, 0, 0, .5), 0 26px 52px -16px rgba(224, 152, 82, .32);
        --glow: 0 0 0 4px rgba(224, 152, 82, .28);

        --hero-a: rgba(224, 152, 82, .20);
        --hero-b: rgba(245, 176, 65, .14);
        --highlight-bg: rgba(251, 191, 36, .28);
        --highlight-fg: #FEF3C7;
    }
}

/* ==========================================================================
   BASE
   ========================================================================== */
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
    font-family: var(--font-body) !important;
}

.stApp {
    background:
        radial-gradient(900px 420px at 0% -10%, var(--hero-a), transparent 60%),
        radial-gradient(800px 380px at 100% -5%, var(--hero-b), transparent 55%),
        var(--bg-canvas) !important;
    color: var(--text-primary);
}

.stMarkdown, .stMarkdown p, .stMarkdown li, label, [data-testid="stWidgetLabel"] p {
    color: var(--text-primary);
}
[data-testid="stCaptionContainer"], .stCaption, small { color: var(--text-muted) !important; }

h1, h2, h3, h4, h5 {
    font-family: var(--font-display) !important;
    letter-spacing: -0.015em;
    color: var(--text-primary) !important;
}

[data-testid="stHeaderActionElements"], .header-anchor { display: none !important; }

/* Header dibuat transparan (bukan disembunyikan) agar tombol sidebar tetap muncul di mobile */
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"], [data-testid="stAppDeployButton"], footer { display: none !important; }

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1120px !important;
}

:focus-visible { outline: 2px solid var(--brand) !important; outline-offset: 2px !important; }
::selection { background: var(--brand-light); color: var(--text-primary); }

/* ==========================================================================
   HERO KOMPAK (judul + 1 kalimat + statistik satu baris)
   ========================================================================== */
.app-hero { padding: .5rem 0 1.5rem; }
.app-eyebrow {
    display: inline-flex; align-items: center; gap: .5rem;
    font-size: .82rem; font-weight: 600; color: var(--brand);
    margin-bottom: .7rem;
}
.app-eyebrow::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--brand); }
.app-title {
    font-family: var(--font-display) !important;
    font-size: clamp(1.75rem, 1.2rem + 2.2vw, 2.7rem);
    font-weight: 700; line-height: 1.15; letter-spacing: -0.03em;
    margin: 0 0 .65rem 0; max-width: 820px; color: var(--text-primary);
}
.app-title em { font-style: normal; color: var(--brand); }
.app-sub {
    font-size: 1.02rem; color: var(--text-secondary);
    line-height: 1.65; margin: 0 0 1rem 0; max-width: 680px;
}
.app-stats {
    display: flex; flex-wrap: wrap; align-items: center; gap: .3rem .65rem;
    font-size: .9rem; color: var(--text-muted);
}
.app-stats b { color: var(--text-primary); font-weight: 700; font-variant-numeric: tabular-nums; }
.app-stats .sep { opacity: .5; }

/* ==========================================================================
   TABS
   ========================================================================== */
.stTabs [data-baseweb="tab-list"] {
    display: inline-flex !important; width: auto !important; max-width: 100%;
    overflow-x: auto; gap: 4px !important; padding: 4px !important;
    margin-bottom: 1.2rem !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border-hairline) !important;
    border-radius: var(--radius-full) !important;
    scrollbar-width: none;
}
.stTabs [data-baseweb="tab"] {
    height: auto !important; min-height: 42px;
    padding: .5rem 1.2rem !important;
    border: 1px solid transparent !important; border-radius: var(--radius-full) !important;
    background: transparent !important;
    font-weight: 600 !important; font-size: .88rem !important;
    transition: color var(--transition-base), background var(--transition-base) !important;
}
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p { color: var(--text-secondary) !important; }
.stTabs [data-baseweb="tab"]:hover, .stTabs [data-baseweb="tab"]:hover p { color: var(--text-primary) !important; }
/* background/shadow hanya pada tab; anak elemen hanya mewarisi warna */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--bg-surface) !important;
    border-color: var(--border-hairline) !important;
    box-shadow: var(--shadow-soft) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: var(--brand) !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ==========================================================================
   INPUT & TOMBOL
   ========================================================================== */
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--bg-surface) !important;
    transition: border-color var(--transition-base), box-shadow var(--transition-base) !important;
}
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea { color: var(--text-primary) !important; background: transparent !important; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
    border-color: var(--brand) !important; box-shadow: var(--glow) !important;
}

/* ==========================================================================
   PENCARIAN — input besar, topik cepat berupa chip kecil satu baris
   ========================================================================== */
div[data-testid="stTextInput"] div[data-baseweb="input"] {
    min-height: 54px !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--bg-surface) !important;
}
div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
    border-color: var(--brand) !important; box-shadow: var(--glow) !important;
}
div[data-testid="stTextInput"] div[data-baseweb="input"] input {
    font-size: 1.05rem !important; font-weight: 500 !important; padding: .8rem 1.1rem !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { min-height: 54px !important; }

.st-key-presets {
    flex-direction: row !important; flex-wrap: wrap !important;
    gap: .5rem !important; margin: .1rem 0 1.4rem 0;
}
.st-key-presets > div { width: auto !important; flex: 0 0 auto !important; }
[class*="st-key-preset_"] div.stButton > button {
    min-height: 34px !important; padding: .2rem .9rem !important;
    font-size: .84rem !important; font-weight: 500 !important;
    background: var(--bg-subtle) !important; border-color: transparent !important;
    color: var(--text-secondary) !important; box-shadow: none !important;
}
[class*="st-key-preset_"] div.stButton > button:hover {
    background: var(--brand-light) !important; border-color: var(--brand-border) !important;
    color: var(--brand) !important; transform: none !important;
}

div.stButton > button, div.stDownloadButton > button, a[data-testid="stBaseLinkButton-secondary"] {
    min-height: 44px;
    border-radius: var(--radius-full) !important;
    font-weight: 600 !important; font-size: .85rem !important;
    padding: .5rem 1.15rem !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    transition: transform var(--transition-base), box-shadow var(--transition-base), border-color var(--transition-base) !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover, a[data-testid="stBaseLinkButton-secondary"]:hover {
    border-color: var(--brand) !important;
}
div.stButton > button:active { transform: translateY(0); }

div.stButton > button[kind="primary"] {
    background: var(--brand) !important; border-color: var(--brand) !important;
    color: var(--on-brand) !important;
    box-shadow: none !important;
}
div.stButton > button[kind="primary"] p { color: var(--on-brand) !important; }
div.stButton > button[kind="primary"]:hover { background: var(--brand-hover) !important; border-color: var(--brand-hover) !important; }

/* ==========================================================================
   FILTER AKTIF (hanya tampil bila berbeda dari default)
   ========================================================================== */
.filter-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: 0 0 .9rem; align-items: center; }
.filter-chips .lbl { font-size: .8rem; font-weight: 600; color: var(--text-muted); margin-right: .2rem; }
.f-chip {
    font-size: .8rem; font-weight: 600;
    color: var(--brand); background: var(--brand-light); border: 1px solid var(--brand-border);
    padding: .15rem .7rem; border-radius: var(--radius-full);
}

/* ==========================================================================
   KARTU HASIL — ringkas: peringkat + relevansi, judul, 1 baris meta, abstrak
   ========================================================================== */
[class*="st-key-res_"] {
    --accent: var(--rel-high);
    position: relative; overflow: hidden;
    padding: 1.3rem 1.5rem 1rem 1.7rem !important;
    margin-bottom: .9rem;
    border-radius: var(--radius-lg);
    background: var(--bg-surface);
    border: 1px solid var(--border-hairline);
    transition: border-color var(--transition-base), box-shadow var(--transition-base);
    animation: rise .45s var(--ease-out) both;
}
[class*="st-key-res_high_"] { --accent: var(--rel-high); }
[class*="st-key-res_med_"]  { --accent: var(--rel-med); }
[class*="st-key-res_low_"]  { --accent: var(--rel-low); }
[class*="st-key-res_"]::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent);
}
[class*="st-key-res_"]:hover { border-color: var(--border-strong); box-shadow: var(--shadow-soft); }
@keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }

.thesis-card { position: relative; }
.card-top { display: flex; justify-content: space-between; align-items: center; gap: .75rem; flex-wrap: wrap; margin-bottom: .65rem; }
.rank-tag { font-size: .82rem; font-weight: 700; color: var(--text-muted); }
.sim-badge-high, .sim-badge-med, .sim-badge-low {
    font-size: .78rem; font-weight: 600;
    padding: .2rem .75rem; border-radius: var(--radius-full); border: 1px solid transparent;
    font-variant-numeric: tabular-nums;
}
.sim-badge-high { color: var(--rel-high); background: var(--rel-high-bg); border-color: var(--rel-high-bd); }
.sim-badge-med  { color: var(--rel-med);  background: var(--rel-med-bg);  border-color: var(--rel-med-bd); }
.sim-badge-low  { color: var(--rel-low);  background: var(--rel-low-bg);  border-color: var(--rel-low-bd); }

.thesis-title {
    font-family: var(--font-display) !important;
    font-size: clamp(1.08rem, 1rem + .35vw, 1.28rem);
    font-weight: 700; line-height: 1.4; letter-spacing: -.01em;
    color: var(--text-primary); margin: 0 0 .5rem 0;
}
.meta-line {
    display: flex; flex-wrap: wrap; align-items: center; gap: .15rem .55rem;
    font-size: .86rem; color: var(--text-muted); margin-bottom: .75rem;
}
.meta-line b { color: var(--text-secondary); font-weight: 600; }
.meta-line .sep { opacity: .5; }

.abstract-text { font-size: .95rem; color: var(--text-secondary); line-height: 1.7; margin-bottom: .6rem; }
.term-highlight { background: var(--highlight-bg); color: var(--highlight-fg); padding: .02rem .25rem; border-radius: 4px; font-weight: 700; }

/* Detail tambahan disembunyikan (progressive disclosure) */
details.more { margin: .4rem 0 .5rem; }
details.more > summary {
    cursor: pointer; list-style: none; display: inline-flex; align-items: center; gap: .45rem;
    font-size: .85rem; font-weight: 600; color: var(--brand);
    padding: .25rem 0;
}
details.more > summary::-webkit-details-marker { display: none; }
details.more > summary::before { content: "+"; font-weight: 700; width: 1ch; }
details.more[open] > summary::before { content: "−"; }
details.more > summary:hover { text-decoration: underline; }
details.more .more-body { padding-top: .6rem; animation: rise .3s var(--ease-out) both; }

.matched-terms-bar { font-size: .85rem; color: var(--text-muted); margin-bottom: .2rem; }
.term-pill {
    display: inline-block; margin: .3rem .4rem 0 0;
    font-family: var(--font-mono); font-size: .75rem;
    color: var(--brand); background: var(--brand-light); border: 1px solid var(--brand-border);
    padding: .12rem .6rem; border-radius: var(--radius-full);
}
.novelty-box {
    margin-top: .8rem; padding: .9rem 1.1rem;
    border-radius: var(--radius-md);
    background: var(--gold-light);
    border: 1px solid var(--gold-border); border-left: 3px solid var(--gold-fill);
    font-size: .9rem; color: var(--text-secondary); line-height: 1.65;
}
.novelty-box b { color: var(--text-primary); }

/* Container ber-border (panel AI, simulator) */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--brand-border) !important;
    background: var(--bg-surface);
    box-shadow: var(--shadow-soft);
    padding: .4rem .6rem;
}

.empty-state-panel {
    text-align: center; margin: 1.5rem 0; padding: 2.5rem 2rem;
    border-radius: var(--radius-lg); border: 1.5px dashed var(--border-strong);
    background: var(--bg-surface); box-shadow: var(--shadow-soft);
}
.empty-state-icon { width: 54px; height: 54px; margin: 0 auto .9rem; color: var(--brand); }
.empty-state-icon svg { width: 100%; height: 100%; }
.empty-state-title { font-family: var(--font-display); font-size: 1.3rem; font-weight: 700; color: var(--text-primary); margin-bottom: .45rem; }
.empty-state-desc { max-width: 580px; margin: 0 auto 1.2rem; font-size: .92rem; line-height: 1.65; color: var(--text-muted); }
.empty-state-hint {
    display: inline-block; padding: .5rem 1.15rem; font-size: .82rem;
    color: var(--brand); background: var(--brand-light);
    border: 1px solid var(--brand-border); border-radius: var(--radius-full);
}

/* ==========================================================================
   SIDEBAR & KOMPONEN NATIF
   ========================================================================== */
section[data-testid="stSidebar"] h4 { font-size: .95rem !important; font-weight: 700; margin: 1.1rem 0 .35rem; }
[data-testid="stWidgetLabel"] p { font-size: .88rem; font-weight: 600; }
section[data-testid="stSidebar"] {
    background: var(--bg-elevated) !important;
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid var(--border-hairline) !important;
}
[data-testid="stExpander"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-hairline) !important;
    background: var(--bg-surface) !important;
    box-shadow: var(--shadow-soft); overflow: hidden;
}
[data-testid="stExpander"] details summary, [data-testid="stExpander"] details summary * { color: var(--text-primary) !important; }

code {
    background: var(--bg-subtle) !important; color: var(--brand) !important;
    border: 1px solid var(--border-hairline) !important;
    padding: .15rem .45rem !important; border-radius: 6px !important;
    font-family: var(--font-mono) !important;
}

[data-testid="stMetric"] {
    padding: 1rem 1.15rem; border-radius: var(--radius-md);
    background: var(--bg-surface); border: 1px solid var(--border-hairline); box-shadow: var(--shadow-soft);
}
[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; font-weight: 700; }
[data-testid="stAlert"] { border-radius: var(--radius-md) !important; border: 1px solid var(--border-hairline) !important; }
[data-testid="stStatusWidget"], [data-testid="stStatus"] { border-radius: var(--radius-md) !important; }
[data-testid="stDataFrame"] { border-radius: var(--radius-md); overflow: hidden; border: 1px solid var(--border-hairline); box-shadow: var(--shadow-soft); }

/* Scroll-driven reveal (peningkatan progresif) */
@supports (animation-timeline: view()) {
    [class*="st-key-res_"] {
        animation: rise linear both;
        animation-timeline: view();
        animation-range: entry 0% entry 35%;
    }
}

/* ==========================================================================
   RESPONSIF & REDUCED MOTION
   ========================================================================== */
@media (max-width: 768px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    [class*="st-key-res_"] { padding: 1.1rem 1rem .9rem 1.3rem !important; }
}
@media (max-width: 480px) {
    .stTabs [data-baseweb="tab"] { padding: .5rem .9rem !important; }
}
@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-duration: .01ms !important; animation-iteration-count: 1 !important;
        transition-duration: .01ms !important; scroll-behavior: auto !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Ikon SVG inline untuk antarmuka (menggantikan emoji sebagai ikon utama)
# ------------------------------------------------------------------------------
def _svg(path: str, size_cls: str = "") -> str:
    return (f'<svg class="{size_cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{path}</svg>')

ICON_BUILDING = _svg('<path d="M3 21h18M5 21V7l7-4 7 4v14M9 21v-6h6v6M9 10h.01M15 10h.01"/>', "badge-ic")
ICON_BOOKS = _svg('<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5v14zM4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/>')
ICON_TARGET = _svg('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>')
ICON_SCALE = _svg('<path d="M12 3v18M5 7h14M5 7l-3 7a3 3 0 0 0 6 0L5 7zM19 7l-3 7a3 3 0 0 0 6 0l-3-7z"/>')
ICON_BOLT = _svg('<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8z"/>')
ICON_SEARCH_BIG = _svg('<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5M8.5 11h5"/>')
ICON_MC_FAK = _svg('<path d="M3 21h18M5 21V7l7-4 7 4v14"/>')
ICON_MC_PRODI = _svg('<path d="m2 9 10-5 10 5-10 5L2 9zM6 11.5V16c0 1.5 3 3 6 3s6-1.5 6-3v-4.5"/>')
ICON_MC_METODE = _svg('<path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3"/>')
ICON_MC_TAHUN = _svg('<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18"/>')

# ==============================================================================
# 3. KONSTANTA STOPWORDS TALA & UTILITAS PEMROSESAN TEKS BAHASA INDONESIA
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_CSV_PATHS = [
    os.path.join(BASE_DIR, "models", "thesis_database_all_prodi.csv"),
    os.path.join(BASE_DIR, "thesis_database_all_prodi.csv"),
    "thesis_database_all_prodi.csv"
]
CANDIDATE_PKL_PATHS = [
    os.path.join(BASE_DIR, "models", "thesis_tfidf_model.pkl"),
    os.path.join(BASE_DIR, "thesis_tfidf_model.pkl"),
    "thesis_tfidf_model.pkl"
]

DEFAULT_STOPWORDS = {
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia", "dua",
    "ia", "seperti", "jika", "sehingga", "kembali", "dan", "tidak", "ini", "karena",
    "kepada", "oleh", "saat", "harus", "sementara", "setelah", "belum", "kami", "sekitar",
    "bagi", "serta", "di", "dari", "telah", "sebagai", "masih", "hal", "ketika", "adalah",
    "itu", "dalam", "bisa", "bahwa", "atau", "hanya", "kita", "dengan", "akan", "juga",
    "ada", "mereka", "sudah", "saya", "terhadap", "secara", "agar", "lain", "apakah",
    "melalui", "studi", "kasus", "penelitian", "menggunakan", "berbasis", "analisis"
}

def clean_indonesian_text(text: str, stopwords_set=DEFAULT_STOPWORDS) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    tokens = [t for t in text.split() if t not in stopwords_set and len(t) > 2]
    return " ".join(tokens)

def infer_fakultas_prodi_if_missing(row):
    """Memastikan setiap baris data hasil scraping EPrints memiliki Fakultas & Prodi yang valid."""
    fak_raw = str(row.get("fakultas", "")).strip()
    prodi_raw = str(row.get("program_studi", "")).strip()
    if prodi_raw not in ["", "Umum", "Lainnya", "nan", "None"] and "Faculty" not in fak_raw:
        return fak_raw, prodi_raw

    teks = (str(row.get("judul", "")) + " " + str(row.get("abstrak", ""))).lower()
    fak_low = fak_raw.lower()

    if any(k in teks for k in ["informatika", "algoritma", "aplikasi", "website", "web", "android", "sistem informasi", "citra", "data mining", "jaringan", "spk", "pakar", "game", "ui/ux", "yolo", "cnn"]):
        return "Fakultas Teknik", "Teknik Informatika S1"
    elif any(k in teks for k in ["elektro", "listrik", "tegangan", "arus", "daya", "mikrokontroler", "arduino", "iot", "esp32", "sensor", "surya", "plc", "motor induksi", "inverter"]):
        return "Fakultas Teknik", "Teknik Elektro S1"
    elif any(k in teks for k in ["mesin", "pengelasan", "baja", "logam", "mekanis", "konversi energi", "turbin", "komposit", "manufaktur", "korosi", "aluminium", "cnc"]):
        return "Fakultas Teknik", "Teknik Mesin S1"
    elif any(k in teks for k in ["akuntansi", "pajak", "auditing", "fraud", "laporan keuangan", "laba", "aset", "piutang", "perpajakan", "keuangan desa", "financial distress"]):
        return "Fakultas Ekonomi dan Bisnis", "Akuntansi S1"
    elif any(k in teks for k in ["manajemen", "pemasaran", "konsumen", "pembelian", "karyawan", "kinerja", "sdm", "kepemimpinan", "umkm", "brand", "harga", "kualitas pelayanan", "saham", "bank"]):
        return "Fakultas Ekonomi dan Bisnis", "Manajemen S1"
    elif any(k in teks for k in ["keperawatan", "pasien", "asuhan", "klinis", "nyeri", "hipertensi", "diabetes", "ansietas", "lansia", "perawat", "rumah sakit", "kesehatan", "terapi", "gastritis"]):
        return "Fakultas Ilmu Kesehatan", "Keperawatan S1"
    elif any(k in teks for k in ["kebidanan", "ibu hamil", "kehamilan", "persalinan", "nifas", "bayi", "balita", "asi", "stunting", "kontrasepsi", "bidan"]):
        return "Fakultas Ilmu Kesehatan", "Kebidanan D3"
    elif any(k in teks for k in ["agama islam", "pai", "al-qur", "akhlak", "fiqih", "pesantren", "madrasah", "dakwah", "santri", "ibadah", "syariah", "zakat", "wakaf"]):
        return "Fakultas Agama Islam", "Pendidikan Agama Islam S1"
    elif any(k in teks for k in ["pemerintahan", "kebijakan publik", "desa", "bupati", "dprd", "politik", "pemilu", "birokrasi", "pelayanan publik", "komunikasi", "media", "jurnalistik"]):
        return "FISIP", "Ilmu Pemerintahan & Komunikasi S1"
    elif any(k in teks for k in ["siswa", "pembelajaran", "guru", "sekolah", "matematika", "pendidikan", "kurikulum", "belajar", "paud", "anak usia dini", "bahasa inggris", "ppkn", "geogebra"]):
        return "FKIP", "Pendidikan Matematika S1"

    if "engineering" in fak_low or "teknik" in fak_low:
        return "Fakultas Teknik", "Teknik Informatika S1"
    elif "economic" in fak_low or "ekonomi" in fak_low:
        return "Fakultas Ekonomi dan Bisnis", "Manajemen S1"
    elif "health" in fak_low or "kesehatan" in fak_low:
        return "Fakultas Ilmu Kesehatan", "Keperawatan S1"
    elif "social" in fak_low or "political" in fak_low or "fisip" in fak_low:
        return "FISIP", "Ilmu Pemerintahan & Komunikasi S1"
    elif "islamic" in fak_low or "agama" in fak_low:
        return "Fakultas Agama Islam", "Pendidikan Agama Islam S1"
    return "Fakultas Teknik", "Teknik Informatika S1"

def infer_metode_dan_rumpun(row):
    """Menentukan metode utama dan rumpun keilmuan secara otomatis jika belum tersedia di CSV."""
    teks = (str(row.get("judul", "")) + " " + str(row.get("abstrak", ""))).lower()
    prodi = str(row.get("program_studi", ""))

    metode = str(row.get("metode_utama", "")).strip()
    if metode in ["", "nan", "None", "Analisis Ilmiah"]:
        if "yolo" in teks: metode = "Deep Learning YOLO Object Detection"
        elif "cnn" in teks or "convolutional" in teks or "mobilenet" in teks: metode = "Convolutional Neural Network (CNN)"
        elif "tf-idf" in teks or "cosine" in teks: metode = "TF-IDF & Cosine Similarity"
        elif "svm" in teks or "support vector" in teks: metode = "Support Vector Machine (SVM)"
        elif "naive bayes" in teks: metode = "Klasifikasi Naive Bayes"
        elif "random forest" in teks or "decision tree" in teks or "c4.5" in teks: metode = "Decision Tree / Random Forest"
        elif "fuzzy" in teks or "anfis" in teks: metode = "Sistem Kendali Logika Fuzzy"
        elif "iot" in teks or "esp32" in teks or "arduino" in teks or "mikrokontroler" in teks: metode = "Internet of Things (IoT) & Embedded System"
        elif "saw" in teks or "topsis" in teks or "ahp" in teks or "moora" in teks: metode = "SPK Multi-Criteria Decision Making (MCDM)"
        elif "regresi" in teks or "partial least square" in teks or "sem" in teks: metode = "Analisis Regresi & Ekonometrika"
        elif "chi-square" in teks or "cross sectional" in teks or "rank spearman" in teks: metode = "Uji Korelasi Klinis Cross-Sectional"
        elif "altman" in teks or "rasio keuangan" in teks: metode = "Analisis Rasio Keuangan & Altman Z-Score"
        elif "r&d" in teks or "add" in teks or "pengembangan" in teks: metode = "Research & Development (R&D / ADDIE)"
        else: metode = "Analisis Empiris & Pemodelan Sistem"

    rumpun = str(row.get("rumpun_minat", "")).strip()
    if rumpun in ["", "nan", "None", "Multi-Disiplin"]:
        if "Informatika" in prodi: rumpun = "Computer Vision, NLP, Data Mining & Rekayasa Perangkat Lunak"
        elif "Elektro" in prodi: rumpun = "Sistem Kendali, IoT Daya & Energi Terbarukan"
        elif "Mesin" in prodi: rumpun = "Manufaktur, Konversi Energi & Metalurgi"
        elif "Manajemen" in prodi: rumpun = "Digital Marketing UMKM & Manajemen SDM"
        elif "Akuntansi" in prodi: rumpun = "Sistem Informasi Akuntansi & Audit Keuangan"
        elif "Keperawatan" in prodi: rumpun = "Keperawatan Klinis, Komunitas & Gerontik"
        elif "Kebidanan" in prodi: rumpun = "Kesehatan Ibu-Anak & Pencegahan Stunting"
        elif "Agama Islam" in prodi: rumpun = "Pendidikan Karakter & Studi Islam Kontemporer"
        elif "Pemerintahan" in prodi: rumpun = "E-Government, Kebijakan Publik & Smart Village"
        else: rumpun = "Teknologi & Inovasi Pembelajaran"

    return metode, rumpun

# ==============================================================================
# 4. PEMUATAN ARTIFAK MODEL HEMAT TOKEN (@st.cache_resource)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_smartthesis_engine():
    csv_found = next((p for p in CANDIDATE_CSV_PATHS if os.path.exists(p)), None)
    pkl_found = next((p for p in CANDIDATE_PKL_PATHS if os.path.exists(p)), None)

    if csv_found:
        df = pd.read_csv(csv_found)
        df = df.dropna(subset=["judul"]).drop_duplicates(subset=["judul"]).reset_index(drop=True)
        mapped_fp = df.apply(infer_fakultas_prodi_if_missing, axis=1)
        df["fakultas"] = [x[0] for x in mapped_fp]
        df["program_studi"] = [x[1] for x in mapped_fp]

        mapped_mr = df.apply(infer_metode_dan_rumpun, axis=1)
        df["metode_utama"] = [x[0] for x in mapped_mr]
        df["rumpun_minat"] = [x[1] for x in mapped_mr]

        if "tahun" not in df.columns:
            df["tahun"] = 2024
        df["tahun"] = df["tahun"].fillna(2024).astype(str).str.extract(r"(\d{4})")[0].fillna("2024")

        if "url" not in df.columns:
            df["url"] = "https://eprints.umpo.ac.id"
        df["url"] = df["url"].fillna("https://eprints.umpo.ac.id")

        if "abstrak" not in df.columns:
            df["abstrak"] = df["judul"]
        df["abstrak"] = df["abstrak"].fillna(df["judul"])

        if pkl_found:
            try:
                with open(pkl_found, "rb") as f:
                    art = pickle.load(f)
                if art["tfidf_matrix"].shape[0] == len(df):
                    return df, art["vectorizer"], art["tfidf_matrix"], art.get("stopwords", DEFAULT_STOPWORDS), "Production Artifact (.pkl + .csv)"
            except Exception:
                pass

        korpus = (
            df["judul"] + " " + df["judul"] + " " +
            df["program_studi"] + " " + df["program_studi"] + " " +
            df["fakultas"] + " " + df["metode_utama"] + " " + df["abstrak"]
        ).apply(clean_indonesian_text)
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3500, sublinear_tf=True)
        mat = vec.fit_transform(korpus)
        return df, vec, mat, DEFAULT_STOPWORDS, "Verified EPrints UMPO Corpus"

    fallback_rows = [
        {"judul": "Implementasi Algoritma YOLOv8 untuk Deteksi Alat Pelindung Diri (APD) Pekerja Konstruksi Secara Real-Time", "abstrak": "Penelitian ini mengimplementasikan arsitektur deep learning YOLOv8 untuk mendeteksi kepatuhan penggunaan helm keselamatan dan rompi reflektif pada pekerja konstruksi.", "fakultas": "Fakultas Teknik", "program_studi": "Teknik Informatika S1", "rumpun_minat": "Computer Vision & AI", "metode_utama": "YOLOv8 Object Detection", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12501/"},
        {"judul": "Klasifikasi Penyakit Daun Tanaman Padi Menggunakan Convolutional Neural Network Arsitektur MobileNetV2", "abstrak": "Studi ini merancang model klasifikasi citra penyakit daun padi (bacterial leaf blight, brown spot, leaf smut) menggunakan transfer learning MobileNetV2.", "fakultas": "Fakultas Teknik", "program_studi": "Teknik Informatika S1", "rumpun_minat": "Computer Vision & AI", "metode_utama": "CNN MobileNetV2", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12507/"},
        {"judul": "Sistem Pendukung Keputusan Pemilihan Judul Skripsi Mahasiswa Menggunakan Metode TF-IDF dan Cosine Similarity", "abstrak": "Sistem cerdas ini menerapkan pembobotan TF-IDF Unigram-Bigram dan Cosine Similarity pada katalog EPrints Universitas Muhammadiyah Ponorogo.", "fakultas": "Fakultas Teknik", "program_studi": "Teknik Informatika S1", "rumpun_minat": "Natural Language Processing & IR", "metode_utama": "TF-IDF & Cosine Similarity", "tahun": "2025", "url": "https://eprints.umpo.ac.id/12513/"},
        {"judul": "Analisis Sentimen Ulasan Pengguna Aplikasi Layanan Publik Pemerintah Menggunakan Algoritma Support Vector Machine", "abstrak": "Penelitian ini mengevaluasi opini masyarakat terhadap layanan digital pemerintah menggunakan pembobotan TF-IDF dan klasifikasi Support Vector Machine.", "fakultas": "Fakultas Teknik", "program_studi": "Teknik Informatika S1", "rumpun_minat": "Natural Language Processing & IR", "metode_utama": "Support Vector Machine (SVM)", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12519/"},
        {"judul": "Rancang Bangun Sistem Monitoring Konsumsi Daya Listrik Tiga Fasa Berbasis IoT Menggunakan Sensor PZEM-004T dan ESP32", "abstrak": "Penelitian ini merancang prototipe pemantauan tegangan, arus, faktor daya, dan konsumsi energi listrik tiga fasa secara real-time berbasis IoT.", "fakultas": "Fakultas Teknik", "program_studi": "Teknik Elektro S1", "rumpun_minat": "Sistem Kendali & IoT Daya", "metode_utama": "IoT ESP32 & Sensor PZEM-004T", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12545/"},
        {"judul": "Optimasi Parameter Kuat Arus dan Sudut Kampuh Pengelasan SMAW terhadap Kekuatan Tarik Baja Karbon Rendah ST 37", "abstrak": "Studi eksperimental ini menguji pengaruh variasi arus pengelasan SMAW terhadap kekuatan tarik dan struktur mikro sambungan las baja ST 37.", "fakultas": "Fakultas Teknik", "program_studi": "Teknik Mesin S1", "rumpun_minat": "Manufaktur & Metalurgi Las", "metode_utama": "Eksperimen Uji Tarik & Metalografi", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12572/"},
        {"judul": "Pengaruh Digital Marketing, Brand Image, dan Electronic Word of Mouth terhadap Keputusan Pembelian Produk UMKM Lokal", "abstrak": "Penelitian kuantitatif ini menganalisis dampak pemasaran digital dan citra merek terhadap keputusan pembelian konsumen UMKM di Kabupaten Ponorogo.", "fakultas": "Fakultas Ekonomi dan Bisnis", "program_studi": "Manajemen S1", "rumpun_minat": "Digital Marketing UMKM & SDM", "metode_utama": "Regresi Linier Berganda", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12597/"},
        {"judul": "Analisis Rasio Likuiditas, Solvabilitas, dan Profitabilitas untuk Memprediksi Kondisi Financial Distress pada Perusahaan Manufaktur", "abstrak": "Penelitian ini menguji akurasi model prediksi kebangkrutan Altman Z-Score dan regresi logistik pada laporan keuangan perusahaan manufaktur.", "fakultas": "Fakultas Ekonomi dan Bisnis", "program_studi": "Akuntansi S1", "rumpun_minat": "Sistem Informasi Akuntansi & Keuangan", "metode_utama": "Regresi Logistik & Altman Z-Score", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12609/"},
        {"judul": "Hubungan Tingkat Stres Akademik dan Kualitas Tidur dengan Kejadian Gastritis pada Mahasiswa Tingkat Akhir", "abstrak": "Studi klinis potong lintang ini menganalisis korelasi antara stres menyusun skripsi dan pola tidur terhadap kekambuhan gastritis mahasiswa.", "fakultas": "Fakultas Ilmu Kesehatan", "program_studi": "Keperawatan S1", "rumpun_minat": "Keperawatan Klinis & Komunitas", "metode_utama": "Uji Korelasi Chi-Square", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12621/"},
        {"judul": "Efektivitas Edukasi Gizi Seimbang Berbasis Media Booklet terhadap Pengetahuan Ibu tentang Pencegahan Stunting pada Balita", "abstrak": "Penelitian kuasi-eksperimen ini mengukur peningkatan literasi gizi ibu balita dalam upaya penurunan prevalensi stunting di wilayah puskesmas.", "fakultas": "Fakultas Ilmu Kesehatan", "program_studi": "Kebidanan D3", "rumpun_minat": "Kesehatan Ibu-Anak & Stunting", "metode_utama": "Quasi-Experiment Pre-Post Test", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12633/"},
        {"judul": "Implementasi Pendidikan Karakter Religius dan Adab Digital Melalui Pembiasaan Ibadah di Madrasah Aliyah", "abstrak": "Penelitian kualitatif lapangan ini mendeskripsikan strategi pembentukan karakter religius dan etika bermedia sosial santri.", "fakultas": "Fakultas Agama Islam", "program_studi": "Pendidikan Agama Islam S1", "rumpun_minat": "Pendidikan Karakter & Literasi Islam", "metode_utama": "Kualitatif Deskriptif Lapangan", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12645/"},
        {"judul": "Inovasi Pelayanan Publik Berbasis Aplikasi Smart Village dalam Meningkatkan Transparansi Administrasi Pemerintahan Desa", "abstrak": "Penelitian ini mengevaluasi penerapan e-government desa melalui platform Smart Village untuk mempercepat layanan kependudukan.", "fakultas": "FISIP", "program_studi": "Ilmu Pemerintahan & Komunikasi S1", "rumpun_minat": "E-Government & Kebijakan Publik", "metode_utama": "Evaluasi Kebijakan & Studi Kasus", "tahun": "2024", "url": "https://eprints.umpo.ac.id/12658/"}
    ]
    df = pd.DataFrame(fallback_rows)
    korpus = (df["judul"] + " " + df["program_studi"] + " " + df["metode_utama"] + " " + df["abstrak"]).apply(clean_indonesian_text)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(korpus)
    return df, vec, mat, DEFAULT_STOPWORDS, "Built-in Demo Corpus"

df_thesis, vectorizer, tfidf_matrix, stopwords_id, engine_mode = load_smartthesis_engine()

if "bookmarked_titles" not in st.session_state:
    st.session_state["bookmarked_titles"] = []

if "active_query" not in st.session_state:
    st.session_state["active_query"] = "klasifikasi citra penyakit daun tanaman padi menggunakan convolutional neural network"

if "llm_responses" not in st.session_state:
    st.session_state["llm_responses"] = {}

# ==============================================================================
# 5. INTEGRASI HYBRID LLM API (GEMINI / OPENAI / GROQ) + GRACEFUL FALLBACK LOKAL
# ==============================================================================
def get_configured_api_key(provider_name: str) -> str:
    """Mengambil API Key dari input sidebar, st.secrets, atau environment variable server."""
    user_key = st.session_state.get("user_llm_api_key", "").strip()
    if user_key:
        return user_key
    env_var_map = {
        "Google Gemini API": "GEMINI_API_KEY",
        "OpenAI / Groq Compatible": "OPENAI_API_KEY"
    }
    env_name = env_var_map.get(provider_name, "GEMINI_API_KEY")
    try:
        if env_name in st.secrets:
            return str(st.secrets[env_name]).strip()
    except Exception:
        pass
    return os.environ.get(env_name, "").strip()

def call_hybrid_llm(prompt_text: str, provider: str, model_name: str, api_key: str) -> tuple[bool, str]:
    """
    Memanggil REST API LLM (Google Gemini atau OpenAI/Groq) secara ringan menggunakan urllib
    tanpa membebani server dengan SDK eksternal yang berat.
    Jika API Key kosong atau terjadi kendala jaringan, mengembalikan (False, pesan_error)
    agar sistem otomatis beralih ke Mesin Sintesis Akademik Lokal (Graceful Fallback).
    """
    if not api_key:
        return False, "API Key belum diisi (Mode Lokal Aktif)."

    try:
        if provider == "Google Gemini API":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": 0.35, "maxOutputTokens": 1200}
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                candidates = res_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return True, parts[0].get("text", "")
            return False, "Respons Gemini kosong."

        else:  # OpenAI / Groq Compatible Endpoint
            base_url = "https://api.groq.com/openai/v1/chat/completions" if "llama" in model_name.lower() or "mixtral" in model_name.lower() else "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Anda adalah Profesor dan Dosen Pembimbing Skripsi ahli metodologi riset lintas disiplin di Universitas Muhammadiyah Ponorogo."},
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.35,
                "max_tokens": 1200
            }
            req = urllib.request.Request(
                base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                choices = res_json.get("choices", [])
                if choices:
                    return True, choices[0].get("message", {}).get("content", "")
            return False, "Respons API kosong."

    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}: Periksa validitas API Key atau kuota Anda."
    except Exception as e:
        return False, f"Kendala koneksi API ({str(e)})."

def build_local_academic_synthesis(user_query: str, top_df: pd.DataFrame, mode: str = "rag_synthesis") -> str:
    """
    Mesin Sintesis Akademik Lokal (Deterministic NLP Fallback) apabila API Key LLM tidak diisi.
    Memastikan aplikasi tetap menghasilkan draf proposal dan analisis celah riset yang kaya & kontekstual.
    """
    if len(top_df) == 0:
        return "Tidak ada dokumen acuan untuk dianalisis."

    r1 = top_df.iloc[0]
    prodi = r1["program_studi"]
    fakultas = r1["fakultas"]
    metode_eksisting = r1["metode_utama"]
    judul_1 = r1["judul"]
    sim_1 = float(r1["similarity_score"]) * 100.0

    blueprint = generate_novelty_blueprint(r1, float(r1["similarity_score"]))
    metode_usulan = blueprint["metode_baru"]
    fokus_gap = blueprint["fokus_gap"]

    ref_list_md = ""
    for idx_r, (_, r_row) in enumerate(top_df.head(3).iterrows(), 1):
        ref_list_md += f"- **Referensi Alumni #{idx_r} ({r_row['tahun']} — {r_row['program_studi']}):** *{r_row['judul']}* (Metode: `{r_row['metode_utama']}`, Kemiripan: `{float(r_row['similarity_score'])*100:.1f}%`)\n"


    if mode == "card_outline":
        return f"""
#### 📑 Kerangka Proposal Skripsi (Bab I – Bab III)
**Berdasarkan Pengembangan dari Referensi:** *{judul_1}*

1. **Rekomendasi Judul Baru yang Bebas Duplikasi:**
   > **{blueprint['saran_judul']}**

2. **Latar Belakang Masalah (Bab I):**
   - Penelitian terdahulu pada repositori EPrints UMPO oleh alumni **{prodi}** ({r1['tahun']}) telah mengkaji topik serupa menggunakan metode **{metode_eksisting}**.
   - Akan tetapi, seiring berkembangnya kompleksitas data dan kebutuhan akurasi pada tahun 2026, diperlukan pendekatan **{metode_usulan}** dengan menitikberatkan pada {fokus_gap}.

3. **Rumusan Masalah:**
   - Bagaimana merancang dan mengimplementasikan metode **{metode_usulan}** pada permasalahan *{user_query}*?
   - Bagaimana perbandingan performa dan efektivitas metode **{metode_usulan}** dibandingkan metode terdahulu (**{metode_eksisting}**)?

4. **Rancangan Metodologi (Bab III):**
   - **Metode Pengumpulan Data:** Observasi primer objek studi kasus 2026 & komparasi pustaka EPrints UMPO.
   - **Metode Analisis / Algoritma:** {metode_usulan}.
   - ** Evaluasi Validasi:** Pengujian metrik akurasi kuantitatif dan uji signifikansi hasil.
"""

    return f"""
#### 🧠 Hasil Sintesis Celah Riset (RAG-Lite Multi-Dokumen)
**Topik Minat Mahasiswa:** *"{user_query}"*  |  **Ranah Keilmuan:** `{fakultas} — {prodi}`

##### 1. Pemetaan Literatur Alumni Terdekat di EPrints UMPO:
{ref_list_md}

##### 2. Identifikasi Celah Penelitian (*Research Gap*):
- **Keterbatasan Metodologis Terdahulu:** Mayoritas penelitian terdahulu pada topik ini masih mengandalkan **{metode_eksisting}** (kemiripan tertinggi `{sim_1:.1f}%`), yang belum mengintegrasikan skema validasi komparatif lintas parameter atau optimasi fitur modern.
- **Peluang Kebaruan (*Novelty* 2026):** Anda dapat mengisi celah riset dengan menerapkan **{metode_usulan}** yang difokuskan pada **{fokus_gap}**.

##### 3. Tiga Rekomendasi Judul Skripsi Baru Siap Ajukan:
1. **Opsi Fokus Algoritma/Metode:** *"Implementasi dan Analisis Komparatif {metode_usulan} terhadap {metode_eksisting} pada Studi {judul_1[:55]}..."*
2. **Opsi Fokus Objek Baru (2026):** *"Pengembangan Model {metode_usulan} dengan Optimasi Parameter untuk Meningkatkan Akurasi pada Kasus {user_query.title()}"*
3. **Opsi Lintas Disiplin:** *"Integrasi Pendekatan {metode_usulan} Berbasis Data Primer Tahun 2026 pada Ranah {prodi}"*
"""

# ==============================================================================
# 6. FUNGSI INTI: EXPLAINABLE IR, HIGHLIGHTING & GENERATOR IDE NOVELTY
# ==============================================================================
def extract_matching_ngrams(query_clean: str, doc_idx: int, top_n: int = 6):
    """Mengekstrak fitur kata/frasa N-Gram yang memiliki kontribusi bobot tertinggi antara kueri dan dokumen."""
    try:
        q_vec = vectorizer.transform([query_clean]).toarray().flatten()
        d_vec = tfidf_matrix[doc_idx].toarray().flatten()
        contrib = q_vec * d_vec
        nonzero_idx = np.where(contrib > 0)[0]
        if len(nonzero_idx) == 0:
            return []
        sorted_idx = nonzero_idx[np.argsort(contrib[nonzero_idx])[::-1][:top_n]]
        feature_names = vectorizer.get_feature_names_out()
        return [(feature_names[i], round(float(contrib[i]), 4)) for i in sorted_idx]
    except Exception:
        return []

def highlight_query_terms(text: str, query_clean: str) -> str:
    """Menyoroti kata kunci kueri pada judul/abstrak secara atomik aman (satu kali substitusi, teks di-escape)."""
    safe = html_lib.escape(str(text), quote=False)
    tokens = [t for t in query_clean.split() if len(t) > 3 and t.lower() not in {"span", "class", "term"}]
    if not tokens:
        return safe
    sorted_tokens = sorted(set(tokens), key=len, reverse=True)[:8]
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) + r"\w*" for t in sorted_tokens) + r")\b", re.IGNORECASE)
    return pattern.sub(r'<span class="term-highlight">\1</span>', safe)

def generate_novelty_blueprint(row, sim_score: float):
    """Menghasilkan cetak biru ide judul baru (Research Gap / Novelty) secara kontekstual per program studi."""
    prodi = str(row["program_studi"])
    judul_lama = str(row["judul"])

    upgrade_map = {
        "Teknik Informatika S1": ("Ensemble Deep Learning / Vision Transformer / IndoBERT", "integrasi deployment edge-computing atau pengujian ketahanan lintas dataset"),
        "Teknik Elektro S1": ("Kendali Prediktif Berbasis AIoT & MQTT Cloud Telemetry", "optimasi efisiensi konsumsi daya real-time dengan sistem peringatan dini"),
        "Teknik Mesin S1": ("Optimasi Taguchi-ANOVA & Simulasi Elemen Hingga (FEA)", "karakterisasi keausan material komposit ramah lingkungan"),
        "Manajemen S1": ("Structural Equation Modeling (SEM-PLS) dengan Variabel Moderasi", "perluasan sampel pada UMKM kreatif pasca-adopsi pembayaran digital QRIS"),
        "Akuntansi S1": ("Machine Learning Fraud Detection & Analisis Beneish M-Score", "komparasi laporan keuangan multi-periode pada sektor industri strategis"),
        "Keperawatan S1": ("Intervensi Keperawatan Berbasis Bukti (EBN) dengan Desain Longitudinal", "pengukuran biomarker klinis dan pendampingan keluarga terstruktur"),
        "Kebidanan D3": ("Skrining Antropometri Digital & Edukasi Interaktif Berbasis Mobile", "pemantauan tumbuh kembang 1000 Hari Pertama Kehidupan (HPK)"),
        "Pendidikan Agama Islam S1": ("Blended Islamic Character Learning & Evaluasi Afektif Berkelanjutan", "penguatan literasi adab digital santri/siswa pada era kecerdasan buatan"),
        "Ilmu Pemerintahan & Komunikasi S1": ("Governance Collaborative Model & Indeks Kepuasan Masyarakat Digital", "evaluasi transparansi anggaran dan partisipasi warga berbasis e-musrenbang")
    }
    metode_baru, fokus_gap = upgrade_map.get(prodi, ("Metode Hibrida Komparatif Terbaru", "perluasan variabel dan objek studi kasus tahun 2026"))

    status_duplikasi = (
        "⚠️ **Kemiripan Tinggi (>50%) — Wajib Modifikasi Objek & Metode:** Ide Anda sangat dekat dengan skripsi alumni ini. Hindari menggunakan objek dan metode yang persis sama."
        if sim_score >= 0.50 else
        "✅ **Zona Aman Pengembangan (<50%) — Peluang Kebaruan Terbuka:** Topik ini memiliki landasan referensi yang kuat namun masih menyisakan ruang inovasi metodologis."
    )

    return {
        "status": status_duplikasi,
        "metode_baru": metode_baru,
        "fokus_gap": fokus_gap,
        "saran_judul": f"Optimasi dan Analisis Komparatif {metode_baru} pada {judul_lama[:65]} (Studi Kasus Tahun 2026)"
    }

# ==============================================================================
# 7. SIDEBAR: FILTER AKADEMIK (UTAMA) → DAFTAR BACAAN → KONFIGURASI AI (OPSIONAL)
# ==============================================================================
with st.sidebar:
    st.markdown("### SmartThesis UMPO")
    st.caption("Hybrid TF-IDF Retriever & AI Research Assistant")

    st.markdown("#### Filter Cakupan Akademik")
    fakultas_opsi = ["Semua Fakultas (Lintas Disiplin)"] + sorted(df_thesis["fakultas"].unique().tolist())
    selected_fakultas = st.selectbox("Fakultas:", fakultas_opsi)

    if selected_fakultas == "Semua Fakultas (Lintas Disiplin)":
        prodi_opsi = ["Semua Program Studi"] + sorted(df_thesis["program_studi"].unique().tolist())
    else:
        sub_p = df_thesis[df_thesis["fakultas"] == selected_fakultas]["program_studi"].unique().tolist()
        prodi_opsi = ["Semua Program Studi"] + sorted(sub_p)

    selected_prodi = st.selectbox("Program Studi:", prodi_opsi)

    tahun_opsi = ["Semua Tahun"] + sorted(df_thesis["tahun"].astype(str).unique().tolist(), reverse=True)
    selected_tahun = st.selectbox("Tahun Publikasi:", tahun_opsi)

    with st.expander(":material/tune: Pengaturan lanjutan", expanded=False):
        top_k = st.slider("Jumlah rekomendasi (Top-K):", min_value=3, max_value=15, value=5, step=1)
        min_threshold = st.slider("Ambang batas Cosine Similarity:", min_value=0.01, max_value=0.60, value=0.08, step=0.02)
        enable_highlight = st.toggle("Soroti kata kunci", value=True)

    st.markdown(f"#### Daftar Bacaan ({len(st.session_state['bookmarked_titles'])})")
    if len(st.session_state["bookmarked_titles"]) == 0:
        st.caption("Belum ada judul tersimpan. Klik **Simpan** pada kartu hasil.")
    else:
        df_saved = pd.DataFrame(st.session_state["bookmarked_titles"])
        for idx_s, s_row in df_saved.iterrows():
            st.markdown(f"- **{s_row['judul'][:50]}...** (`{s_row['kemiripan']}`)")
        csv_saved = df_saved.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Unduh Daftar Referensi (.CSV)",
            data=csv_saved,
            file_name="Daftar_Referensi_Skripsi_SmartThesis_UMPO.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True
        )
        if st.button("Kosongkan Daftar", icon=":material/delete:", use_container_width=True):
            st.session_state["bookmarked_titles"] = []
            st.rerun()

    st.divider()
    # Panel API LLM: jarang diubah, jadi default tertutup
    with st.expander(":material/smart_toy: Konfigurasi AI LLM (Opsional)", expanded=False):
        llm_provider = st.selectbox(
            "Provider LLM:",
            ["Google Gemini API", "OpenAI / Groq Compatible"],
            help="Mesin utama tetap TF-IDF & Cosine Similarity (~1.82 ms). API LLM dipakai untuk sintesis Research Gap & draf Bab I-III."
        )
        if llm_provider == "Google Gemini API":
            llm_model = st.selectbox("Model Gemini:", ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"])
        else:
            llm_model = st.selectbox("Model OpenAI/Groq:", ["gpt-4o-mini", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"])

        st.text_input(
            "API Key (kosongkan untuk mode lokal):",
            type="password",
            key="user_llm_api_key",
            placeholder="AQ... atau AIzaSy... atau sk-..."
        )
        active_api_key = get_configured_api_key(llm_provider)
        if active_api_key:
            st.success(f"Terhubung ke {llm_provider} ({llm_model})", icon=":material/check_circle:")
        else:
            st.info("Mode sintesis NLP lokal aktif (tanpa kuota API)", icon=":material/memory:")

    with st.expander(":material/groups: Tim Pengembang (TI UMPO)", expanded=False):
        st.markdown(
            "1. **Prasetya Riski Wa'afan** (`23533743`)\n"
            "2. **Agya Gilar Cahyono** (`23533753`)\n"
            "3. **Muhammad Taufik Hidayat** (`23533739`)\n"
            "4. **Galang Finto Anergi** (`23533835`)\n"
            "5. **Muhammad Nur Arifin** (`23533792`)\n\n"
            "[prasetyarizki.my.id](https://prasetyarizki.my.id)"
        )

# ==============================================================================
# 8. HEADER RINGKAS & STRIP KPI KOMPAK
# ==============================================================================
st.markdown(f"""
<div class="app-hero">
    <div class="app-eyebrow">SmartThesis UMPO</div>
    <h1 class="app-title">Pemilihan <em>Judul Skripsi</em> Lintas Prodi</h1>
    <p class="app-sub">
        Cari referensi skripsi alumni UMPO, cek kemiripan judul, dan rumuskan celah penelitian dengan bantuan asisten AI.
    </p>
    <div class="app-stats">
        <span><b>{len(df_thesis)}</b> dokumen</span><span class="sep">•</span>
        <span><b>{df_thesis['program_studi'].nunique()}</b> program studi</span><span class="sep">•</span>
        <span><b>{df_thesis['fakultas'].nunique()}</b> fakultas</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 9. EMPAT TAB INTERAKTIF UTAMA
# ==============================================================================
tab_discover, tab_novelty, tab_analytics, tab_eval = st.tabs([
    ":material/search: Pencarian & AI",
    ":material/biotech: Simulator Kebaruan",
    ":material/bar_chart: Analitik Korpus",
    ":material/fact_check: Evaluasi & Matriks"
])

# ------------------------------------------------------------------------------
# TAB 1: PENCARIAN SEMANTIK & REKOMENDASI JUDUL
# ==============================================================================
with tab_discover:
    col_input, col_action = st.columns([4, 1.3])
    with col_input:
        user_query = st.text_input(
            "Kata kunci minat, metode, atau masalah skripsi:",
            value=st.session_state["active_query"],
            placeholder="Ketik topik, metode, atau masalah skripsi Anda...",
            label_visibility="collapsed"
        )
    with col_action:
        sort_mode = st.selectbox(
            "Urutkan:",
            ["Kemiripan Tertinggi", "Tahun Terbaru"],
            label_visibility="collapsed"
        )

    st.caption("Coba topik cepat:")
    presets = [
        ("Visi Komputer & AI", "deteksi klasifikasi citra penyakit daun tanaman padi dan apd menggunakan cnn deep learning"),
        ("NLP & Temu Balik", "analisis sentimen ulasan aplikasi layanan publik pemerintah menggunakan support vector machine tf-idf"),
        ("IoT & Daya", "monitoring konsumsi daya listrik tiga fasa berbasis iot sensor pzem-004t mikrokontroler esp32"),
        ("Bisnis & UMKM", "pengaruh digital marketing brand image keputusan pembelian umkm dan prediksi financial distress"),
        ("Kesehatan", "hubungan tingkat stres akademik kualitas tidur kejadian gastritis mahasiswa dan pencegahan stunting balita"),
        ("Sosial & PAI", "inovasi pelayanan publik smart village pemerintahan desa dan pendidikan karakter religius"),
    ]
    with st.container(key="presets"):
        for p_idx, (btn_label, preset_q) in enumerate(presets):
            if st.button(btn_label, key=f"preset_{p_idx}"):
                st.session_state["active_query"] = preset_q
                st.rerun()

    if user_query.strip():
        t0 = time.perf_counter()
        q_clean = clean_indonesian_text(user_query, stopwords_id)
        q_vec = vectorizer.transform([q_clean])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        df_results = df_thesis.copy()
        df_results["doc_idx"] = df_results.index
        df_results["similarity_score"] = sims

        if selected_fakultas != "Semua Fakultas (Lintas Disiplin)":
            df_results = df_results[df_results["fakultas"] == selected_fakultas]
        if selected_prodi != "Semua Program Studi":
            df_results = df_results[df_results["program_studi"] == selected_prodi]
        if selected_tahun != "Semua Tahun":
            df_results = df_results[df_results["tahun"].astype(str) == selected_tahun]

        df_filtered = df_results[df_results["similarity_score"] >= min_threshold].copy()
        if sort_mode == "Tahun Terbaru":
            df_filtered = df_filtered.sort_values(by=["tahun", "similarity_score"], ascending=[False, False]).head(top_k)
        else:
            df_filtered = df_filtered.sort_values(by="similarity_score", ascending=False).head(top_k)

        # Chip filter aktif: hanya yang berbeda dari default
        chip_items = []
        if selected_fakultas != "Semua Fakultas (Lintas Disiplin)":
            chip_items.append(selected_fakultas)
        if selected_prodi != "Semua Program Studi":
            chip_items.append(selected_prodi)
        if selected_tahun != "Semua Tahun":
            chip_items.append(selected_tahun)
        if sort_mode != "Kemiripan Tertinggi":
            chip_items.append(sort_mode)
        if chip_items:
            chips_html = "".join(f'<span class="f-chip">{html_lib.escape(str(c))}</span>' for c in chip_items)
            st.markdown(f'<div class="filter-chips"><span class="lbl">Filter aktif:</span>{chips_html}</div>', unsafe_allow_html=True)

        trigger_rag = False
        if len(df_filtered) > 0:
            col_summary, col_rag_btn = st.columns([3, 1.5], vertical_alignment="center")
            with col_summary:
                st.markdown(f"**{len(df_filtered)} rekomendasi** untuk *'{user_query}'*")
            with col_rag_btn:
                trigger_rag = st.button("Sintesis Research Gap (AI)", type="primary", icon=":material/psychology:", use_container_width=True)

        # RAG-Lite: hasil disimpan di session_state agar tidak hilang saat tombol lain ditekan
        rag_key = f"{user_query}|{selected_fakultas}|{selected_prodi}|{selected_tahun}|{top_k}|{min_threshold}|{sort_mode}"
        if trigger_rag and len(df_filtered) > 0:
            with st.status("Menganalisis 3 skripsi acuan & merumuskan celah kebaruan...", expanded=False) as rag_status:
                ref_context = "\n".join([
                    f"{i+1}. Judul: {r['judul']} ({r['program_studi']}, {r['tahun']}) | Metode: {r['metode_utama']} | Kemiripan: {float(r['similarity_score'])*100:.1f}% | Abstrak: {str(r['abstrak'])[:280]}"
                    for i, (_, r) in enumerate(df_filtered.head(3).iterrows())
                ])
                rag_prompt = (
                    f"Mahasiswa mengajukan minat topik skripsi: '{user_query}'.\n"
                    f"Berikut adalah 3 skripsi alumni paling mirip dari repositori EPrints UMPO (hasil pencarian TF-IDF Cosine Similarity):\n"
                    f"{ref_context}\n\n"
                    "Sebagai Dosen Pembimbing Skripsi, berikan analisis terstruktur dalam Bahasa Indonesia yang mencakup:\n"
                    "1. Peta Perbandingan & Keterbatasan 3 Skripsi Alumni tersebut.\n"
                    "2. Celah Penelitian (Research Gap) dari sisi metode maupun objek studi.\n"
                    "3. Tiga (3) Rekomendasi Judul Skripsi Baru (Tahun 2026) yang spesifik, modern, dan bebas duplikasi beserta alasan kebaruannya."
                )
                ok_llm, llm_text = call_hybrid_llm(rag_prompt, llm_provider, llm_model, active_api_key)
                if not ok_llm:
                    llm_text = build_local_academic_synthesis(user_query, df_filtered, mode="rag_synthesis")
                    source_tag = "Disusun oleh Mesin Sintesis NLP Lokal (fallback aktif)"
                else:
                    source_tag = f"Disusun secara generatif oleh {llm_provider} ({llm_model})"
                st.session_state["rag_result"] = {"key": rag_key, "text": llm_text, "tag": source_tag}
                rag_status.update(label="Sintesis research gap selesai", state="complete")

        rag_saved = st.session_state.get("rag_result")
        if rag_saved and rag_saved["key"] == rag_key and len(df_filtered) > 0:
            with st.container(border=True):
                st.caption(rag_saved["tag"])
                st.markdown(rag_saved["text"])

        if len(df_filtered) == 0:
            st.markdown(f"""
            <div class="empty-state-panel">
                <div class="empty-state-icon">{ICON_SEARCH_BIG}</div>
                <div class="empty-state-title">Tidak Ditemukan Skripsi yang Memenuhi Ambang Batas</div>
                <div class="empty-state-desc">
                    Kueri <i>"{html_lib.escape(user_query)}"</i> tidak menghasilkan dokumen di atas ambang Cosine Similarity (<b>{min_threshold:.2f}</b>) pada filter yang dipilih.
                </div>
                <div class="empty-state-hint">
                    <b>Saran:</b> turunkan slider <i>Ambang Batas Cosine Similarity</i> di panel kiri (mis. 0.04) atau pilih <i>Semua Fakultas</i>.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for rank_no, (_, row) in enumerate(df_filtered.iterrows(), start=1):
                sim_val = float(row["similarity_score"])
                sim_pct = sim_val * 100.0
                if sim_pct >= 40.0:
                    badge_cls, badge_txt, level = "sim-badge-high", f"Sangat relevan · {sim_pct:.1f}%", "high"
                elif sim_pct >= 20.0:
                    badge_cls, badge_txt, level = "sim-badge-med", f"Relevan · {sim_pct:.1f}%", "med"
                else:
                    badge_cls, badge_txt, level = "sim-badge-low", f"Terkait · {sim_pct:.1f}%", "low"

                judul_tampil = highlight_query_terms(row["judul"], q_clean) if enable_highlight else html_lib.escape(str(row["judul"]), quote=False)
                abstrak_ringkas = str(row["abstrak"])[:260].rstrip() + ("..." if len(str(row["abstrak"])) > 260 else "")
                abstrak_tampil = highlight_query_terms(abstrak_ringkas, q_clean) if enable_highlight else html_lib.escape(abstrak_ringkas, quote=False)

                matched_terms = extract_matching_ngrams(q_clean, int(row["doc_idx"]), top_n=6)
                terms_html = "".join([f'<span class="term-pill">{html_lib.escape(str(t))} ({w})</span>' for t, w in matched_terms]) if matched_terms else '<span class="term-pill">pencocokan semantik korpus</span>'

                novelty_info = generate_novelty_blueprint(row, sim_val)
                metode_esc = html_lib.escape(str(row["metode_utama"]))

                with st.container(key=f"res_{level}_{rank_no}"):
                    st.markdown(f"""
                    <div class="thesis-card">
                        <div class="card-top">
                            <span class="rank-tag">#{rank_no}</span>
                            <span class="{badge_cls}">{badge_txt}</span>
                        </div>
                        <div class="thesis-title">{judul_tampil}</div>
                        <div class="meta-line">
                            <b>{html_lib.escape(str(row['program_studi']))}</b><span class="sep">•</span>
                            <span>{html_lib.escape(str(row['tahun']))}</span><span class="sep">•</span>
                            <span>{metode_esc}</span>
                        </div>
                        <div class="abstract-text">{abstrak_tampil}</div>
                        <details class="more">
                            <summary>Detail analisis &amp; celah riset</summary>
                            <div class="more-body">
                                <div class="matched-terms-bar"><b>Fitur TF-IDF pemicu:</b><br>{terms_html}</div>
                                <div class="novelty-box">
                                    {novelty_info['status']}<br>
                                    <b>Celah riset &amp; rekomendasi kebaruan (2026):</b> Tingkatkan metode acuan (<i>{metode_esc}</i>) menggunakan <b>{novelty_info['metode_baru']}</b> dengan fokus pada {novelty_info['fokus_gap']}.
                                </div>
                            </div>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)

                    b_col1, b_col2, b_col3 = st.columns([1, 1.2, 1.4])
                    with b_col1:
                        if st.button("Simpan", icon=":material/bookmark_add:", key=f"save_{row['doc_idx']}_{rank_no}", use_container_width=True):
                            entry = {
                                "peringkat": rank_no,
                                "judul": row["judul"],
                                "program_studi": row["program_studi"],
                                "fakultas": row["fakultas"],
                                "metode_utama": row["metode_utama"],
                                "tahun": row["tahun"],
                                "kemiripan": f"{sim_pct:.2f}%",
                                "url": row["url"]
                            }
                            if not any(d["judul"] == row["judul"] for d in st.session_state["bookmarked_titles"]):
                                st.session_state["bookmarked_titles"].append(entry)
                                st.toast(f"Judul #{rank_no} ditambahkan ke Daftar Bacaan.", icon=":material/check_circle:")
                            else:
                                st.toast("Judul ini sudah ada di Daftar Bacaan.", icon=":material/info:")
                    with b_col2:
                        if st.button("Buat proposal", icon=":material/auto_awesome:", key=f"outline_{row['doc_idx']}_{rank_no}", use_container_width=True):
                            prompt_card = (
                                f"Berdasarkan skripsi acuan alumni UMPO berjudul '{row['judul']}' (Prodi: {row['program_studi']}, Metode: {row['metode_utama']}), "
                                f"mahasiswa ingin meneliti topik '{user_query}'. Susunkan: (1) Satu judul baru tahun 2026 yang bebas duplikasi, "
                                "(2) Poin Latar Belakang Masalah Bab I, (3) Tiga Rumusan Masalah, dan (4) Rancangan Metodologi Penelitian Bab III."
                            )
                            with st.spinner("Menyusun kerangka proposal..."):
                                ok_c, txt_c = call_hybrid_llm(prompt_card, llm_provider, llm_model, active_api_key)
                                if not ok_c:
                                    txt_c = build_local_academic_synthesis(user_query, pd.DataFrame([row]), mode="card_outline")
                            st.session_state["llm_responses"][f"card_{row['doc_idx']}"] = txt_c
                    with b_col3:
                        st.link_button("Buka di EPrints", str(row["url"]), icon=":material/open_in_new:", use_container_width=True)

                    if f"card_{row['doc_idx']}" in st.session_state["llm_responses"]:
                        with st.expander(f"Draf kerangka proposal — #{rank_no}", expanded=True):
                            st.markdown(st.session_state["llm_responses"][f"card_{row['doc_idx']}"])

        with st.expander(":material/tune: Detail teknis pencarian", expanded=False):
            st.markdown(
                f"- Token kueri bersih: `{q_clean}`\n"
                f"- Waktu TF-IDF + Cosine Similarity: `{latency_ms:.2f} ms`\n"
                f"- Mode korpus: `{engine_mode}`\n"
                f"- Ambang cosine: `{min_threshold:.2f}` • Top-K: `{top_k}`"
            )

# ------------------------------------------------------------------------------
# TAB 2: SIMULATOR KEBARUAN (RESEARCH GAP) & KONSULTAN PROPOSAL AI
# ------------------------------------------------------------------------------
with tab_novelty:
    st.markdown("#### Uji Orisinalitas Draf Judul & Konsultan Proposal AI")
    st.caption(
        "Masukkan rancangan judul skripsi Anda. Sistem mengaudit kemiripan terhadap seluruh arsip EPrints UMPO "
        "dengan TF-IDF Cosine Similarity, lalu menyusun kerangka proposal Bab I–III."
    )

    draft_title = st.text_area(
        "Draf judul skripsi Anda:",
        value="Implementasi Metode Convolutional Neural Network untuk Klasifikasi Penyakit Daun Tanaman Padi Berbasis Citra Digital",
        height=80
    )
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        custom_obj = st.text_input("Objek / lokasi studi kasus baru:", value="Balai Penyuluhan Pertanian Kabupaten Ponorogo")
    with col_n2:
        custom_method = st.text_input("Algoritma / metode baru yang diusulkan:", value="Hybrid MobileNetV3 dan Vision Transformer (ViT)")

    if "simulator_audit_data" not in st.session_state:
        st.session_state["simulator_audit_data"] = None

    if st.button("Jalankan Audit Duplikasi & Susun Draf Proposal", type="primary", icon=":material/science:"):
        q_draft_clean = clean_indonesian_text(draft_title, stopwords_id)
        q_draft_vec = vectorizer.transform([q_draft_clean])
        draft_sims = cosine_similarity(q_draft_vec, tfidf_matrix).flatten()
        top_idx = int(np.argmax(draft_sims))
        max_sim = float(draft_sims[top_idx])
        closest_row = df_thesis.iloc[top_idx]

        synthesized_title = f"Implementasi dan Analisis Kinerja {custom_method} pada {draft_title[:55]}... (Studi Kasus: {custom_obj})"

        with st.status("Menyusun draf kerangka proposal Bab I – Bab III...", expanded=False) as prop_status:
            proposal_prompt = (
                f"Mahasiswa mengajukan judul baru: '{synthesized_title}'.\n"
                f"Skripsi alumni terdekat sebelumnya adalah: '{closest_row['judul']}' yang memakai metode {closest_row['metode_utama']}.\n"
                "Buatkan draf ringkas Proposal Skripsi yang berisi: (1) Latar Belakang Masalah (mengapa metode baru ini penting), "
                "(2) Rumusan Masalah (3 poin), (3) Batasan Masalah, dan (4) Tahapan Metodologi Penelitian."
            )
            ok_prop, prop_text = call_hybrid_llm(proposal_prompt, llm_provider, llm_model, active_api_key)
            if not ok_prop:
                temp_df = pd.DataFrame([closest_row.to_dict()])
                temp_df["similarity_score"] = [max_sim]
                prop_text = build_local_academic_synthesis(synthesized_title, temp_df, mode="card_outline")
            prop_status.update(label="Draf proposal selesai", state="complete")

        st.session_state["simulator_audit_data"] = {
            "max_sim": max_sim,
            "closest_row": closest_row,
            "custom_method": custom_method,
            "custom_obj": custom_obj,
            "synthesized_title": synthesized_title,
            "prop_text": prop_text
        }

    if st.session_state["simulator_audit_data"] is not None:
        data = st.session_state["simulator_audit_data"]
        max_sim = data["max_sim"]
        closest_row = data["closest_row"]
        synthesized_title = data["synthesized_title"]
        prop_text = data["prop_text"]
        custom_method = data["custom_method"]
        custom_obj = data["custom_obj"]

        with st.container(border=True):
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Skor Kemiripan Tertinggi", f"{max_sim*100:.1f}%")
            c_m2.metric("Indeks Orisinalitas", f"{(1.0 - max_sim)*100:.1f}%")
            status_text = "Perlu revisi diferensiasi" if max_sim >= 0.55 else "Aman / orisinalitas baik"
            c_m3.metric("Status Kelayakan Judul", status_text)

        with st.container(border=True):
            st.markdown("##### Skripsi Alumni Terdekat di EPrints")
            st.markdown(
                f"- **Judul Alumni:** *{closest_row['judul']}*\n"
                f"- **Program Studi & Tahun:** `{closest_row['program_studi']}` ({closest_row['tahun']})\n"
                f"- **Metode Utama Alumni:** `{closest_row['metode_utama']}`\n"
                f"- **Tautan Dokumen Asli:** [{closest_row['url']}]({closest_row['url']})"
            )

        with st.container(border=True):
            st.markdown("##### Usulan Judul Skripsi Baru (Bebas Duplikasi)")
            st.markdown(f"### {synthesized_title}")
            st.markdown(
                f"- **Pembeda metodologis (*methodological gap*):** menggantikan metode *{closest_row['metode_utama']}* dengan **{custom_method}**.\n"
                f"- **Pembeda konteks/objek (*empirical gap*):** pengambilan data primer tahun 2026 di **{custom_obj}**."
            )

        with st.container(border=True):
            st.markdown("##### Draf Kerangka Proposal Siap Konsultasi")
            st.markdown(prop_text)

# ------------------------------------------------------------------------------
# TAB 3: ANALITIK KORPUS LINTAS PROGRAM STUDI
# ------------------------------------------------------------------------------
with tab_analytics:
    st.markdown("#### Distribusi & Statistik Repositori Skripsi")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**Sebaran dokumen per program studi:**")
        st.bar_chart(df_thesis["program_studi"].value_counts(), color="#8A4F1D")

    with col_chart2:
        st.markdown("**Metode penelitian dominan:**")
        st.bar_chart(df_thesis["metode_utama"].value_counts().head(8), color="#F59E0B")

    st.markdown("#### Tabel Penelusuran Korpus")
    search_table_kw = st.text_input("Filter cepat berdasarkan judul, prodi, atau metode:", value="")
    df_display = df_thesis.copy()
    if search_table_kw.strip():
        mask = (
            df_display["judul"].str.contains(search_table_kw, case=False, na=False) |
            df_display["program_studi"].str.contains(search_table_kw, case=False, na=False) |
            df_display["metode_utama"].str.contains(search_table_kw, case=False, na=False)
        )
        df_display = df_display[mask]

    cols_order = [c for c in ["judul", "fakultas", "program_studi", "metode_utama", "tahun", "url"] if c in df_display.columns]
    st.dataframe(df_display[cols_order], use_container_width=True, height=380)

# ------------------------------------------------------------------------------
# TAB 4: EVALUASI ILMIAH, PRECISION@K, MRR & CONFUSION MATRIX
# ------------------------------------------------------------------------------
with tab_eval:
    st.markdown("#### Validasi Kuantitatif Information Retrieval")
    st.caption("Metrik selaras dengan hasil Pipeline 1 di Google Colaboratory dan naskah publikasi Jurnal Almudaya.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision@3", "97,33%", help="Peringkat #1 tepat pada 24 dari 25 kueri uji")
    m2.metric("MRR", "98,0%")
    m3.metric("Akurasi Confusion Matrix", "69,33%", help="Precision 72,09% • F1 69,14% (weighted)")
    m4.metric("Latensi Inferensi", "~1,82 ms")

    col_ev1, col_ev2 = st.columns([1.15, 1])
    with col_ev1:
        st.markdown("##### 1. Laporan Klasifikasi Lintas 8 Program Studi (N = 75 Prediksi Top-3)")
        df_cm_report = pd.DataFrame([
            {"No": "1", "Program Studi Target (Ground Truth)": "Ilmu Pemerintahan & Komunikasi S1", "Support": 3, "TP": 3, "Precision (%)": "75,00%", "Recall (%)": "100,00%", "F1-Score (%)": "85,71%"},
            {"No": "2", "Program Studi Target (Ground Truth)": "Kebidanan D3", "Support": 9, "TP": 3, "Precision (%)": "60,00%", "Recall (%)": "33,33%", "F1-Score (%)": "42,86%"},
            {"No": "3", "Program Studi Target (Ground Truth)": "Keperawatan S1", "Support": 9, "TP": 7, "Precision (%)": "77,78%", "Recall (%)": "77,78%", "F1-Score (%)": "77,78%"},
            {"No": "4", "Program Studi Target (Ground Truth)": "Manajemen S1", "Support": 9, "TP": 4, "Precision (%)": "57,14%", "Recall (%)": "44,44%", "F1-Score (%)": "50,00%"},
            {"No": "5", "Program Studi Target (Ground Truth)": "Pendidikan Agama Islam S1", "Support": 3, "TP": 3, "Precision (%)": "33,33%", "Recall (%)": "100,00%", "F1-Score (%)": "50,00%"},
            {"No": "6", "Program Studi Target (Ground Truth)": "Teknik Elektro S1", "Support": 21, "TP": 16, "Precision (%)": "72,73%", "Recall (%)": "76,19%", "F1-Score (%)": "74,42%"},
            {"No": "7", "Program Studi Target (Ground Truth)": "Teknik Informatika S1", "Support": 15, "TP": 12, "Precision (%)": "80,00%", "Recall (%)": "80,00%", "F1-Score (%)": "80,00%"},
            {"No": "8", "Program Studi Target (Ground Truth)": "Teknik Mesin S1", "Support": 6, "TP": 4, "Precision (%)": "100,00%", "Recall (%)": "66,67%", "F1-Score (%)": "80,00%"},
            {"No": "-", "Program Studi Target (Ground Truth)": "Rata-Rata Makro (Macro Average)", "Support": 75, "TP": 52, "Precision (%)": "69,50%", "Recall (%)": "72,30%", "F1-Score (%)": "67,60%"},
            {"No": "-", "Program Studi Target (Ground Truth)": "Rata-Rata Terbobot (Weighted Avg)", "Support": 75, "TP": 52, "Precision (%)": "72,09%", "Recall (%)": "69,33%", "F1-Score (%)": "69,14%"}
        ])
        st.dataframe(df_cm_report, use_container_width=True, hide_index=True)

        st.markdown("##### 2. Matriks Kontingensi 8×8 (Ground Truth vs Predicted Top-3)")
        prodi_short = ["Ilmu Pem.", "Kebidanan", "Keperawatan", "Manajemen", "PAI", "T. Elektro", "T. Informatika", "T. Mesin"]
        cm_matrix_8x8 = pd.DataFrame(
            [
                [3, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 0, 1, 3, 1, 0, 0],
                [0, 0, 7, 1, 0, 1, 0, 0],
                [0, 0, 1, 4, 2, 2, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0],
                [0, 1, 1, 1, 0, 16, 2, 0],
                [0, 1, 0, 0, 1, 1, 12, 0],
                [0, 0, 0, 0, 0, 1, 1, 4]
            ],
            index=[f"Aktual: {p}" for p in prodi_short],
            columns=[f"Pred: {p}" for p in prodi_short]
        )
        st.dataframe(cm_matrix_8x8, use_container_width=True)

    with col_ev2:
        st.markdown("##### 3. Heatmap Confusion Matrix")
        cm_img_candidates = [
            os.path.join(BASE_DIR, "confusion_matrix_skripsi.png"),
            os.path.join(os.path.dirname(BASE_DIR), "04_testing", "confusion_matrix_skripsi.png"),
            "confusion_matrix_skripsi.png"
        ]
        cm_img_found = next((p for p in cm_img_candidates if os.path.exists(p)), None)
        if cm_img_found:
            st.image(cm_img_found, caption="Gambar 4. Heatmap Confusion Matrix Kesesuaian Rekomendasi Lintas 8 Program Studi (200 Data EPrints UMPO)", use_container_width=True)

        st.info(
            "**Interpretasi akademis off-diagonal:**\n"
            "- **Overall Accuracy (69,33%):** 52 dari 75 rekomendasi Top-3 tepat berada pada diagonal utama program studi target.\n"
            "- **Irisan lintas disiplin (23 prediksi off-diagonal):** mencerminkan keterkaitan riset antarfakultas di UMPO, misalnya topik *mikrokontroler IoT* antara Teknik Elektro & Teknik Informatika, atau *stunting & kesehatan masyarakat* antara Kebidanan & Keperawatan.",
            icon=":material/insights:"
        )