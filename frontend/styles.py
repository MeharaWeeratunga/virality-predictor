"""
CSS styles
"""

def get_css() -> str:
    return """
<style>
/* Root tokens - light theme */
:root {
    --accent:    #1f77b4;
    --accent-dim:#1566a0;
    --viral:     #856404;
    --viral-bg:  #fff3cd;
    --viral-bdr: #ffc107;
    --safe:      #0c5460;
    --safe-bg:   #d1ecf1;
    --safe-bdr:  #17a2b8;
    --text:      #212529;
    --muted:     #6c757d;
    --border:    #dee2e6;
    --surface:   #f8f9fa;
    --radius:    10px;
}

/* Masthead */
.masthead {
    text-align: center;
}
.masthead-title {
    font-size: 3rem !important;
    font-weight: bold;
    color: var(--accent) !important;
    margin: 0;
    line-height: 1.2;
}
.masthead-sub {
    font-size: 0.95rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* Prediction result */
.pred-viral {
    background-color: var(--viral-bg);
    border: 2px solid var(--viral-bdr);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin: 1rem 0;
}
.pred-safe {
    background-color: var(--safe-bg);
    border: 2px solid var(--safe-bdr);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin: 1rem 0;
}
.pred-icon  { font-size: 2.5rem; line-height: 1; flex-shrink: 0; }
.pred-label { font-weight: 600; margin: 0; line-height: 1; }
.pred-viral .pred-label { color: var(--viral); font-size: 2.5rem; }
.pred-safe  .pred-label { color: var(--safe); font-size: 2.5rem; }
.pred-sub   { font-size: 0.85rem; color: var(--muted); margin-top: 0.3rem; }

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.45rem 0;
}
.prob-label {
    font-size: 0.8rem;
    color: var(--muted);
    width: 160px;
    flex-shrink: 0;
    text-align: right;
}
.prob-track {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
}
.prob-fill-viral { background: #ffc107; height: 100%; border-radius: 4px; }
.prob-fill-safe  { background: #17a2b8; height: 100%; border-radius: 4px; }
.prob-fill-mid   { background: #fd7e14; height: 100%; border-radius: 4px; }
.prob-val {
    font-size: 0.82rem;
    font-weight: 600;
    width: 42px;
    text-align: right;
    color: var(--text);
    flex-shrink: 0;
}

/* Feature chip grid */
.chip-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
    gap: 0.6rem;
    margin-top: 0.6rem;
}
.chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
}
.chip-name {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.15rem;
}
.chip-val           { font-size: 1rem; font-weight: 600; color: var(--text); }
.chip-val.highlight { color: var(--accent); }

/* Metric row */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin: 0.8rem 0;
}
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-val  { font-size: 1.5rem; font-weight: bold; color: var(--accent); line-height: 1; }
.metric-name { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }

/* Card */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Section heading */
.section-head {
    font-size: 1.15rem;
    font-weight: 600;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.35rem;
    margin: 1.5rem 0 0.8rem;
}
</style>
"""