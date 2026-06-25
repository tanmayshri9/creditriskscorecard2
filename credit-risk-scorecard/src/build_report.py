"""
build_report.py
===============
Assembles a single, self-contained HTML case-study report from results.json and
the generated figures (base64-embedded so the file is fully portable / shareable).

Run AFTER run_pipeline.py:  python src/build_report.py
"""

import os
import json
import base64

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs")
FIG = os.path.join(OUT, "figures")

with open(os.path.join(OUT, "results.json")) as f:
    R = json.load(f)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def img(name):
    with open(os.path.join(FIG, name), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f"data:image/png;base64,{b64}"


def figure(name, caption, width="100%"):
    return f"""
    <figure class="fig">
      <img src="{img(name)}" alt="{caption}" style="width:{width}"/>
      <figcaption>{caption}</figcaption>
    </figure>"""


def table(records, columns, headers=None, highlight_col=None, fmt=None):
    headers = headers or columns
    fmt = fmt or {}
    head = "".join(f"<th>{h}</th>" for h in headers)
    rows = []
    for rec in records:
        cells = []
        for c in columns:
            v = rec.get(c, "")
            if c in fmt and isinstance(v, (int, float)):
                v = fmt[c](v)
            cls = ' class="hl"' if c == highlight_col else ""
            cells.append(f"<td{cls}>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""<div class="tablewrap"><table>
      <thead><tr>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>"""


def pct(x):  return f"{x*100:.1f}%"
def pct2(x): return f"{x*100:.2f}%"
def f3(x):   return f"{x:.3f}"
def f4(x):   return f"{x:.4f}"
def money(x): return f"${x:,.0f}"
def comma(x): return f"{int(x):,}"


# --------------------------------------------------------------------------- #
# CSS  (plain string — not an f-string)
# --------------------------------------------------------------------------- #
CSS = """
:root{
  --ink:#0B1F33; --ink2:#1C3349; --soft:#52606D; --faint:#8A96A3;
  --paper:#F5F6F8; --card:#FFFFFF; --line:#E1E5EA;
  --teal:#127475; --teal-d:#0C5253; --good:#1B7A43; --bad:#C23B22; --amber:#C68A12;
  --grad:linear-gradient(90deg,#C23B22 0%,#D98324 28%,#C68A12 50%,#6FA63B 74%,#1B7A43 100%);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.mono{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;}
h1,h2,h3,.disp,.eyebrow,nav,.stat-v,.tag,th,.btn{font-family:"Space Grotesk",system-ui,sans-serif;}

/* ---------- layout shell ---------- */
.wrap{max-width:1180px;margin:0 auto;padding:0 28px;}
.grid{display:grid;grid-template-columns:230px 1fr;gap:56px;align-items:start;}
@media(max-width:900px){.grid{grid-template-columns:1fr;gap:0}}

/* ---------- masthead ---------- */
.mast{background:var(--ink);color:#EAF0F4;padding:54px 0 0;position:relative;overflow:hidden;}
.mast .wrap{position:relative;z-index:2;}
.mast .kicker{font-family:"Space Grotesk";letter-spacing:.26em;text-transform:uppercase;
  font-size:12px;color:#7FB2B2;font-weight:600;}
.mast h1{font-size:48px;line-height:1.04;margin:14px 0 10px;font-weight:600;letter-spacing:-.01em;color:#fff;}
.mast h1 em{font-style:normal;color:#E7B65A;}
.mast .sub{font-size:18px;color:#AEBDC9;max-width:760px;margin:0 0 30px;}
.gradbar{height:7px;background:var(--grad);width:100%;}
.gradbar.thin{height:3px;}
.mast-meta{display:flex;flex-wrap:wrap;gap:26px;padding:20px 0 30px;font-family:"Space Grotesk";font-size:13px;color:#9DB0BD;}
.mast-meta b{color:#EAF0F4;font-weight:600;}
.mast-deco{position:absolute;right:-60px;top:-40px;width:520px;height:520px;opacity:.10;z-index:1;}

/* ---------- exec stat strip ---------- */
.execstrip{background:var(--ink2);border-top:1px solid #2A4considerable;}
.statrow{display:grid;grid-template-columns:repeat(5,1fr);gap:0;}
@media(max-width:760px){.statrow{grid-template-columns:repeat(2,1fr)}}
.stat{padding:22px 20px;border-right:1px solid rgba(255,255,255,.08);}
.stat:last-child{border-right:none;}
.stat-v{font-size:30px;font-weight:600;color:#fff;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
.stat-v .u{font-size:15px;color:#7FB2B2;margin-left:3px;}
.stat-l{font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;color:#90A2B0;margin-top:5px;font-family:"Space Grotesk";}

/* ---------- nav rail ---------- */
nav{position:sticky;top:24px;font-size:13.5px;}
nav .navttl{font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--faint);margin:0 0 14px;font-weight:600;}
nav a{display:flex;gap:11px;align-items:baseline;color:var(--soft);text-decoration:none;padding:7px 0;
  border-left:2px solid transparent;padding-left:12px;margin-left:-14px;transition:.15s;}
nav a .n{color:var(--teal);font-weight:600;font-size:12px;min-width:20px;}
nav a:hover{color:var(--ink);border-left-color:var(--teal);}
nav .navfoot{margin-top:22px;padding-top:18px;border-top:1px solid var(--line);font-size:12px;color:var(--faint);font-family:"Source Serif 4";}

/* ---------- sections ---------- */
main{padding:64px 0 40px;min-width:0;}
section{margin-bottom:76px;scroll-margin-top:24px;}
.eyebrow{display:inline-block;font-size:11.5px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--teal-d);font-weight:600;margin-bottom:10px;}
.snum{font-family:"Space Grotesk";font-size:13px;color:var(--amber);font-weight:600;letter-spacing:.05em;}
h2{font-size:30px;line-height:1.12;margin:4px 0 18px;font-weight:600;letter-spacing:-.01em;}
h3{font-size:19px;margin:34px 0 12px;font-weight:600;color:var(--ink2);}
p{margin:0 0 16px;}
.lead{font-size:19px;color:var(--ink2);line-height:1.6;}
strong{font-weight:600;color:var(--ink);}
a.inl{color:var(--teal-d);}

/* ---------- components ---------- */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:22px 0;}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 18px 16px;}
.card .v{font-family:"Space Grotesk";font-size:26px;font-weight:600;color:var(--ink);letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
.card .v .u{font-size:14px;color:var(--teal);}
.card .l{font-size:12px;color:var(--soft);margin-top:4px;line-height:1.35;}
.card.tealtop{border-top:3px solid var(--teal);}
.card.ambertop{border-top:3px solid var(--amber);}
.card.goodtop{border-top:3px solid var(--good);}

.note{background:#fff;border:1px solid var(--line);border-left:3px solid var(--teal);
  border-radius:8px;padding:16px 20px;margin:22px 0;font-size:15.5px;color:var(--ink2);}
.note .nt{font-family:"Space Grotesk";font-size:11px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--teal-d);font-weight:600;display:block;margin-bottom:6px;}
.note.gov{border-left-color:var(--amber);}
.note.gov .nt{color:#9a6a08;}

.fig{margin:26px 0;text-align:center;}
.fig img{max-width:100%;border:1px solid var(--line);border-radius:10px;background:#fff;}
.fig figcaption{font-size:13px;color:var(--soft);margin-top:10px;font-style:italic;}

.tablewrap{overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:10px;}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;}
th{background:var(--ink);color:#EAF0F4;text-align:left;padding:11px 13px;font-weight:500;
  font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap;}
td{padding:9px 13px;border-top:1px solid var(--line);color:var(--ink2);
  font-family:"JetBrains Mono",monospace;font-size:12.5px;font-variant-numeric:tabular-nums;}
tr:nth-child(even) td{background:#FAFBFC;}
td.hl{background:#EAF4F4 !important;color:var(--teal-d);font-weight:600;}

pre{background:var(--ink);color:#D6E0E8;border-radius:10px;padding:18px 20px;overflow-x:auto;
  font-family:"JetBrains Mono",monospace;font-size:12.5px;line-height:1.6;margin:20px 0;}
pre .c{color:#6F8597;} pre .k{color:#E7B65A;} pre .s{color:#8FC9A0;} pre .f{color:#7FB2B2;}
code{font-family:"JetBrains Mono",monospace;font-size:.9em;background:#EDF0F3;padding:1px 6px;border-radius:5px;color:var(--teal-d);}

.formula{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 22px;margin:18px 0;
  font-family:"JetBrains Mono",monospace;font-size:14px;color:var(--ink);text-align:center;}
.two{display:grid;grid-template-columns:1fr 1fr;gap:30px;align-items:center;}
@media(max-width:760px){.two{grid-template-columns:1fr}}
ul.clean{padding-left:0;list-style:none;margin:16px 0;}
ul.clean li{padding:7px 0 7px 26px;position:relative;border-bottom:1px solid var(--line);font-size:15.5px;}
ul.clean li:before{content:"▸";color:var(--teal);position:absolute;left:4px;top:7px;}
ul.clean li:last-child{border-bottom:none;}
.pill{display:inline-block;font-family:"Space Grotesk";font-size:11px;font-weight:600;padding:3px 10px;
  border-radius:20px;letter-spacing:.03em;}
.pill.good{background:#E3F1E8;color:var(--good);}
.pill.amber{background:#FBF0DA;color:#9a6a08;}
.pill.bad{background:#F8E2DD;color:var(--bad);}
.pill.ink{background:#E5EAF0;color:var(--ink2);}

footer{background:var(--ink);color:#9DB0BD;padding:48px 0;font-size:14px;}
footer .wrap{display:flex;justify-content:space-between;flex-wrap:wrap;gap:20px;}
footer b{color:#EAF0F4;font-family:"Space Grotesk";}
footer a{color:#7FB2B2;text-decoration:none;}
.disc{font-size:12.5px;color:#6F8597;max-width:620px;line-height:1.5;}

@media print{nav{display:none}.grid{grid-template-columns:1fr}.mast-deco{display:none}}
"""
# fix accidental tokens
CSS = CSS.replace("#2A4considerable", "#2A4055")


# --------------------------------------------------------------------------- #
# pull key values
# --------------------------------------------------------------------------- #
D   = R["data"]
FS  = R["feature_selection"]
SC  = R["scorecard"]
champ_test = [r for r in R["validation"]["champion"] if "Test" in r["segment"]][0]
champ_oot  = [r for r in R["validation"]["champion"] if "Out" in r["segment"]][0]
champ_tr   = [r for r in R["validation"]["champion"] if "Train" in r["segment"]][0]
MC  = R["validation"]["model_comparison"]
MON = R["monitoring"]
BIZ = R["business"]
PB  = BIZ["profit_best"]
SV  = BIZ["swap_verdict"]

GENERIC_NAME = "Tanmay Shrivastava"   # the candidate replaces this


# --------------------------------------------------------------------------- #
# MASTHEAD + EXEC STRIP
# --------------------------------------------------------------------------- #
DECO_SVG = '''<svg class="mast-deco" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<g fill="none" stroke="#7FB2B2" stroke-width="0.5">
<path d="M5 80 Q30 78 50 60 T95 20"/><path d="M5 88 Q35 60 55 50 T95 35"/>
<circle cx="50" cy="50" r="42"/><circle cx="50" cy="50" r="30"/>
</g></svg>'''

masthead = f"""
<header class="mast">
  {DECO_SVG}
  <div class="wrap">
    <div class="kicker">Credit Risk &nbsp;·&nbsp; Model Development Case Study</div>
    <h1>Probability-of-Default<br>Application <em>Scorecard</em></h1>
    <p class="sub">An end-to-end, regulator-grade consumer-credit scorecard — from raw bureau,
    transactional and alternative data through Weight-of-Evidence engineering, a scaled logistic
    model, champion–challenger benchmarking, and live PSI/CSI monitoring, to the lending-strategy
    P&amp;L it drives.</p>
    <div class="mast-meta">
      <span>Prepared by &nbsp;<b>{GENERIC_NAME}</b></span>
      <span>Domain &nbsp;<b>Retail / Consumer Lending</b></span>
      <span>Stack &nbsp;<b>Python · scikit-learn · XGBoost</b></span>
      <span>Governance &nbsp;<b>SR 11-7 aligned</b></span>
    </div>
  </div>
  <div class="gradbar"></div>
  <div class="execstrip"><div class="wrap"><div class="statrow">
    <div class="stat"><div class="stat-v">{champ_test['ks']*100:.1f}</div><div class="stat-l">KS statistic</div></div>
    <div class="stat"><div class="stat-v">{champ_test['gini']:.3f}</div><div class="stat-l">Gini coefficient</div></div>
    <div class="stat"><div class="stat-v">{champ_test['auc']:.3f}</div><div class="stat-l">AUC-ROC</div></div>
    <div class="stat"><div class="stat-v">{D['n_total']//1000}<span class="u">k</span></div><div class="stat-l">Applications modelled</div></div>
    <div class="stat"><div class="stat-v">{PB['approved_bad_rate']*100:.1f}<span class="u">%</span></div><div class="stat-l">Bad rate at optimal cutoff</div></div>
  </div></div></div>
</header>
"""

# --------------------------------------------------------------------------- #
# NAV
# --------------------------------------------------------------------------- #
NAV_ITEMS = [
    ("01","Business problem","problem"),
    ("02","Data & target","data"),
    ("03","WOE & Information Value","woe"),
    ("04","Feature selection","selection"),
    ("05","Scorecard & scaling","scorecard"),
    ("06","Validation","validation"),
    ("07","Champion–challenger","challenger"),
    ("08","Model monitoring","monitoring"),
    ("09","Lending strategy & P&L","business"),
    ("10","Governance & MRM","governance"),
]
nav = '<nav><div class="navttl">Contents</div>'
for n,label,anchor in NAV_ITEMS:
    nav += f'<a href="#{anchor}"><span class="n">{n}</span><span>{label}</span></a>'
nav += f'<div class="navfoot">A reproducible build — every figure and metric on this page is generated by the accompanying Python pipeline.</div></nav>'

# --------------------------------------------------------------------------- #
# SECTION 01 — PROBLEM
# --------------------------------------------------------------------------- #
s01 = f"""
<section id="problem">
  <span class="eyebrow">Section 01 · Framing</span>
  <h2>The business problem</h2>
  <p class="lead">A consumer lender must decide, at the point of application, whether to extend
  credit — and on what terms. Approve too freely and credit losses erode the book; approve too
  tightly and you forgo good customers and growth. The job of an application scorecard is to
  <strong>rank every applicant by probability of default (PD)</strong> so that a single, defensible
  cut-off can balance growth, profitability and risk.</p>

  <div class="cards">
    <div class="card tealtop"><div class="v">PD</div><div class="l">Estimate the probability an applicant goes 90+ days past due within 12 months</div></div>
    <div class="card tealtop"><div class="v">Rank</div><div class="l">Separate likely goods from likely bads across the score range</div></div>
    <div class="card tealtop"><div class="v">Explain</div><div class="l">Produce adverse-action reason codes that satisfy fair-lending review</div></div>
    <div class="card tealtop"><div class="v">Govern</div><div class="l">Stay monitorable and stable under SR 11-7 model risk management</div></div>
  </div>

  <p>This case study builds that scorecard end to end on a {comma(D['n_total'])}-application
  consumer-lending book, then carries it all the way through to the lending decision it powers. The
  emphasis throughout is the judgment a model owner is paid for — target design, leakage control,
  the interpretability-versus-power trade-off, and post-deployment stability — not just fitting a
  classifier.</p>

  <div class="note"><span class="nt">Why this matters commercially</span>
  A one-point lift in KS, or a half-point reduction in approved-book bad rate, is worth millions on
  a portfolio of any size. The final section translates the model directly into expected portfolio
  P&amp;L and a swap-set against an incumbent policy.</div>
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 02 — DATA & TARGET
# --------------------------------------------------------------------------- #
s02 = f"""
<section id="data">
  <span class="eyebrow">Section 02 · Data foundation</span>
  <h2>Data &amp; target definition</h2>
  <p class="lead">The book combines the three data families a real lender underwrites on:
  application/demographic, credit-bureau, and behavioural/alternative data — the last reflecting the
  digital footprints and alternative sources increasingly used to score thin-file customers.</p>

  <div class="two">
    <div>
      <h3>Feature families</h3>
      <ul class="clean">
        <li><strong>Application</strong> — age, income, employment status &amp; tenure, residence</li>
        <li><strong>Bureau</strong> — bureau score, utilisation, 24-mo delinquencies, 6-mo inquiries, trades, public records, history depth</li>
        <li><strong>Behavioural / alternative</strong> — debt-to-income, requested amount, offered rate, a digital-engagement score</li>
      </ul>
      <p>Targets are defined on a <strong>12-month performance window</strong> following an
      observation point: <span class="pill ink">bad = 90+ DPD</span> within the window,
      <span class="pill ink">good</span> otherwise. Indeterminates would normally be excluded; here
      the definition is clean by construction.</p>
    </div>
    <div>
      <div class="cards" style="grid-template-columns:1fr 1fr">
        <div class="card"><div class="v">{comma(D['n_dev'])}</div><div class="l">Development book (2023&nbsp;H1 vintage)</div></div>
        <div class="card"><div class="v">{comma(D['n_oot'])}</div><div class="l">Out-of-time book (2024&nbsp;H1) for OOT + monitoring</div></div>
        <div class="card ambertop"><div class="v">{pct2(D['dev_bad_rate'])}</div><div class="l">Development bad rate (through-the-door)</div></div>
        <div class="card ambertop"><div class="v">{pct2(D['oot_bad_rate'])}</div><div class="l">Out-of-time bad rate — a deliberate downturn</div></div>
      </div>
    </div>
  </div>

  {figure("01_target_overview.png","Target distribution, the monotone fall in bad rate across bureau-score bands, and the income separation between goods and bads — the signal a model can exploit.")}

  <div class="note"><span class="nt">Design choice · sampling windows</span>
  Development and out-of-time samples are drawn from different vintages so the model is validated on
  a population it never saw, and so the monitoring section has genuine drift to detect rather than a
  re-shuffle of the same data.</div>
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 03 — WOE / IV
# --------------------------------------------------------------------------- #
bins_bureau = R["binning_examples"]["bureau_score"]
woe_code = '''<span class="c"># Weight of Evidence linearises each predictor against the log-odds of default</span>
<span class="k">def</span> <span class="f">weight_of_evidence</span>(bin_goods, bin_bads, tot_goods, tot_bads):
    dist_good = bin_goods / tot_goods
    dist_bad  = bin_bads  / tot_bads
    woe = np.log(dist_good / dist_bad)          <span class="c"># &gt;0 safer, &lt;0 riskier</span>
    iv  = (dist_good - dist_bad) * woe           <span class="c"># bin contribution to IV</span>
    <span class="k">return</span> woe, iv

<span class="c"># Supervised, monotonic binning via a shallow decision tree per feature,</span>
<span class="c"># then greedy merge of any bins that break WOE monotonicity.</span>'''

s03 = f"""
<section id="woe">
  <span class="eyebrow">Section 03 · Feature engineering</span>
  <h2>Weight of Evidence &amp; Information Value</h2>
  <p class="lead">Rather than feed raw predictors to the model, each variable is binned and replaced
  by the <strong>Weight of Evidence (WOE)</strong> of its bin. WOE linearises the relationship
  between predictor and log-odds — exactly what logistic regression assumes — while taming outliers
  and turning missing values into an informative bin.</p>

  <div class="formula">WOE<sub>bin</sub> = ln( %Good<sub>bin</sub> / %Bad<sub>bin</sub> )
  &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  IV = Σ ( %Good − %Bad ) · WOE</div>

  <p>Binning is <strong>supervised and monotonic</strong>: a shallow decision tree proposes split
  points per feature (respecting a minimum 5% bin population), then adjacent bins that violate WOE
  monotonicity are merged. Monotonic WOE is what makes the eventual scorecard explainable and
  reason-code friendly.</p>

  <pre>{woe_code}</pre>

  <p><strong>Information Value</strong> then ranks predictive power. The bureau score dominates
  (IV {R['iv']['bureau_score']:.2f}), followed by revolving utilisation, recent delinquencies and
  debt-to-income — economically exactly what we expect on an unsecured book.</p>

  {figure("02_iv_ranking.png","Information Value by predictor, against the standard weak / medium / strong thresholds.")}

  <h3>Worked binning table — bureau score</h3>
  <p>Every bin's bad rate falls as the score rises and WOE increases monotonically from
  {bins_bureau[0]['woe']:+.2f} to {bins_bureau[-1]['woe']:+.2f} — a clean, defensible characteristic.</p>
  {table(bins_bureau,
         ["bin","n","event_rate","woe","iv"],
         ["Score bin","Count","Bad rate","WOE","IV contrib."],
         highlight_col="woe",
         fmt={"event_rate":pct,"woe":lambda v:f"{v:+.3f}","iv":f4})}

  {figure("03_woe_grid.png","WOE profiles for the nine modelled characteristics. Green bins are safer than average, red riskier — every profile is monotone and economically sensible.")}
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 04 — FEATURE SELECTION
# --------------------------------------------------------------------------- #
vif_rows = [{"f":k,"vif":v} for k,v in FS["final_vif"].items()]
s04 = f"""
<section id="selection">
  <span class="eyebrow">Section 04 · Selection &amp; QC</span>
  <h2>Feature selection</h2>
  <p class="lead">Predictive power is necessary but not sufficient. A production characteristic must
  also be <strong>causally defensible, non-redundant and correctly signed</strong>. Selection runs a
  four-gate funnel.</p>

  <ul class="clean">
    <li><strong>Gate 1 — Information Value band.</strong> Keep predictors with IV in a usable range;
    drop the inert ones (public records, open trades, requested amount).</li>
    <li><strong>Gate 2 — Endogeneity / leakage.</strong> The offered <code>interest_rate</code> has a
    high IV ({R['iv']['interest_rate']:.2f}) — but it is <em>priced off the very risk we are
    predicting</em>. Including it leaks the answer and corrupts reason codes, so it is excluded on
    judgment, not statistics.</li>
    <li><strong>Gate 3 — Multicollinearity.</strong> Variance Inflation Factors are computed on the
    WOE matrix; any feature above a VIF of 5 is dropped iteratively.</li>
    <li><strong>Gate 4 — Coefficient sign.</strong> After fitting, every WOE coefficient must be
    negative (higher WOE → lower PD). A counter-intuitive sign signals confounding and the feature
    is removed.</li>
  </ul>

  <div class="note gov"><span class="nt">Modeler's judgment</span>
  Excluding a high-IV variable like the offered rate is the kind of call that separates a scorecard
  that passes validation from one that fails it six months later. Statistically it looks like the
  second-best predictor; causally it is an outcome, not an input.</div>

  <div class="two">
    <div>
      <h3>Surviving feature set ({FS['n_final']})</h3>
      <p>All nine retained characteristics clear every gate, span all three data families, and carry
      VIFs comfortably below threshold — so each coefficient is stable and interpretable.</p>
      <p>Notably, the <strong>alternative-data</strong> signal
      (<code>digital_engagement_score</code>) survives selection, illustrating how non-traditional
      attributes add lift for thin-file applicants.</p>
    </div>
    <div>
      {table(vif_rows, ["f","vif"], ["Final feature","VIF"], fmt={"vif":lambda v:f"{v:.2f}"})}
    </div>
  </div>

  {figure("04_corr_heatmap.png","Pairwise correlation of the WOE-transformed features. No off-diagonal block is high enough to destabilise the logistic coefficients.")}
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 05 — SCORECARD & SCALING
# --------------------------------------------------------------------------- #
pts_bureau = [r for r in SC["points_table"] if r["feature"]=="bureau_score"]
coef_rows = [{"f":k,"b":v} for k,v in FS["coefficients"].items()]
scale_code = f'''<span class="c"># Scale model log-odds to an industry points system</span>
factor = PDO / np.log(2)                         <span class="c"># PDO = {SC['pdo']}</span>
offset = base_score - factor * np.log(base_odds) <span class="c"># {SC['base_score']} pts @ {SC['base_odds']}:1</span>

<span class="c"># Points for each attribute (bin), distributing the intercept across n features</span>
points = -(beta_i * woe_i + intercept / n) * factor + offset / n'''

s05 = f"""
<section id="scorecard">
  <span class="eyebrow">Section 05 · The model</span>
  <h2>Scorecard development &amp; scaling</h2>
  <p class="lead">The champion is a <strong>logistic regression on WOE inputs</strong> — the banking
  standard for application scorecards. It is monotone, additive and fully transparent, and it maps
  cleanly to points and reason codes that Model Risk Management and fair-lending review require. The
  marginal accuracy of a black box (Section 07) does not justify losing that.</p>

  <p>Model log-odds are scaled to a familiar points system so a business user never has to read a
  probability. The anchors here: <strong>{SC['base_score']} points at {SC['base_odds']}:1
  good:bad odds, with {SC['pdo']} points to double the odds (PDO)</strong>.</p>

  <pre>{scale_code}</pre>

  <div class="cards">
    <div class="card tealtop"><div class="v">{SC['factor']:.1f}</div><div class="l">Factor = PDO / ln(2)</div></div>
    <div class="card tealtop"><div class="v">{SC['offset']:.0f}</div><div class="l">Offset (score anchor)</div></div>
    <div class="card ambertop"><div class="v">{SC['score_min']}–{SC['score_max']}</div><div class="l">Achievable score range</div></div>
    <div class="card"><div class="v">{FS['intercept']:+.2f}</div><div class="l">Model intercept (log-odds)</div></div>
  </div>

  <div class="two">
    <div>
      <h3>Fitted coefficients</h3>
      <p>Every coefficient is negative, confirming each characteristic pushes risk in the expected
      direction once WOE-encoded. Magnitudes track the IV ranking — bureau score carries the most
      weight.</p>
      {table(coef_rows, ["f","b"], ["Feature (WOE)","β"], fmt={"b":lambda v:f"{v:+.3f}"})}
    </div>
    <div>
      <h3>Scorecard points — bureau score</h3>
      <p>The log-odds become integer points an underwriter can read directly: the lowest score band
      contributes {pts_bureau[0]['points']} points, the highest {pts_bureau[-1]['points']}.</p>
      {table(pts_bureau, ["bin","woe","points"], ["Score bin","WOE","Points"],
             highlight_col="points", fmt={"woe":lambda v:f"{v:+.2f}"})}
    </div>
  </div>
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 06 — VALIDATION
# --------------------------------------------------------------------------- #
val_rows = R["validation"]["champion"]
band = R["score_band_table"]
s06 = f"""
<section id="validation">
  <span class="eyebrow">Section 06 · Does it work</span>
  <h2>Validation</h2>
  <p class="lead">A scorecard is only deployable if it does three things at once:
  <strong>discriminates</strong> (separates goods from bads), <strong>rank-orders</strong>
  monotonically, and is <strong>well calibrated</strong> (predicted PD matches reality). We test all
  three — on a held-out in-time sample and again out-of-time.</p>

  <div class="two">
    <div>{figure("05_ks_curve.png","KS = max gap between the cumulative good and bad distributions.")}</div>
    <div>{figure("07_score_distribution.png","Score densities for goods and bads — visibly separated, with goods shifted right.")}</div>
  </div>

  <h3>Headline metrics across samples</h3>
  <p>Performance is consistent from train to test — <strong>no overfitting</strong> — and holds out
  of time: discrimination (KS {champ_oot['ks']*100:.1f}, Gini {champ_oot['gini']:.3f}) is essentially
  unchanged on the unseen 2024 vintage even though its bad rate is far higher.</p>
  {table(val_rows, ["segment","n","bad_rate","ks","gini","auc","brier"],
         ["Sample","N","Bad rate","KS","Gini","AUC","Brier"],
         fmt={"n":comma,"bad_rate":pct2,"ks":lambda v:f"{v*100:.1f}","gini":f3,"auc":f3,"brier":lambda v:f"{v:.4f}"})}

  <h3>Rank-ordering &amp; calibration</h3>
  <p>Across ten score bands the bad rate falls monotonically from
  {band[0]['bad_rate']*100:.1f}% in the riskiest band to {band[-1]['bad_rate']*100:.1f}% in the
  safest — a {band[0]['bad_rate']/band[-1]['bad_rate']:.0f}× spread, and a perfect rank order. The
  calibration curve tracks the diagonal, so the scores can be read directly as probabilities.</p>

  <div class="two">
    <div>{figure("08_rank_ordering.png","Bad rate by score band — strictly monotone.")}</div>
    <div>{figure("09_calibration.png","Predicted PD vs observed default rate.")}</div>
  </div>

  {table(band, ["band","n","min_score","max_score","bad_rate","cum_bad_pct","ks","lift"],
         ["Band","N","Min","Max","Bad rate","Cum % bad","KS","Lift"],
         fmt={"n":comma,"bad_rate":pct,"cum_bad_pct":pct,"ks":f3,"lift":lambda v:f"{v:.2f}×"})}
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 07 — CHAMPION / CHALLENGER
# --------------------------------------------------------------------------- #
xgb = [r for r in MC if "XGB" in r["segment"]][0]
rf  = [r for r in MC if "Random" in r["segment"]][0]
lg  = [r for r in MC if "champion" in r["segment"]][0]
gini_gap = (max(xgb["gini"], rf["gini"]) - lg["gini"])
s07 = f"""
<section id="challenger">
  <span class="eyebrow">Section 07 · Benchmark</span>
  <h2>Champion–challenger</h2>
  <p class="lead">Is the interpretable model leaving money on the table? To find out, the logistic
  champion is benchmarked against two challengers named in every modern risk stack —
  <strong>XGBoost</strong> and <strong>Random Forest</strong> — fit on raw features so they can
  exploit non-linearities and interactions the scorecard cannot.</p>

  <div class="two">
    <div>{figure("06_roc.png","ROC curves — the three models are nearly indistinguishable.")}</div>
    <div>{figure("10_model_comparison.png","KS, Gini and AUC side by side.")}</div>
  </div>

  {table(MC, ["segment","ks","gini","auc","brier"],
         ["Model","KS","Gini","AUC","Brier"],
         highlight_col=None,
         fmt={"ks":lambda v:f"{v*100:.1f}","gini":f3,"auc":f3,"brier":lambda v:f"{v:.4f}"})}

  <div class="note gov"><span class="nt">The decision — and why it's the senior one</span>
  The challengers improve Gini by only <strong>{gini_gap:.3f}</strong> ({gini_gap/lg['gini']*100:.1f}%)
  over the scorecard, while their probabilities are far less calibrated (Brier {xgb['brier']:.3f} vs
  {lg['brier']:.3f}) and they cannot natively produce adverse-action reason codes. On a regulated
  lending decision the interpretable champion wins decisively; the tiny lift does not justify the
  governance, explainability and stability cost. The challenger is retained as a benchmark and a
  source of candidate features, not as the production engine.</div>

  <p>Reassuringly, XGBoost's importance ranking corroborates the WOE/IV story — bureau score,
  delinquencies and utilisation lead — which is itself evidence the champion is capturing the real
  signal rather than an artefact of its functional form.</p>

  {figure("11_feature_importance.png","XGBoost gain importance — consistent with the Information Value ranking.")}
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 08 — MONITORING
# --------------------------------------------------------------------------- #
csi = MON["csi"]
s08 = f"""
<section id="monitoring">
  <span class="eyebrow">Section 08 · Staying healthy</span>
  <h2>Model monitoring — PSI &amp; CSI</h2>
  <p class="lead">A model decays the moment it goes live. Monitoring quantifies that decay before it
  becomes a loss. Two indices form the core of the model-health MIS that governance reviews each
  cycle.</p>

  <div class="formula">PSI = Σ ( %actual − %expected ) · ln( %actual / %expected )</div>

  <div class="two">
    <div>
      <h3>Population Stability Index</h3>
      <p>PSI compares the <strong>score distribution</strong> of the recent book against
      development. Here PSI = <strong>{MON['score_psi']:.3f}</strong> — squarely in the
      <span class="pill amber">investigate</span> zone (0.10–0.25). The recent population has
      migrated sharply toward the low-score bands.</p>
      <ul class="clean">
        <li>PSI &lt; 0.10 — <span class="pill good">stable</span></li>
        <li>0.10–0.25 — <span class="pill amber">moderate shift, investigate</span></li>
        <li>&gt; 0.25 — <span class="pill bad">significant shift, recalibrate / rebuild</span></li>
      </ul>
    </div>
    <div>{figure("12_psi.png","Development vs recent score distribution.")}</div>
  </div>

  <h3>Characteristic Stability Index — what's driving the shift</h3>
  <p>CSI applies the same arithmetic to each <strong>input</strong>, so the alert is actionable.
  Revolving utilisation (CSI {csi[0]['csi']:.2f}) and bureau score
  ({[c['csi'] for c in csi if c['characteristic']=='bureau_score'][0]:.2f}) are doing the moving — a
  classic late-cycle deterioration in applicant quality.</p>

  {figure("13_csi.png","CSI by characteristic, against the 0.10 / 0.25 action thresholds.")}
  {table(csi, ["characteristic","csi","flag"], ["Characteristic","CSI","Status"],
         fmt={"csi":f3})}

  <div class="note gov"><span class="nt">The nuanced conclusion</span>
  Critically, <strong>discrimination held</strong> out-of-time (Gini barely moved) even as PSI
  flagged red and the bad rate nearly doubled. That tells a precise story: the model still
  <em>ranks</em> risk correctly, but its score-to-PD mapping is now <em>mis-calibrated</em> to a
  riskier population. The right action is <strong>recalibration of the cut-off and PD anchors</strong>,
  not a full redevelopment — a distinction that saves months and is exactly what an MRM review wants
  to see articulated.</div>
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 09 — BUSINESS
# --------------------------------------------------------------------------- #
swap = BIZ["swap_set"]
s09 = f"""
<section id="business">
  <span class="eyebrow">Section 09 · The money</span>
  <h2>Lending strategy &amp; P&amp;L</h2>
  <p class="lead">A scorecard only earns its keep when it changes a decision. This section turns the
  score into an approval policy and quantifies it in dollars — the conversation Product, Risk and
  Finance actually have.</p>

  <h3>The growth–risk frontier</h3>
  <p>Each candidate cut-off trades approval volume against approved-book bad rate. The frontier is
  the menu of strategies leadership chooses from; the model's job is to push that whole curve down
  and to the right.</p>
  {figure("14_strategy_curve.png","Approval rate vs bad rate of the approved book, swept across cut-offs.")}

  <h3>Profit-optimised cut-off</h3>
  <p>Applying simple unit economics — <span class="pill ink">{money(BIZ['unit_economics']['margin_per_good'])} margin per good</span>,
  <span class="pill ink">{money(BIZ['unit_economics']['loss_per_bad'])} loss per bad</span> —
  expected portfolio profit is maximised at a score of <strong>{PB['cutoff_score']:.0f}</strong>.
  That cut-off approves <strong>{PB['approval_rate']*100:.0f}%</strong> of applicants at a
  <strong>{PB['approved_bad_rate']*100:.1f}%</strong> bad rate — less than half the
  {BIZ['through_door_bad_rate']*100:.1f}% through-the-door rate — for
  <strong>{money(PB['expected_profit'])}</strong> of expected profit on the test book
  ({money(PB['profit_per_app'])} per application).</p>

  <div class="cards">
    <div class="card goodtop"><div class="v">{PB['cutoff_score']:.0f}</div><div class="l">Profit-optimal score cut-off</div></div>
    <div class="card goodtop"><div class="v">{PB['approval_rate']*100:.0f}<span class="u">%</span></div><div class="l">Approval rate</div></div>
    <div class="card goodtop"><div class="v">{PB['approved_bad_rate']*100:.1f}<span class="u">%</span></div><div class="l">Approved-book bad rate</div></div>
    <div class="card goodtop"><div class="v">${PB['expected_profit']/1e6:.1f}<span class="u">M</span></div><div class="l">Expected profit (test book)</div></div>
  </div>
  {figure("15_profit_curve.png","Expected portfolio profit as a function of the approval cut-off.")}

  <h3>Swap-set vs an incumbent policy</h3>
  <p>The sharpest proof of value is a swap-set against a naïve single-variable rule (approve if
  bureau score ≥ 620). At a comparable approval rate, the scorecard <strong>swaps out</strong>
  accounts running a {SV['swap_out_bad_rate']*100:.1f}% bad rate and <strong>swaps in</strong>
  accounts at just {SV['swap_in_bad_rate']*100:.1f}% — it finds good customers the bureau cut-off
  rejects and rejects bad customers the bureau cut-off would have booked.</p>

  {figure("16_swap_set.png","Swap-set bad rates: the scorecard's swap-ins are far cleaner than the swap-outs it removes.")}
  {table(swap, ["group","n","bads","bad_rate"], ["Population","N","Bads","Bad rate"],
         fmt={"n":comma,"bad_rate":pct})}
</section>
"""

# --------------------------------------------------------------------------- #
# SECTION 10 — GOVERNANCE
# --------------------------------------------------------------------------- #
s10 = f"""
<section id="governance">
  <span class="eyebrow">Section 10 · Controls</span>
  <h2>Governance &amp; Model Risk Management</h2>
  <p class="lead">In a regulated lender, a model is only as good as its controls. The build is
  designed against <strong>SR 11-7</strong> model-risk principles and consumer-protection
  expectations from the first decision onward.</p>

  <div class="cards" style="grid-template-columns:1fr 1fr">
    <div class="card tealtop"><div class="l"><strong>Conceptual soundness</strong><br>Transparent, monotone WOE + logistic form; every coefficient signed and economically justified.</div></div>
    <div class="card tealtop"><div class="l"><strong>Outcomes analysis</strong><br>KS, Gini, calibration and rank-ordering verified in-time and out-of-time.</div></div>
    <div class="card tealtop"><div class="l"><strong>Ongoing monitoring</strong><br>PSI / CSI dashboard with defined action thresholds and a recalibrate-vs-rebuild rule.</div></div>
    <div class="card tealtop"><div class="l"><strong>Effective challenge</strong><br>Independent XGBoost / RF benchmark bounds the cost of interpretability.</div></div>
  </div>

  <h3>Fair lending &amp; explainability</h3>
  <p>Because the scorecard is additive in points, each declined applicant's largest negative point
  contributions yield <strong>adverse-action reason codes</strong> directly — a regulatory
  requirement that black-box models struggle to meet. Protected-class attributes are excluded by
  construction, and a production deployment would add disparate-impact testing on model outcomes.</p>

  <h3>Known limitations &amp; next steps</h3>
  <ul class="clean">
    <li>Data is <strong>synthetic</strong> but engineered to reproduce realistic risk relationships, bad rates and stability behaviour; the methodology transfers directly to production data.</li>
    <li>Reject inference is not modelled here — a real through-the-door scorecard must correct for the truncation of declined applicants (e.g. parcelling, augmentation).</li>
    <li>Next steps: champion–challenger in production shadow mode, segment-level scorecards (thin-file vs thick-file), and a quarterly recalibration cadence triggered by the PSI thresholds above.</li>
  </ul>

  <div class="note"><span class="nt">What this case study demonstrates</span>
  The full lifecycle a model-development lead owns — target design and leakage control, WOE/IV
  engineering, scaled logistic scorecards, champion–challenger judgment, PSI/CSI monitoring, and the
  translation of all of it into lending strategy and P&amp;L — under explicit model-risk governance.</div>
</section>
"""

# --------------------------------------------------------------------------- #
# FOOTER
# --------------------------------------------------------------------------- #
footer = f"""
<div class="gradbar thin"></div>
<footer><div class="wrap">
  <div>
    <b>Probability-of-Default Scorecard — Model Development Case Study</b><br>
    <span style="color:#9DB0BD">Prepared by {GENERIC_NAME} &nbsp;·&nbsp; Python · scikit-learn · XGBoost</span>
  </div>
  <div class="disc">
    Built as a technical portfolio piece. All data is synthetic and generated by the accompanying
    pipeline; no real consumer data is used. Figures and metrics are fully reproducible by running
    <code>python src/run_pipeline.py</code>. Source &amp; documentation on
    <a href="#">GitHub</a>.
  </div>
</div></footer>
"""

# --------------------------------------------------------------------------- #
# ASSEMBLE
# --------------------------------------------------------------------------- #
HTML = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PD Scorecard — Model Development Case Study</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head><body>
{masthead}
<div class="wrap"><div class="grid">
{nav}
<main>
{s01}{s02}{s03}{s04}{s05}{s06}{s07}{s08}{s09}{s10}
</main>
</div></div>
{footer}
</body></html>"""

out_path = os.path.join(OUT, "Credit_Risk_Scorecard_Report.html")
with open(out_path, "w") as f:
    f.write(HTML)
size_kb = os.path.getsize(out_path) / 1024
print(f"Report written: {out_path}  ({size_kb:.0f} KB)")
