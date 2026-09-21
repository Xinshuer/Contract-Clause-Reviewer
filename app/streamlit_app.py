"""Streamlit UI: upload -> risk list -> click into the clause -> accept / reject.

    streamlit run app/streamlit_app.py

The human is the last step: nothing is changed in the contract unless a person
accepts a proposal here and exports it. "Reject all" and "export original" are
always one click away.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clausecheck.config import Settings  # noqa: E402
from clausecheck.models import ReviewReport  # noqa: E402
from clausecheck.pipeline import review_contract  # noqa: E402
from clausecheck.redline import apply_replacement  # noqa: E402
from clausecheck.split import load_contract  # noqa: E402
from clausecheck.tracing import ENABLED as TRACING, flush  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_contract.txt"
BADGE = {"accept": "🟢", "flag": "🟠", "redline": "🔴"}
RISK = {"low": "low", "medium": "MED", "high": "HIGH"}

st.set_page_config(page_title="Clause Check", page_icon="📄", layout="wide")
st.title("Clause Check - contract clause reviewer")
st.caption("Deterministic split -> per-clause review against the playbook -> you decide. The tool never rewrites the contract.")

with st.sidebar:
    st.header("Run")
    default_settings = Settings()
    modes = ["auto", "live", "deepseek", "local", "mock"]
    mode = st.selectbox("Mode", modes, index=modes.index(default_settings.mode if default_settings.mode in modes else "auto"))
    src = st.radio("Contract", ["Sample SaaS agreement", "Upload"], horizontal=True)
    upload = st.file_uploader("PDF or TXT", type=["pdf", "txt", "md"]) if src == "Upload" else None
    run = st.button("Review", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"Model: {default_settings.model}  |  Langfuse tracing: {'on' if TRACING else 'off'}")

if run:
    settings = Settings(mode=mode)
    if upload is not None:
        suffix = Path(upload.name).suffix.lower() or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(upload.getvalue())
            path = Path(tmp.name)
        name = Path(upload.name).stem
    else:
        path, name = SAMPLE, SAMPLE.stem
    try:
        contract = load_contract(path, name=name)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    bar = st.progress(0.0, text="Reviewing...")
    log = st.empty()

    def progress(i: int, n: int, review) -> None:
        bar.progress(i / n, text=f"Reviewing clause {review.clause.clause_id} ({i}/{n})")
        log.caption(f"{review.clause.clause_id}: {review.verdict.verdict} - tools: {', '.join(review.trace.tool_calls) or 'none'}")

    report = review_contract(contract, settings, progress=progress)
    flush()
    bar.empty()
    log.empty()
    st.session_state["report"] = report.model_dump()
    st.session_state["contract_text"] = contract.text
    st.session_state["decisions"] = {}

if "report" not in st.session_state:
    st.info("Pick a contract and press Review. Without credentials the app runs the offline rule-based reviewer (mode = mock).")
    st.stop()

report = ReviewReport.model_validate(st.session_state["report"])
decisions: dict[str, str] = st.session_state["decisions"]
s = report.summary

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Clauses", s["clauses_reviewed"])
c2.metric("Redline", s["counts"]["redline"])
c3.metric("Flag", s["counts"]["flag"])
c4.metric("Accept", s["counts"]["accept"])
c5.metric("Est. cost", f"${s['estimated_cost_usd']}")
st.caption(f"mode={report.mode} model={report.model} tokens={s['tokens']} cache hit={s['cache_hit_ratio']} latency={s['total_latency_ms']} ms")

left, right = st.columns([1, 2], gap="large")
order = {"redline": 0, "flag": 1, "accept": 2}
reviews = sorted(report.reviews, key=lambda r: (order[r.verdict.verdict], {"high": 0, "medium": 1, "low": 2}[r.verdict.risk_level], r.clause.clause_id))

with left:
    st.subheader("Risk list")
    show_accept = st.toggle("Show accepted clauses", value=False)
    rows = [r for r in reviews if show_accept or r.verdict.verdict != "accept"]
    labels = [f"{BADGE[r.verdict.verdict]} {r.clause.clause_id}  {r.clause.heading[:34]}  [{RISK[r.verdict.risk_level]}]" for r in rows]
    if not rows:
        st.success("Nothing needs attention.")
        st.stop()
    pick = st.radio("Clause", labels, label_visibility="collapsed")
    selected = rows[labels.index(pick)]

with right:
    r = selected
    v = r.verdict
    st.subheader(f"{BADGE[v.verdict]} Clause {r.clause.clause_id} - {r.clause.heading}")
    st.caption(f"topic: {v.topic}  |  risk: {v.risk_level}  |  confidence: {v.confidence}  |  rules: {', '.join(v.rule_ids) or '-'}  |  consulted: {', '.join(v.consulted_clauses) or '-'}")
    st.markdown("**Original text**")
    text = r.clause.text
    if r.redline and r.redline.located:
        a, b = r.redline.start, r.redline.end
        st.markdown(text[:a] + " **:red[" + text[a:b] + "]** " + text[b:])
    else:
        st.write(text)
    st.markdown("**Why**")
    st.write(v.rationale)
    if r.redline:
        st.markdown("**Proposed replacement**" + ("" if r.redline.located else "  (anchor not found - place manually)"))
        st.info(r.redline.replacement)
        if r.redline.diff:
            with st.expander("diff"):
                st.code(r.redline.diff, language="diff")
    with st.expander("trace"):
        st.json({"tool_calls": r.trace.tool_calls, "steps": r.trace.steps, "tokens": r.trace.total_tokens, "latency_ms": r.trace.latency_ms, "stop_reason": r.trace.stop_reason, "error": r.trace.error})
    if v.verdict != "accept":
        choice = st.radio("Your decision", ["undecided", "accept proposal", "reject"], index=["undecided", "accept proposal", "reject"].index(decisions.get(r.clause.clause_id, "undecided")), horizontal=True, key=f"dec-{r.clause.clause_id}")
        decisions[r.clause.clause_id] = choice

st.divider()
e1, e2, e3, e4 = st.columns(4)
if e1.button("Reject all proposals"):
    for rv in report.reviews:
        if rv.verdict.verdict != "accept":
            decisions[rv.clause.clause_id] = "reject"
    st.rerun()
accepted = [rv for rv in report.reviews if decisions.get(rv.clause.clause_id) == "accept proposal" and rv.redline and rv.redline.located]
e2.download_button("Export report (JSON)", data=report.model_dump_json(indent=2), file_name=f"{report.contract_name}.report.json", mime="application/json")
e3.download_button("Export original text", data=st.session_state["contract_text"], file_name=f"{report.contract_name}.original.txt")

redlined = st.session_state["contract_text"]
for rv in sorted(accepted, key=lambda x: x.clause.char_start, reverse=True):
    seg = redlined[rv.clause.char_start:rv.clause.char_end]
    off = seg.find(rv.redline.matched_text or "")
    if off >= 0:
        s0 = rv.clause.char_start + off
        redlined = apply_replacement(redlined, s0, s0 + len(rv.redline.matched_text), rv.redline.replacement)
e4.download_button(f"Export with {len(accepted)} accepted redline(s)", data=redlined, file_name=f"{report.contract_name}.redlined.txt", disabled=not accepted)
st.caption("Decisions: " + json.dumps({k: v for k, v in decisions.items() if v != 'undecided'}) if any(v != "undecided" for v in decisions.values()) else "No decisions yet.")
