"""
Deeper — a Streamlit front end for a multi-agent research pipeline.

The pipeline itself lives in `agents.py` (build_search_agent, build_reader_agent,
writer_chain, critic_chain) — this file only orchestrates the UI and calls those
four pieces in the same order as the original run_research_pipeline() script:
search -> read -> write -> critique.

Run with:
    streamlit run app.py

This file expects `agents.py` to be importable from wherever you launch it
(usually: sitting in the same folder). If your agents module lives somewhere
else, adjust the import line below.
"""

import re
from datetime import datetime

import streamlit as st

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


# =============================================================================
# Page config — must be the first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="Deeper",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = ["Search", "Read", "Write", "Critique"]


# =============================================================================
# Helpers
# =============================================================================
def _extract_text(result) -> str:
    """Pull plain text out of whatever shape an agent or chain hands back."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        messages = result.get("messages") or result.get("message")
        if messages:
            last = messages[-1]
            return getattr(last, "content", str(last))
        return str(result)
    return getattr(result, "content", str(result))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _reading_minutes(text: str) -> int:
    return max(1, round(_word_count(text) / 200))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "report"


@st.cache_resource(show_spinner=False)
def _get_search_agent():
    return build_search_agent()


@st.cache_resource(show_spinner=False)
def _get_reader_agent():
    return build_reader_agent()


# =============================================================================
# Styling — Streamlit's own [theme] in .streamlit/config.toml sets the base
# palette; this CSS layers on the typography and the bespoke elements
# (hero, stepper, meta row) that theming alone can't reach.
# =============================================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            letter-spacing: -0.01em;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #22334F;
        }

        .sidebar-title {
            font-family: 'Fraunces', serif;
            font-size: 1.05rem;
            color: #F3EFE4;
            margin-bottom: 0.4rem;
        }

        .hero {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.5rem 0 1.25rem 0;
            border-bottom: 1px solid #22334F;
            margin-bottom: 1.5rem;
        }
        .hero-mark {
            font-size: 2.1rem;
            color: #C99A46;
            line-height: 1;
        }
        .hero-text h1 {
            margin: 0;
            font-size: 2.3rem;
            font-weight: 600;
            color: #F3EFE4;
        }
        .hero-text p {
            margin: 0.25rem 0 0 0;
            color: #90A1B8;
            font-size: 0.98rem;
        }

        .stepper {
            display: flex;
            align-items: flex-start;
            margin: 0.25rem 0 1.5rem 0;
        }
        .step {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.35rem;
            min-width: 90px;
        }
        .step-marker {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            border: 1px solid #22334F;
            color: #90A1B8;
            background: transparent;
        }
        .step.active .step-marker {
            border-color: #C99A46;
            color: #C99A46;
        }
        .step.done .step-marker {
            background: #3E8073;
            border-color: #3E8073;
            color: #0B1220;
        }
        .step-label {
            font-size: 0.74rem;
            color: #90A1B8;
        }
        .step.active .step-label,
        .step.done .step-label {
            color: #F3EFE4;
        }
        .step-connector {
            flex: 1;
            height: 1px;
            background: #22334F;
            margin: 13px -6px 0 -6px;
        }
        .step-connector.done {
            background: #3E8073;
        }

        .meta-row {
            display: flex;
            gap: 1.5rem;
            align-items: baseline;
            flex-wrap: wrap;
            margin: 0.25rem 0 1rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #22334F;
        }
        .meta-row .meta-topic {
            font-family: 'Fraunces', serif;
            font-size: 1.2rem;
            color: #F3EFE4;
        }
        .meta-row .meta-dim {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #90A1B8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(placeholder, active_index, done=False):
    """Draw the 4-stage progress strip into a reusable placeholder."""
    parts = []
    for i, label in enumerate(STEPS):
        if done or i < active_index:
            state = "done"
        elif i == active_index:
            state = "active"
        else:
            state = "pending"
        parts.append(
            f'<div class="step {state}"><div class="step-marker">{i + 1}</div>'
            f'<div class="step-label">{label}</div></div>'
        )
        if i < len(STEPS) - 1:
            connector_state = "done" if (done or i < active_index) else "pending"
            parts.append(f'<div class="step-connector {connector_state}"></div>')
    placeholder.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


# =============================================================================
# Pipeline — same order as run_research_pipeline(), with live UI feedback
# =============================================================================
def run_pipeline(topic: str, stepper_placeholder) -> dict:
    state = {}

    render_stepper(stepper_placeholder, 0)
    with st.status("Searching for recent, reliable sources...", expanded=True) as status:
        search_agent = _get_search_agent()
        search_result = search_agent.invoke(
            {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
        )
        state["search_results"] = _extract_text(search_result)
        st.caption(f"{_word_count(state['search_results'])} words gathered")
        status.update(label="Search complete", state="complete")

    render_stepper(stepper_placeholder, 1)
    with st.status("Reading the most relevant source in depth...", expanded=True) as status:
        reader_agent = _get_reader_agent()
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about {topic}, "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:800]}",
                    )
                ]
            }
        )
        state["scraped_content"] = _extract_text(reader_result)
        st.caption(f"{_word_count(state['scraped_content'])} words extracted")
        status.update(label="Reading complete", state="complete")

    render_stepper(stepper_placeholder, 2)
    with st.status("Drafting the report...", expanded=True) as status:
        combined_research = (
            f"SEARCH_RESULT:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )
        report_raw = writer_chain.invoke({"topic": topic, "research": combined_research})
        state["report"] = _extract_text(report_raw)
        st.caption(f"{_word_count(state['report'])} words written")
        status.update(label="Draft complete", state="complete")

    render_stepper(stepper_placeholder, 3)
    with st.status("Critic agent is reviewing the draft...", expanded=True) as status:
        feedback_raw = critic_chain.invoke({"topic": topic, "research": state["report"]})
        state["feedback"] = _extract_text(feedback_raw)
        status.update(label="Review complete", state="complete")

    render_stepper(stepper_placeholder, len(STEPS) - 1, done=True)
    return state


# =============================================================================
# App
# =============================================================================
def main():
    inject_css()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "current" not in st.session_state:
        st.session_state.current = None

    with st.sidebar:
        st.markdown('<div class="sidebar-title">History</div>', unsafe_allow_html=True)
        if not st.session_state.history:
            st.caption("Reports you generate this session will show up here.")
        else:
            for i, entry in enumerate(reversed(st.session_state.history)):
                if st.button(entry["topic"], key=f"hist-{i}", use_container_width=True):
                    st.session_state.current = entry

    st.markdown(
        """
        <div class="hero">
            <div class="hero-mark">◆</div>
            <div class="hero-text">
                <h1>Deeper</h1>
                <p>Four agents, one report: search, read, write, and critique.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("topic_form", clear_on_submit=False):
        col1, col2 = st.columns([5, 1])
        with col1:
            topic = st.text_input(
                "Topic",
                placeholder="e.g. Solid-state batteries for grid storage",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Generate report", use_container_width=True)

    stepper_placeholder = st.empty()
    current = st.session_state.current
    if current and not submitted:
        render_stepper(stepper_placeholder, len(STEPS) - 1, done=True)
    else:
        render_stepper(stepper_placeholder, 0)

    if submitted:
        if not topic or not topic.strip():
            st.warning("Enter a topic to begin.")
        else:
            try:
                result_state = run_pipeline(topic.strip(), stepper_placeholder)
                entry = {
                    "topic": topic.strip(),
                    "state": result_state,
                    "timestamp": datetime.now().strftime("%b %d, %H:%M"),
                }
                st.session_state.history.append(entry)
                st.session_state.current = entry
            except Exception as exc:
                st.error(f"The pipeline hit an error: {exc}")

    current = st.session_state.current
    if current:
        state = current["state"]
        report_text = state.get("report", "")
        feedback_text = state.get("feedback", "")
        search_text = state.get("search_results", "")
        scraped_text = state.get("scraped_content", "")

        left, right = st.columns([5, 1])
        with left:
            st.markdown(
                f"""
                <div class="meta-row">
                    <span class="meta-topic">{current['topic']}</span>
                    <span class="meta-dim">{current['timestamp']}</span>
                    <span class="meta-dim">{_word_count(report_text)} words · {_reading_minutes(report_text)} min read</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.download_button(
                "Download .md",
                data=report_text,
                file_name=f"{_slugify(current['topic'])}-report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        tab_report, tab_critic, tab_sources = st.tabs(["Report", "Critic review", "Sources"])

        with tab_report:
            with st.container(border=True):
                st.markdown(report_text if report_text else "_No report generated._")

        with tab_critic:
            with st.container(border=True):
                st.markdown(feedback_text if feedback_text else "_No feedback generated._")

        with tab_sources:
            with st.expander("Search results", expanded=False):
                st.markdown(search_text if search_text else "_None._")
            with st.expander("Scraped content", expanded=False):
                st.markdown(scraped_text if scraped_text else "_None._")
    else:
        st.caption("Enter a topic above and click Generate report to start.")


if __name__ == "__main__":
    main()