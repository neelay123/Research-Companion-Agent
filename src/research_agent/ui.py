"""streamlit run -m research_agent.ui"""
import asyncio
import streamlit as st
from .graph import research_app

st.set_page_config(page_title="Research Companion", layout="wide")
st.title("Research Companion")

q = st.text_input("Question:")
go = st.button("Ask")

async def run(question: str):
    progress = st.empty()
    answer_box = st.empty()
    citations_box = st.empty()
    events: list[str] = []
    async with research_app() as app:
        cfg = {
            "configurable": {"thread_id": f"session-{hash(question)}",
                             "max_concurrency": 4},
            "recursion_limit": 50,
        }
        async for ev in app.astream({"question": question}, cfg, stream_mode="updates"):
            events.append(str(ev)[:300])
            progress.code("\n".join(events[-20:]))
        state = await app.aget_state(cfg)
        answer_box.markdown(state.values.get("answer", "(no answer)"))
        citations_box.write(state.values.get("citations", []))

if go and q:
    asyncio.run(run(q))
