"""Advanced Streamlit interface with debug mode and analytics."""

import streamlit as st
import sys
import os
import time
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import create_agent

st.set_page_config(
    page_title="Placement Assistant (Advanced)",
    page_icon="🎓",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .debug-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    """Load and cache the agent."""
    with st.spinner("Loading AI Agent..."):
        return create_agent(use_llm=True)


def main():
    st.title("🎓 Placement Assistant (Advanced Mode)")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        debug_mode = st.checkbox("🔍 Debug Mode", value=False)
        show_raw_data = st.checkbox("📊 Show Raw Data", value=False)
        
        st.markdown("---")
        
        # Quick filters
        st.header("🔧 Quick Queries")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📍 Bangalore"):
                st.session_state.quick_query = "Companies in Bangalore"
            if st.button("💰 High Stipend"):
                st.session_state.quick_query = "Companies with stipend more than 50000"
        with col2:
            if st.button("📋 All Companies"):
                st.session_state.quick_query = "List all companies"
            if st.button("🎯 Dell Process"):
                st.session_state.quick_query = "Selection process for Dell"
        
        st.markdown("---")
        
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Initialize
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    agent = load_agent()
    
    # Main chat area
    chat_container = st.container()
    
    # Display history
    with chat_container:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["query"])
            
            with st.chat_message("assistant"):
                st.markdown(entry["response"])
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    conf = entry.get("confidence", 0)
                    color = "green" if conf > 0.7 else "orange" if conf > 0.4 else "red"
                    st.metric("Confidence", f"{conf:.0%}", delta_color="off")
                with col2:
                    st.metric("Time", f"{entry.get('time', 0):.2f}s")
                with col3:
                    st.metric("Intent", entry.get("intent", "N/A"))
                with col4:
                    st.metric("Retries", entry.get("retries", 0))
                
                # Debug info
                if debug_mode and "debug" in entry:
                    with st.expander("🔍 Debug Info"):
                        st.json(entry["debug"])
                
                # Raw data
                if show_raw_data and "raw_data" in entry:
                    with st.expander("📊 Raw Data"):
                        st.json(entry["raw_data"])
    
    # Handle quick query
    if "quick_query" in st.session_state:
        query = st.session_state.quick_query
        del st.session_state.quick_query
        process_and_display(agent, query, debug_mode, show_raw_data)
        st.rerun()
    
    # Input
    query = st.chat_input("Ask about placements...")
    if query:
        process_and_display(agent, query, debug_mode, show_raw_data)
        st.rerun()
    
    # Company browser
    with st.expander("📚 Browse Companies"):
        companies = agent.get_companies()
        
        search = st.text_input("Search company:", "")
        filtered = [c for c in companies if search.lower() in c.lower()] if search else companies
        
        cols = st.columns(4)
        for i, company in enumerate(filtered[:20]):
            with cols[i % 4]:
                if st.button(company, key=f"company_{company}"):
                    st.session_state.quick_query = f"Tell me about {company} internship"


def process_and_display(agent, query: str, debug_mode: bool, show_raw: bool):
    """Process query and add to history."""
    
    start_time = time.time()
    
    try:
        response = agent.query(query, verbose=False)
        elapsed = time.time() - start_time
        
        entry = {
            "query": query,
            "response": response.answer,
            "confidence": response.feedback.confidence_score,
            "time": elapsed,
            "intent": response.plan.intent,
            "retries": response.retries,
            "companies": response.plan.companies_mentioned
        }
        
        if debug_mode:
            entry["debug"] = {
                "intent": response.plan.intent,
                "tools_used": [t.get("tool") for t in response.plan.tools_to_use],
                "companies_detected": response.plan.companies_mentioned,
                "attributes": response.plan.attributes_requested,
                "reasoning": response.plan.reasoning,
                "critic_feedback": {
                    "complete": response.feedback.is_complete,
                    "relevant": response.feedback.is_relevant,
                    "reasoning": response.feedback.reasoning
                }
            }
        
        if show_raw:
            entry["raw_data"] = {
                "enriched_companies": list(response.execution.enriched_results.keys()) if response.execution.enriched_results else [],
                "tool_results": [
                    {"tool": r.tool_name, "success": r.success, "message": r.message}
                    for r in response.execution.tool_results
                ]
            }
        
        st.session_state.chat_history.append(entry)
        
    except Exception as e:
        st.session_state.chat_history.append({
            "query": query,
            "response": f"❌ Error: {str(e)}",
            "confidence": 0,
            "time": time.time() - start_time,
            "intent": "error",
            "retries": 0
        })


if __name__ == "__main__":
    main()
