"""Streamlit chatbot interface for Placement RAG Agent."""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="Placement RAG Assistant", layout="wide")

st.title("🎓 Placement RAG Assistant")

# Import agent after streamlit config
from agent.orchestrator import create_agent


@st.cache_resource
def load_agent():
    """Load and cache the placement agent."""
    return create_agent(use_llm=True)


# Load agent
agent = load_agent()
companies_list = agent.get_companies()

# Sidebar controls
st.sidebar.header("⚙️ Query Options")

show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
show_retrieved = st.sidebar.checkbox("Show Retrieved Data", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🏢 Filter by Company")
company_filter = st.sidebar.selectbox(
    "Select Company (optional)",
    options=["(none)"] + sorted(companies_list),
    index=0
)
if company_filter == "(none)":
    company_filter = None

st.sidebar.markdown("---")
st.sidebar.header("💡 Example Queries")

example_queries = [
    "What is the selection process for Dell?",
    "Companies hiring in Bangalore",
    "Stipend offered by Intel",
    "Skills required for data science roles",
    "Compare Dell and Bosch",
    "List companies with stipend > 40000",
]

for eq in example_queries:
    if st.sidebar.button(eq, key=f"ex_{eq[:15]}", use_container_width=True):
        st.session_state.query_input = eq

st.sidebar.markdown("---")
st.sidebar.header("📊 Stats")
st.sidebar.write(f"Companies in DB: **{len(companies_list)}**")
st.sidebar.write(f"Sample: {', '.join(companies_list[:5])}...")

# Main input
default_query = st.session_state.get("query_input", "")
query = st.text_input(
    "Enter your question about placements/internships",
    value=default_query,
    placeholder="e.g. What is the selection process for Dell?"
)

# Clear the session state after using it
if "query_input" in st.session_state:
    del st.session_state.query_input

submit = st.button("🔍 Ask", type="primary")

if submit and query.strip():
    # Modify query if company filter is set
    actual_query = query.strip()
    if company_filter and company_filter.lower() not in actual_query.lower():
        actual_query = f"{actual_query} for {company_filter}"
    
    with st.spinner("🔄 Retrieving and generating answer..."):
        start_time = time.time()
        
        try:
            response = agent.query(actual_query, verbose=False)
            elapsed = time.time() - start_time
            
            # Display answer
            st.subheader("📝 Answer")
            st.markdown(response.answer)
            
            # Confidence indicator
            conf = response.feedback.confidence_score
            conf_color = "green" if conf >= 0.7 else "orange" if conf >= 0.4 else "red"
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence", f"{conf:.0%}")
            with col2:
                st.metric("Time", f"{elapsed:.2f}s")
            with col3:
                st.metric("Intent", response.plan.intent)
            with col4:
                st.metric("Retries", response.retries)
            
            # Debug info
            if show_debug:
                st.subheader("🔍 Debug Info")
                with st.expander("Query Plan", expanded=False):
                    st.json({
                        "intent": response.plan.intent,
                        "tools": [t.get("tool") for t in response.plan.tools_to_use],
                        "companies_detected": response.plan.companies_mentioned,
                        "attributes": response.plan.attributes_requested,
                        "reasoning": response.plan.reasoning
                    })
                
                with st.expander("Critic Feedback", expanded=False):
                    st.json({
                        "is_complete": response.feedback.is_complete,
                        "is_relevant": response.feedback.is_relevant,
                        "confidence": response.feedback.confidence_score,
                        "reasoning": response.feedback.reasoning,
                        "missing_info": response.feedback.missing_info
                    })
            
            # Retrieved data
            if show_retrieved and response.execution.enriched_results:
                st.subheader(f"📚 Retrieved Data ({len(response.execution.enriched_results)} companies)")
                
                for company, data in response.execution.enriched_results.items():
                    with st.expander(f"🏢 {company.upper()}", expanded=False):
                        # Facts
                        facts = data.get("facts", [])
                        if facts:
                            st.markdown("**📋 Roles & Facts:**")
                            for fact in facts:
                                if isinstance(fact, dict):
                                    role = fact.get('role', fact.get('role_title', 'N/A'))
                                    stipend = fact.get('stipend', fact.get('stipend_salary', {}))
                                    
                                    if isinstance(stipend, dict) and stipend.get('amount'):
                                        st.write(f"• **{role}**: ₹{stipend.get('amount')}/month")
                                    else:
                                        st.write(f"• **{role}**")
                                    
                                    # Additional details
                                    location = fact.get('location', [])
                                    if location:
                                        loc_str = ', '.join(location) if isinstance(location, list) else location
                                        st.caption(f"  📍 Location: {loc_str}")
                                    
                                    elig = fact.get('eligibility', {})
                                    if isinstance(elig, dict):
                                        cgpa = elig.get('cgpa_pg') or elig.get('cgpa_ug')
                                        if cgpa:
                                            st.caption(f"  📚 Min CGPA: {cgpa}")
                        
                        # Semantic data
                        semantic = data.get("semantic", {})
                        
                        if semantic.get("interview_process"):
                            st.markdown("**🎯 Interview Process:**")
                            st.info(semantic["interview_process"][:800] + "..." if len(semantic.get("interview_process", "")) > 800 else semantic["interview_process"])
                        
                        if semantic.get("skills_required"):
                            st.markdown("**🛠️ Skills Required:**")
                            st.info(semantic["skills_required"][:600] + "..." if len(semantic.get("skills_required", "")) > 600 else semantic["skills_required"])
            
            # Tool results
            if show_retrieved:
                for tool_result in response.execution.tool_results:
                    if tool_result.success and tool_result.data:
                        # Aggregation results
                        if "results" in tool_result.data or "companies" in tool_result.data:
                            with st.expander(f"📊 {tool_result.tool_name} Results", expanded=False):
                                if "companies" in tool_result.data:
                                    st.write(f"Found {len(tool_result.data['companies'])} companies")
                                    st.write(", ".join(tool_result.data["companies"][:20]))
                                if "results" in tool_result.data:
                                    st.write(f"Found {len(tool_result.data['results'])} results")
                                    for r in tool_result.data["results"][:10]:
                                        if isinstance(r, dict):
                                            st.write(f"• {r.get('company', 'N/A')} - {r.get('role', r.get('role_title', ''))}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please try rephrasing your question or check if the company name is correct.")

else:
    st.info("💡 Enter a query above and press **Ask** to get started.")
    
    # Show quick stats
    st.markdown("---")
    st.subheader("📊 Database Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Companies", len(companies_list))
    with col2:
        st.metric("Query Types", "4 (Facts, Semantic, Hybrid, Compare)")
    with col3:
        st.metric("LLM Powered", "Yes ✅")
    
    st.markdown("### 🏢 Available Companies")
    st.write(", ".join(sorted(companies_list)))

st.markdown("---")
st.caption("🎓 Placement RAG Assistant • Built with Streamlit & LLM")
