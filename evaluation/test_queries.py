"""Test queries for evaluating the placement agent."""

# Test cases: (query, expected_type, key_info_to_check)
TEST_QUERIES = [
    # Aggregation queries
    {
        "query": "How many companies are hiring in Bangalore?",
        "type": "aggregation",
        "expected": ["count", "companies", "bangalore"],
        "category": "location_filter"
    },
    {
        "query": "List all companies with stipend more than 40000",
        "type": "aggregation", 
        "expected": ["stipend", "40000"],
        "category": "stipend_filter"
    },
    {
        "query": "Which companies have CGPA requirement less than 7?",
        "type": "aggregation",
        "expected": ["cgpa", "7"],
        "category": "cgpa_filter"
    },
    
    # Company-specific queries
    {
        "query": "What is the selection process for Dell?",
        "type": "company_detail",
        "expected": ["dell", "selection", "interview", "rounds"],
        "category": "selection_process"
    },
    {
        "query": "What is the stipend offered by Intel?",
        "type": "company_detail",
        "expected": ["intel", "stipend", "INR"],
        "category": "stipend"
    },
    {
        "query": "What skills are required for Bosch internship?",
        "type": "company_detail",
        "expected": ["bosch", "skills"],
        "category": "skills"
    },
    {
        "query": "Tell me about Amazon internship eligibility",
        "type": "company_detail",
        "expected": ["amazon", "eligibility", "cgpa"],
        "category": "eligibility"
    },
    
    # Comparison queries
    {
        "query": "Compare Dell and Intel internships",
        "type": "comparison",
        "expected": ["dell", "intel", "stipend"],
        "category": "comparison"
    },
    
    # General queries
    {
        "query": "Which company offers the highest stipend?",
        "type": "aggregation",
        "expected": ["highest", "stipend"],
        "category": "ranking"
    },
    {
        "query": "What are the best companies for data science roles?",
        "type": "general",
        "expected": ["data science"],
        "category": "role_search"
    },
]
