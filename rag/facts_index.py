"""Structured facts index for attribute-based queries."""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactsIndex:
    """Structured index for facts-based queries."""
    
    def __init__(self):
        from rag.config import FACTS_FILE, FACTS_INDEX_FILE
        
        self.facts_file = FACTS_FILE
        self.index_file = FACTS_INDEX_FILE
        
        self.facts: List[Dict[str, Any]] = []
        self.df: Optional[pd.DataFrame] = None
        self._company_index: Dict[str, List[int]] = {}
        self._role_index: Dict[str, List[int]] = {}
    
    def load_facts(self, facts_file: Path = None) -> bool:
        """Load facts from JSON file."""
        file_path = facts_file or self.facts_file
        
        if not file_path.exists():
            logger.error(f"Facts file not found: {file_path}")
            return False
        
        logger.info(f"Loading facts from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.facts = json.load(f)
        
        logger.info(f"Loaded {len(self.facts)} facts")
        self._build_indices()
        return True
    
    def _build_indices(self):
        """Build internal indices for fast lookup."""
        logger.info("Building indices...")
        
        # Build company index
        self._company_index = {}
        self._role_index = {}
        
        for idx, fact in enumerate(self.facts):
            # Company index
            company = fact.get('company_name', '').lower()
            if company:
                if company not in self._company_index:
                    self._company_index[company] = []
                self._company_index[company].append(idx)
            
            # Role index
            role = fact.get('primary_key', '')
            if role:
                self._role_index[role] = idx
        
        # Build pandas DataFrame for complex queries
        self._build_dataframe()
        
        logger.info(f"Indexed {len(self._company_index)} companies")
    
    def _build_dataframe(self):
        """Build pandas DataFrame for complex queries."""
        rows = []
        
        for fact in self.facts:
            row = {
                'primary_key': fact.get('primary_key', ''),
                'company_name': fact.get('company_name', ''),
                'role_name': fact.get('role_name', ''),
                'role_title': fact.get('role_title', ''),
                'employment_type': fact.get('employment_type', ''),
                'duration': fact.get('duration', ''),
                'work_mode': fact.get('work_mode', ''),
                'batch_year': fact.get('batch_year', ''),
            }
            
            # Extract stipend
            stipend = fact.get('stipend_salary', {})
            if isinstance(stipend, dict):
                row['stipend_amount'] = self._parse_number(stipend.get('amount', ''))
                row['stipend_currency'] = stipend.get('currency', 'INR')
            else:
                row['stipend_amount'] = self._parse_number(str(stipend))
                row['stipend_currency'] = 'INR'
            
            # Extract eligibility
            elig = fact.get('eligibility', {})
            if isinstance(elig, dict):
                row['cgpa_ug'] = self._parse_number(elig.get('cgpa_ug', ''))
                row['cgpa_pg'] = self._parse_number(elig.get('cgpa_pg', ''))
                row['cgpa_10th'] = self._parse_number(elig.get('cgpa_10th', ''))
                row['cgpa_12th'] = self._parse_number(elig.get('cgpa_12th', ''))
                row['degrees'] = ', '.join(elig.get('degrees', []))
                row['branches'] = ', '.join(elig.get('branches', []))
                row['backlogs'] = elig.get('backlogs', '')
            
            # Extract location
            locations = fact.get('location', [])
            if isinstance(locations, list):
                row['locations'] = ', '.join(locations)
            else:
                row['locations'] = str(locations)
            
            # Selection process rounds
            selection = fact.get('selection_process', [])
            row['num_rounds'] = len(selection) if isinstance(selection, list) else 0
            
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        logger.info(f"DataFrame built with {len(self.df)} rows")
    
    def _parse_number(self, value: Any) -> Optional[float]:
        """Parse numeric value from string."""
        if value is None or value == '':
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # Extract numbers from string
        import re
        numbers = re.findall(r'[\d.]+', str(value).replace(',', ''))
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        return None
    
    def save(self):
        """Save index to disk."""
        logger.info(f"Saving facts index to {self.index_file}")
        data = {
            'facts': self.facts,
            'company_index': self._company_index,
            'role_index': self._role_index
        }
        with open(self.index_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self) -> bool:
        """Load index from disk."""
        if not self.index_file.exists():
            return self.load_facts()
        
        logger.info(f"Loading facts index from {self.index_file}")
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
        
        self.facts = data['facts']
        self._company_index = data['company_index']
        self._role_index = data['role_index']
        self._build_dataframe()
        
        logger.info(f"Loaded {len(self.facts)} facts")
        return True
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_by_company(self, company: str) -> List[Dict[str, Any]]:
        """Get all facts for a company."""
        company_lower = company.lower()
        
        # Exact match first
        if company_lower in self._company_index:
            indices = self._company_index[company_lower]
            return [self.facts[i] for i in indices]
        
        # Partial match
        results = []
        for comp, indices in self._company_index.items():
            if company_lower in comp or comp in company_lower:
                results.extend([self.facts[i] for i in indices])
        
        return results
    
    def get_by_primary_key(self, primary_key: str) -> Optional[Dict[str, Any]]:
        """Get fact by primary key."""
        if primary_key in self._role_index:
            return self.facts[self._role_index[primary_key]]
        return None
    
    def get_all_companies(self) -> List[str]:
        """Get list of all companies."""
        return list(set(f.get('company_name', '') for f in self.facts))
    
    def get_all_stipends(self) -> List[Dict[str, Any]]:
        """Get stipend info for all companies/roles."""
        results = []
        for fact in self.facts:
            stipend = fact.get('stipend_salary', {})
            amount = None
            if isinstance(stipend, dict):
                amount = stipend.get('amount', '')
            else:
                amount = str(stipend)
            
            results.append({
                'company': fact.get('company_name', ''),
                'role': fact.get('role_name', ''),
                'role_title': fact.get('role_title', ''),
                'stipend': amount,
                'primary_key': fact.get('primary_key', '')
            })
        
        return results
    
    def filter_by_stipend(
        self,
        min_amount: float = None,
        max_amount: float = None
    ) -> List[Dict[str, Any]]:
        """Filter companies by stipend range."""
        if self.df is None:
            return []
        
        df = self.df.copy()
        
        if min_amount is not None:
            df = df[df['stipend_amount'] >= min_amount]
        if max_amount is not None:
            df = df[df['stipend_amount'] <= max_amount]
        
        # Get corresponding facts
        primary_keys = df['primary_key'].tolist()
        return [f for f in self.facts if f.get('primary_key', '') in primary_keys]
    
    def filter_by_cgpa(
        self,
        max_cgpa_required: float,
        degree: str = 'pg'  # 'ug' or 'pg'
    ) -> List[Dict[str, Any]]:
        """Filter companies by CGPA requirement."""
        if self.df is None:
            return []
        
        col = f'cgpa_{degree}'
        df = self.df.copy()
        
        # Include entries with no requirement or requirement <= max
        df = df[(df[col].isna()) | (df[col] <= max_cgpa_required)]
        
        primary_keys = df['primary_key'].tolist()
        return [f for f in self.facts if f.get('primary_key', '') in primary_keys]
    
    def filter_by_location(self, location: str) -> List[Dict[str, Any]]:
        """Filter companies by location."""
        if self.df is None:
            return []
        
        location_lower = location.lower()
        df = self.df[self.df['locations'].str.lower().str.contains(location_lower, na=False)]
        
        primary_keys = df['primary_key'].tolist()
        return [f for f in self.facts if f.get('primary_key', '') in primary_keys]
    
    def filter_by_branch(self, branch: str) -> List[Dict[str, Any]]:
        """Filter companies by eligible branch."""
        if self.df is None:
            return []
        
        branch_lower = branch.lower()
        df = self.df[self.df['branches'].str.lower().str.contains(branch_lower, na=False)]
        
        primary_keys = df['primary_key'].tolist()
        return [f for f in self.facts if f.get('primary_key', '') in primary_keys]
    
    def search_attribute(
        self,
        attribute: str,
        value: str = None,
        companies: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for specific attribute across companies.
        
        Args:
            attribute: Attribute to search (stipend, cgpa, location, etc.)
            value: Optional filter value
            companies: Optional list of companies to search
        
        Returns:
            List of {company, role, attribute_value}
        """
        results = []
        
        facts_to_search = self.facts
        if companies:
            companies_lower = [c.lower() for c in companies]
            facts_to_search = [
                f for f in self.facts 
                if f.get('company_name', '').lower() in companies_lower
            ]
        
        attribute_map = {
            'stipend': lambda f: f.get('stipend_salary', {}),
            'location': lambda f: f.get('location', []),
            'duration': lambda f: f.get('duration', ''),
            'cgpa': lambda f: f.get('eligibility', {}).get('cgpa_pg', ''),
            'branches': lambda f: f.get('eligibility', {}).get('branches', []),
            'selection_process': lambda f: f.get('selection_process', []),
            'work_mode': lambda f: f.get('work_mode', ''),
            'apply_before': lambda f: f.get('apply_before', ''),
        }
        
        getter = attribute_map.get(attribute.lower())
        if not getter:
            # Try direct attribute access
            getter = lambda f, attr=attribute: f.get(attr, '')
        
        for fact in facts_to_search:
            attr_value = getter(fact)
            
            results.append({
                'company': fact.get('company_name', ''),
                'role': fact.get('role_name', ''),
                'role_title': fact.get('role_title', ''),
                'primary_key': fact.get('primary_key', ''),
                attribute: attr_value
            })
        
        return results
    
    def compare_companies(
        self,
        companies: List[str],
        attributes: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple companies on specified attributes.
        
        Args:
            companies: List of company names
            attributes: List of attributes to compare
        
        Returns:
            DataFrame with comparison
        """
        if attributes is None:
            attributes = ['stipend_amount', 'cgpa_pg', 'locations', 'num_rounds', 'work_mode']
        
        if self.df is None:
            return pd.DataFrame()
        
        # Filter companies
        companies_lower = [c.lower() for c in companies]
        df = self.df[self.df['company_name'].str.lower().isin(companies_lower)]
        
        # Select columns
        cols = ['company_name', 'role_name', 'role_title'] + [
            a for a in attributes if a in self.df.columns
        ]
        
        return df[cols]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.df is None:
            return {}
        
        return {
            "total_entries": len(self.facts),
            "total_companies": len(self._company_index),
            "companies": list(self._company_index.keys()),
            "avg_stipend": self.df['stipend_amount'].mean(),
            "max_stipend": self.df['stipend_amount'].max(),
            "min_stipend": self.df['stipend_amount'].min(),
            "locations": self.df['locations'].value_counts().to_dict(),
        }
