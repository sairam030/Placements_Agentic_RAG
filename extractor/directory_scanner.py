"""Scanner to discover placement folders and their files."""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlacementEntry:
    """Represents a single placement entry (company + role)."""
    company_name: str
    role_name: Optional[str]  # None if no specific role subfolder
    folder_path: Path
    files: List[Path] = field(default_factory=list)
    batch_year: Optional[str] = None
    
    @property
    def primary_key(self) -> str:
        """Generate a unique primary key for this entry."""
        if self.role_name:
            return f"{self.company_name}_{self.role_name}"
        return self.company_name


def parse_folder_name(folder_name: str) -> tuple:
    """Parse company name and batch year from folder name."""
    # Pattern: CompanyName_MTech_2026 or CompanyName_MTech_2026_
    pattern = r'^(.+?)(?:_MTech_(\d{4}))?_?$'
    match = re.match(pattern, folder_name)
    
    if match:
        company = match.group(1).replace('_', ' ').strip()
        year = match.group(2)
        return company, year
    
    return folder_name.replace('_', ' '), None


def scan_placements_directory(base_path: Path) -> List[PlacementEntry]:
    """
    Scan the placements directory and return list of placement entries.
    Handles both flat structure and Role-based subfolders.
    """
    entries = []
    
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        return entries
    
    for company_folder in base_path.iterdir():
        if not company_folder.is_dir():
            continue
        
        # Skip Info folder
        if company_folder.name == "Info":
            continue
        
        company_name, batch_year = parse_folder_name(company_folder.name)
        
        # Check if there are Role subfolders
        role_folders = [
            f for f in company_folder.iterdir() 
            if f.is_dir() and (f.name.lower().startswith('role') or f.name.lower().startswith('role -'))
        ]
        
        if role_folders:
            # Process each role as a separate entry
            for role_folder in role_folders:
                role_name = role_folder.name
                files = collect_files(role_folder)
                
                entry = PlacementEntry(
                    company_name=company_name,
                    role_name=role_name,
                    folder_path=role_folder,
                    files=files,
                    batch_year=batch_year
                )
                entries.append(entry)
                logger.info(f"Found: {entry.primary_key} with {len(files)} files")
        else:
            # Single entry without role subfolders
            files = collect_files(company_folder)
            
            entry = PlacementEntry(
                company_name=company_name,
                role_name=None,
                folder_path=company_folder,
                files=files,
                batch_year=batch_year
            )
            entries.append(entry)
            logger.info(f"Found: {entry.primary_key} with {len(files)} files")
    
    return entries


def collect_files(folder: Path) -> List[Path]:
    """Collect all relevant files from a folder (including nested)."""
    supported_extensions = {
        '.txt', '.pdf', '.docx', '.doc', '.pptx', '.ppt',
        '.xlsx', '.xls', '.png', '.jpg', '.jpeg'
    }
    
    files = []
    for item in folder.rglob('*'):
        if item.is_file() and item.suffix.lower() in supported_extensions:
            # Skip certain files that are not job descriptions
            skip_patterns = ['seating', 'cv format', 'test_df', 'train_df', 'predictions']
            if not any(p in item.name.lower() for p in skip_patterns):
                files.append(item)
    
    return files
