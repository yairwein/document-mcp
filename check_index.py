#!/usr/bin/env python3
"""Check what's in the index."""

import lancedb
import os
from pathlib import Path

# Connect to the database using environment variable or default
lancedb_path = os.getenv("LANCEDB_PATH", "./vector_index")
db = lancedb.connect(str(Path(lancedb_path).expanduser().absolute()))

# Check catalog table
try:
    catalog = db.open_table("catalog")
    df = catalog.to_pandas()
    
    print(f"📚 Total documents indexed: {len(df)}")
    print("\n📁 Documents in index:\n")
    
    # Show all documents
    for idx, row in df.iterrows():
        print(f"{idx+1}. {row['file_name']}")
        # Check if it might be a legal document
        legal_keywords = ['legal', 'agreement', 'contract', 'terms', 'policy', 'license', 'nda', 'confidential', 'privacy']
        name_lower = row['file_name'].lower()
        summary_lower = str(row.get('summary', '')).lower()
        
        is_legal = any(kw in name_lower or kw in summary_lower for kw in legal_keywords)
        if is_legal:
            print(f"   ⚖️  LEGAL DOCUMENT")
            print(f"   Summary: {str(row['summary'])[:150]}...")
            print(f"   Keywords: {row.get('keywords', '')}")
            print()
    
    # Summary
    legal_docs = []
    for idx, row in df.iterrows():
        name_lower = row['file_name'].lower()
        summary_lower = str(row.get('summary', '')).lower()
        if any(kw in name_lower or kw in summary_lower for kw in legal_keywords):
            legal_docs.append(row['file_name'])
    
    print(f"\n📊 Summary:")
    print(f"   Total documents: {len(df)}")
    print(f"   Legal documents found: {len(legal_docs)}")
    
    if legal_docs:
        print("\n⚖️  Legal Documents:")
        for doc in legal_docs:
            print(f"   • {doc}")
            
except Exception as e:
    print(f"Error: {e}")
    print("The index might be empty or not initialized yet.")