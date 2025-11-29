#!/usr/bin/env python3
"""
CLI Example: Ingest a folder into ChromaDB

Usage:
    python ingest_my_code.py /path/to/your/project my_collection
"""

import sys
import requests
import time

API_URL = "http://localhost:8080"

def ingest_folder(folder_path, collection_name, profile="codebase"):
    """Start folder ingestion"""
    print(f"🚀 Starting ingestion of '{folder_path}' into '{collection_name}'...")
    
    response = requests.post(f"{API_URL}/ingest", json={
        "folder_path": folder_path,
        "collection_name": collection_name,
        "profile": profile,
        "recursive": True
    })
    
    if response.status_code != 200:
        print(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
        return None
    
    data = response.json()
    task_id = data["task_id"]
    print(f"✅ Task started: {task_id}")
    return task_id

def poll_task(task_id):
    """Poll task status until completion"""
    print("⏳ Waiting for completion...")
    
    while True:
        response = requests.get(f"{API_URL}/task/{task_id}")
        data = response.json()
        
        status = data.get("status")
        
        if status == "processing":
            progress = data.get("progress", 0)
            current_file = data.get("current_file", "")
            processed = data.get("processed", 0)
            total = data.get("total", 0)
            print(f"📊 Progress: {progress}% - {current_file} ({processed}/{total})")
        
        elif status == "success":
            result = data.get("result", {})
            print(f"\n✅ Ingestion complete!")
            print(f"   Processed: {result.get('processed_files', 0)} files")
            print(f"   Failed: {result.get('failed_files', 0)} files")
            print(f"   Collection: {result.get('collection', 'unknown')}")
            break
        
        elif status == "failed":
            print(f"\n❌ Ingestion failed: {data.get('error', 'Unknown error')}")
            break
        
        time.sleep(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_my_code.py <folder_path> <collection_name> [profile]")
        print("Example: python ingest_my_code.py /home/user/myproject my_codebase codebase")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    collection_name = sys.argv[2]
    profile = sys.argv[3] if len(sys.argv) > 3 else "codebase"
    
    task_id = ingest_folder(folder_path, collection_name, profile)
    if task_id:
        poll_task(task_id)
