import time
import subprocess
import sys
from datetime import datetime

while True:
    print("\n----------------------------------------") 
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running Bluesky collector...") 
    print("----------------------------------------")
    subprocess.run([sys.executable, "src/blueskyToMongoBackfill.py"])
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done. Sleeping for 1 hour.")
    time.sleep(3600)
