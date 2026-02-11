import os
import subprocess
import sys
import time

# --- CONFIGURATION ---
BATCH_SIZE = 500  # Conservative batch size to avoid memory/timeout issues
MAX_RETRIES = 5
RETRY_DELAY = 10  # Seconds
GIT_BUFFER_SIZE = "524288000"  # 500 MB

def run_command(args, fail_on_error=True, capture_output=True):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            args,
            check=fail_on_error,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            encoding='utf-8',
            errors='replace' # Handle encoding errors gracefully
        )
        return result
    except subprocess.CalledProcessError as e:
        if fail_on_error:
            print(f"[ERROR] Command failed: {' '.join(args)}")
            print(f"Stderr: {e.stderr}")
            sys.exit(1)
        return e

def optimize_git_config():
    print("[INFO] Optimizing Git configuration...")
    # Increase buffer size for large pushes
    run_command(["git", "config", "http.postBuffer", GIT_BUFFER_SIZE])
    run_command(["git", "config", "ssh.postBuffer", GIT_BUFFER_SIZE])
    # Increase window memory
    run_command(["git", "config", "pack.windowMemory", "100m"])
    run_command(["git", "config", "pack.packSizeLimit", "100m"])
    run_command(["git", "config", "pack.threads", "1"]) # Reduce memory usage

def get_pending_files():
    """Get list of untracked and modified files."""
    print("[INFO] Scanning for changes...")
    # --porcelain gives robust parsing
    # -uall shows individual files in untracked directories
    res = run_command(["git", "status", "--porcelain", "-uall"], capture_output=True)
    
    files = []
    lines = res.stdout.strip().splitlines()
    for line in lines:
        if not line: continue
        # Format: XY PATH
        # XY are status codes. Path starts after index 3 usually.
        # But paths with spaces are quoted.
        raw_path = line[3:].strip()
        if raw_path.startswith('"') and raw_path.endswith('"'):
            raw_path = raw_path[1:-1] # Unquote
        files.append(raw_path)
    return files

def push_with_retry():
    """Push current branch with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[PUSH] Attempt {attempt}/{MAX_RETRIES}...")
        res = run_command(["git", "push"], fail_on_error=False, capture_output=False)
        
        if res.returncode == 0:
            return True
        
        print(f"[WARN] Push failed. Retrying in {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)
    
    return False

def main():
    print("=========================================")
    print("   ANTIGRAVITY SMART GIT PUSHER 🚀     ")
    print("=========================================")
    
    # 1. Check for .gitignore
    if not os.path.exists(".gitignore"):
        confirm = input("[WARNING] No .gitignore found! You might push temporary files. Continue? (y/n): ")
        if confirm.lower() != 'y':
            sys.exit(0)
            
    # 2. Optimize
    optimize_git_config()
    
    # 3. Get Files
    files = get_pending_files()
    total_files = len(files)
    print(f"[INFO] Found {total_files} files to process.")
    
    if total_files == 0:
        print("[INFO] No pending files found. Exiting.")
        return

    # 4. Burn-down Loop
    processed_count = 0
    batch_num = 1
    
    # We create a temp file for pathspecs to avoid CLI length limits
    temp_pathspec = "temp_git_batch.txt"
    
    try:
        while processed_count < total_files:
            batch = files[processed_count : processed_count + BATCH_SIZE]
            current_batch_size = len(batch)
            
            print(f"\n--- Batch {batch_num} ({processed_count + 1} to {processed_count + current_batch_size}) ---")
            
            # Write batch to file
            with open(temp_pathspec, 'w', encoding='utf-8') as f:
                f.write('\n'.join(batch))
            
            # Stage
            print(f"[ACTION] Staging {current_batch_size} files...")
            run_command(["git", "add", "--pathspec-from-file=" + temp_pathspec])
            
            # Commit
            commit_msg = f"Auto-commit: Batch {batch_num} ({processed_count+1}-{processed_count+current_batch_size})"
            print(f"[ACTION] Committing: '{commit_msg}'")
            run_command(["git", "commit", "-m", commit_msg])
            
            # Push
            success = push_with_retry()
            if not success:
                print("\n[CRITICAL] Max retries reached. Push failed.")
                print("Suggestion: Check network or remote repo limits.")
                break
            
            processed_count += current_batch_size
            batch_num += 1
            print("[SUCCESS] Batch processed.")
            
    finally:
        # Cleanup
        if os.path.exists(temp_pathspec):
            os.remove(temp_pathspec)

    print("\n=========================================")
    if processed_count == total_files:
        print("   ALL FILES PUSHED SUCCESSFULLY! ✅   ")
    else:
        print("   PROCESS INTERRUPTED ❌              ")
    print("=========================================")

if __name__ == "__main__":
    main()
