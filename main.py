import subprocess
import sys
import os

def main():
    # Define your scripts in order
    scripts = [
        "db1_fetch_from_web.py",
        "db2_fetch_content.py",
        "db3_sent_content_to_gemini.py",
        os.path.join("rag", "embedding.py")
    ]

    for script in scripts:
        print(f"Running {script}...")
        try:
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in {script}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()