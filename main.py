import subprocess
import sys
import os

def run_script(relative_path):
    # Runs a python script located at relative_path using the current Python interpreter.
    
    if not os.path.exists(relative_path):
        print(f"❌ Error: Could not find file '{relative_path}'")
        return

    print(f"\n--- 🚀 Starting {relative_path} ---")
    try:
        subprocess.run([sys.executable, relative_path], check=True)
        print(f"--- ✅ Finished {relative_path} ---")
    except subprocess.CalledProcessError:
        print(f"--- ⚠️  Failed to run {relative_path} ---")

if __name__ == "__main__":
    # Define the paths
    path_sym = os.path.join("asymmetrical", "main.py")
    path_asym = os.path.join("symmetrical", "main.py")

    print("Which process would you like to run?")
    print("1: Asymmetrical")
    print("2: Symmetrical")
    
    # Get user input
    choice = input("\nEnter choice (1 or 2): ").strip()

    # Logic to handle the choice
    if choice == "1":
        run_script(path_sym)
    elif choice == "2":
        run_script(path_asym)
    else:
        print("❌ Invalid selection. Please restart and type 1 or 2.")