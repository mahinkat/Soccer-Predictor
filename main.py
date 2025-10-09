import subprocess
import sys
import os

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_menu():
    clear_screen()
    print("=" * 50)
    print("      SOCCER LEAGUE SIMULATOR")
    print("=" * 50)
    print("\nSelect a league to simulate:\n")
    print("  1. Bundesliga")
    print("  2. La Liga")
    print("  3. Ligue 1")
    print("  4. Premier League")
    print("  5. Serie A")
    print("  6. Exit")
    print("\n" + "=" * 50)

def run_simulation(script_name):
    try:
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        script_path = os.path.join(base_dir, script_name)
        
        print(f"\nRunning {script_name}...\n")
        
        if not os.path.exists(script_path):
            print(f"\nError: {script_name} not found!")
            input("\nPress Enter to continue...")
            return
        
        print("Starting simulation (this may take 30-60 seconds)...\n")
        print("-" * 50)
    
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, 
            text=True,
            cwd=base_dir 
        )
        
        print("-" * 50)
        

        if result.stdout:
            print(result.stdout)
            
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("\n✓ Simulation completed successfully!")
        else:
            print(f"\n✗ Simulation failed (return code: {result.returncode})")
            
        input("\nPress Enter to continue...")
        
    except Exception as e:
        print(f"\nError running simulation: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to continue...")

def main():
    league_scripts = {
        '1': ('bundesliga.py', 'Bundesliga'),
        '2': ('laliga.py', 'La Liga'),
        '3': ('ligue1.py', 'Ligue 1'),
        '4': ('premier.py', 'Premier League'),
        '5': ('seriea.py', 'Serie A')
    }
    
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '6':
            clear_screen()
            print("\nThank you for using Soccer League Simulator!")
            print("Goodbye!\n")
            sys.exit(0)
        
        if choice in league_scripts:
            script, league_name = league_scripts[choice]
            clear_screen()
            print(f"\n{league_name} Simulation Selected")
            print("=" * 50)
            run_simulation(script)
        else:
            print("\nInvalid choice! Please enter a number between 1 and 6.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()