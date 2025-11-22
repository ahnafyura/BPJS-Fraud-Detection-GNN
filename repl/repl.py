import sys
from gnn import hybrid_gnn
from louvain import louvain
from etl import load
from etl import export
from config import env
from getpass import getpass
import os

class Menu:
    def __init__(self, name, text, runner):
        self.name = name
        self.text = text
        self.runner = runner
    
    def run(self, arg):
        self.runner(arg)
    
    def display(self):

        print(self.name)
        print(self.text)

def clear_terminal():
    # Clear terminal
    # Windows
    if os.name == 'nt':
        os.system('cls')
    # macOS / Linux
    else:
        os.system('clear')


current_menu = None

def run_main_menu(arg):
    etc = "\nEnter to Continue..."
    cctc = "Ctrl+C to Cancel"

    if (arg == "1"):
        load.load_data()
        louvain.run()
        export.export_data()
        hybrid_gnn.run()

        input(etc)

    elif (arg == "2"):
        load.load_data()

        input(etc)
        
    elif (arg == "3"):
        export.export_data()

        input(etc)

    elif (arg == "4"):
        louvain.run()

        input(etc)

    elif (arg == "5"):
        hybrid_gnn.run()

        input(etc)

    elif (arg == "6"):
        print(cctc)
        try:
            env.url = input("Enter URL: ")
            env.uname = input("Enter username: ")
            env.pw = getpass("Enter password: ")
        except KeyboardInterrupt:
            print("\nCancelled")
    
    elif (arg == "7"):
        print(cctc)
        try:
            env.raw_input_data = input("Enter directory:")

        except KeyboardInterrupt:
            print("\nCancelled")
    
    elif (arg == "8"):
        while True:
            print(cctc)
            try:
                env.test_size = max(0.1, min(.9, float(input("Enter a number [0.1-.9]:"))))
                break
            
            except KeyboardInterrupt:
                print("\nCancelled")
                break

            except ValueError:
                continue

    elif (arg == "9"):
        env.skip_gnn_training = not env.skip_gnn_training

    else:
        print("Invalid Input!")
main_menu = Menu("Main Menu", 
    f"""

    1. Run Entire Pipeline
    2. Load Raw Data to Neo4j
    3. Export Neo4j Graph to CSV
    4. Run Louvain on Current Graph
    5. Run GNN using Exported Data
    6. Change Neo4j Database
    7. Change raw_input_data Directory
    8. Set test_size
    9. Toggle skip_gnn_training

""", 
    lambda arg: run_main_menu(arg)

)

current_menu = main_menu

if __name__ == "__main__":
    clear_terminal()

    while True:
        try:
            print(f"""

            ░██████╗░██████╗░░█████╗░███████╗░█████╗░███╗░░██╗░█████╗░
            ██╔════╝░██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗░██║██╔══██╗
            ██║░░██╗░██████╔╝███████║█████╗░░███████║██╔██╗██║███████║
            ██║░░╚██╗██╔══██╗██╔══██║██╔══╝░░██╔══██║██║╚████║██╔══██║
            ╚██████╔╝██║░░██║██║░░██║██║░░░░░██║░░██║██║░╚███║██║░░██║
            ░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝
            
            Licensed under the terms in LICENSE (MIT)



    Connected to Neo4j
        url: {env.url}
        uname: {env.uname}
        password: HIDDEN
            
    Data Variables
        raw_input_data: {env.raw_input_data}
            - The input CSV data
        test_size: {env.test_size}
            - The testing data ratio for train-test splitting on training epochs
        skip_gnn_training: {env.skip_gnn_training}
            - Do not train if true. Use existing model instead on {env.BEST_GNN_HYBRID_PATH} if exists.
        
            
            """)

            current_menu.display()
            user_input = input("Input:")

            current_menu.run(user_input)

            clear_terminal()

        except KeyboardInterrupt:
            try:
                if (input("\n\nExit? (Y/n): ").lower() == "n"):
                    clear_terminal()
                    continue
                else:
                    sys.exit()
            except KeyboardInterrupt:
                sys.exit()