import cmd2
from utils.utilities import print_dev_stamp, toilet_banner_metal
from handlers.agents_handler import Agent, ModelInstructions
from handlers.library_handler import Curator


def print_agentslib_menu():
    """
    Menu for the agents lobby.
    """
    agentslib_menu_template = """
    _______________________________________________________

    Terminal: Agents Library

    [1] List Existing Agents
    [2] Update an Agent's Instructions
    [3] Create a new Agent

    [9] Back to main menu (or type 'back' or 'main')

    _______________________________________________________
    """
    return print(agentslib_menu_template)


def print_agentslib_intro():
    print("\n\n")
    toilet_banner_metal("AGENTS\nLIBRARY")
    print_agentslib_menu()

class AgentsLibCli(cmd2.Cmd):
    def __init__(self):
        super().__init__()
        self.prompt = "3J:Agents>"
        self.intro = print_agentslib_intro()

    def do_1(self, line):
        print("Getting agent(s) info...")
        curator = Curator()
        curator.list_existing_agents()
        print_agentslib_menu()
    
    def do_2(self, line):
        # Update agent instructions
        try:
            agent_name = input("Enter the name of the agent to update: ")
            instructions = ModelInstructions(agent_name.lower(), "load")
            instructions.update_model_instructions()
        except Exception as e:
            print(f"Error: {e}")

    def do_3(self, line):
        try:
            print("Agent creator started...")
            curator = Curator()
            curator.create_new_agent()
            print("Agent creator exited...\n\n")
            print_agentslib_menu()
        except Exception as e:
            print(f"Error: {e}")

    def do_9(self, line):
        print("Heading back to base...")
        return True
    
    def do_main(self, line):
        print("Heading back to base...")
        return True

    def do_back(self, line):
        print("Heading back to base...")
        return True  # Returning True exits the application
    
    def do_menu(self, line):
        print_agentslib_menu()
