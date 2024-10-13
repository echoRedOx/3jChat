import sys
import cmd2
from chat_cli import ChatApp
from agent_library_cli import AgentsLibCli
from ollama_cli import OllamaApp
from handlers.ollama_handler import ollama_pull_model
from utils.utilities import print_dev_stamp, print_main_banner, stream_disco_def, stream_terminal_output
from handlers.agents_handler import Agent


def print_main_menu():
    """
    Pretty sure the name says it all on this one but here is a docstring to appease the linting gods..
    """
    menu_template = """
    _______________________________________________________

    Terminal: Main

    [1] Chat Lobby
    [2] Agents Library
    [3] Ollama Library
    [4] Games Lobby
    
    [9] Exit

    _______________________________________________________
    """
    return print(menu_template)


def main_intro():
    stream_disco_def()
    print_main_banner()
    print_dev_stamp("3Juliet, AI by @technomoonbase (2023)")
    print_main_menu()


def child_break_banner():
    print("\n\n")
    stream_terminal_output("3Juliet, AI: @technomoonbase (2023)", delay=0.01)
    print_main_menu()


class Main(cmd2.Cmd):
    prompt = "3J:Moonbase> "
    intro = main_intro()

    def do_1(self, line):
        print("Entering chat lobby...")
        chat_app = ChatApp()
        chat_app.cmdloop()
        child_break_banner()
    
    def do_2(self, line):
        print("Entering agents library...")
        agents_app = AgentsLibCli()
        agents_app.cmdloop()
        child_break_banner()
    
    def do_3(self, line):
        print("Entering Ollama library...")
        ollama_app = OllamaApp()
        ollama_app.cmdloop()
        child_break_banner()
    
    def do_4(self, line):
        """
        AI Odyssey (2023) - text-based adventure game
        AI Ships (2023) - just another battleship game
        """
        pass        
    
    def do_5(self, line):
        pass
    
    def do_6(self, line):
        child_break_banner()
    
    def do_9(self, line):
        print("Exiting...")
        sys.exit(0)

    def do_exit(self, line):
        print("Exiting...")
        return True  # Returning True exits the application

    def do_help(self, line):
        print("\n\nThe Help Desk is unavailable at this time. We do encourage you to try again later. Thank you for your patience.\n\n")
        print_dev_stamp("DiscoChat by 3Juliet, AI:@technomoonbase (2023)")
        print_main_menu()
        intro = "Welcome to DISCO Chat by 3jai! Type ? for help"

if __name__ == '__main__':
    app = Main()
    app.cmdloop()
