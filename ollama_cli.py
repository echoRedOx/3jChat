import cmd2
from handlers.ollama_handler import ollama_list_downloaded_models, ollama_pull_model, ollama_remove_model
from utils.utilities import print_dev_stamp, toilet_banner_metal


def print_ollama_menu():
    """
    Menu for the Ollama Library. 
    """
    ollama_menu_template = """
    _______________________________________________________

    Terminal: Ollama Library

    [1] List Downloaded Models
    [2] Download New Model
    [3] Remove Downloaded Model
    [4] Create Model from Modelfile

    [9] Back to main menu (or type 'back' or 'main')

    _______________________________________________________
    """
    return print(ollama_menu_template)


def print_ollama_intro():
    print("\n\n")
    toilet_banner_metal("Ollama Library")
    print_ollama_menu()


class OllamaApp(cmd2.Cmd):
    def __init__(self):
        super().__init__()
        self.prompt = "3J:Ollama>"
        self.intro = print_ollama_intro()

    def do_1(self, line):
        print("\nHere are the models you have available with Ollama...\n\n")
        ollama_list_downloaded_models()
        print("\n\n")
    
    def do_2(self, line):
        try:
            # Download a new model TODO: this needs to run in another thread or background and reverse to main at exec
            ollama_pull_model(input("Enter the name of the model to download: "))
        except Exception as e:
            print(f"Error: {e}\n\n")

    def do_3(self, line):
        model_name = input("Enter the name of the model to remove: ")
        ollama_remove_model(model_name)
        print(f"\nModel {model_name} removed.\n\n")

    def do_4(self, line):
        pass

    def do_9(self, line):
        print("\nHeading back to base...")
        return True
    
    def do_main(self, line):
        print("\nHeading back to base...")
        return True

    def do_back(self, line):
        print("\nHeading back to base...")
        return True  # Returning True exits the application