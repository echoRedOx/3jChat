import socket
import subprocess


class OllamaServer:
    def __init__(self):
        self.process = None

    def start_server(self, port):
        command = f"OLLAMA_HOST=127.0.0.1:{port} ollama serve"
        try:
            self.process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"Ollama server started on port {port}")
        except Exception as e:
            print(f"Error starting Ollama server on port {port}: {e}")

    def stop_server(self):
        if self.process is None:
            print("No server process to stop.")
            return

        # Gracefully terminate the process
        self.process.terminate()

        # Wait for a bit to see if it shuts down
        try:
            self.process.wait(timeout=5)  # Wait for 5 seconds
        except subprocess.TimeoutExpired:
            print("Server did not terminate gracefully, forcefully stopping it.")
            self.process.kill()

        print("Server process stopped.")
        self.process = None

    @staticmethod
    def find_available_port(start_port=4200, end_port=4300):
        for port in range(start_port, end_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return port
                except socket.error:
                    continue
        raise ValueError(f"No available ports found in the range {start_port}-{end_port}")

def ollama_pull_model(model_name: str) -> None:
    """
    Download a model from the Ollama server.

    :param model_name: The name of the model to download.
    """
    subprocess.run(['ollama', 'pull', model_name])

# remove model
def ollama_remove_model(model_name: str) -> None:
    """
    Remove a model from your local Ollama repo.

    :param model_name: The name of the model to remove.
    """
    subprocess.run(['ollama', 'rm', model_name])

# copy model
def ollama_copy_model(src_model_name: str, dest_model_name: str) -> None:
    """
    Copies a model from one name to another. Intended for use during fine-tuning or new model creation.

    :param src_model_name: The name of the model to copy.
    :param dest_model_name: The name of the new model.
    """
    subprocess.run(['ollama', 'cp', src_model_name, dest_model_name])

# create model from Modelfile
def ollama_create_model_from_modelfile(model_name: str) -> None:
    """
    Uses the Modelfile to create a new model.

    :param model_name: The name of the model to create.
    """
    with open(f'agents/{model_name}/Modelfile', 'w') as f:
        modelfile = f.read()
    subprocess.run(['ollama', 'create', model_name])


def ollama_list_downloaded_models() -> None:
    """
    Lists all the models you have downloaded from the Ollama server.
    """
    subprocess.run(['ollama', 'list'])