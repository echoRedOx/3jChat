from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
from string import Template
from uuid import uuid4
import requests
import yaml
from handlers.ollama_handler import OllamaServer
from utils.utilities import chroma_results_format_to_prompt, debug_print_function_return, message_cache_format_to_prompt, stream_agent_response, toilet_banner_metal, toilet_banner_plain
from handlers.chroma_handler import chroma_get_or_create_collection, chroma_query_collection, chroma_upsert_to_collection
from handlers.conversation_handler import MessageCache, start_new_conversation, Message, Turn


@dataclass
class ModelInstructions:
    """
    Agent configuration dataclass for managing prompts and messaging to the agent.
    """
    name: str = None
    description: str = None
    llm_model: str = None
    system_message: str = None
    assistant_intro: str = None
    assistant_focus: str = None
    commands: dict = None
    prompt_script: str = None
    start_token: str = None
    end_token: str = None
    mem_start_token: str = None
    mem_end_token: str = None
    chat_start_token: str = None
    chat_end_token: str = None
    completions_url: str = None

    def __init__(self, method: str, assistant_name: str = None) -> None:
        """
        Model instructions init takes a method param as ['create', 'load'] to determine if the instructions should be loaded from a yaml file or created from the CLI.

        :param method: The method to use to create the instructions.
        """
        if method == 'load':
            if assistant_name:
                self.load_from_yaml(assistant_name)
                print(f"Loaded instructions for {self.name}")
                print(asdict(self))
            else:
                print("Error: No assistant name provided.")    
        elif method == 'create':
            self.load_defaults_from_yaml()
            print("Creating new assistant instructions...")
            customize = input("Would you like to customize the instructions? (y/n): ").strip()
            if customize == 'y':
                instructions = self.to_dict()
                for key, value in instructions.items():
                    new_value = input(f"\n{key} ({value}): Press enter to keep current value or enter a new one: ").strip()
                    if new_value:
                        setattr(self, key, new_value)
                print(asdict(self))
            else:
                print("Using default instructions. You can customize these later.")
                print(asdict(self))
            
            # Create the agent directories and populate them with the default files
            agents_dir = Path('agents/')
            templates_dir = Path('agent-templates/')
            templates = [file for file in templates_dir.iterdir() if file.suffix in ['.md', '.yml', '.yaml', '.txt']]
            # this gives the ability to default specifically named blank files and types for template inclusion
            include_files = []  

            directories = [
                'fine-tuning'
            ]

            try:
                print('Checking for agents directory...')
                if not agents_dir.exists():
                    os.mkdir(agents_dir)
                    print('----------------------------------------')
                    print('Directory (agents) created')
                
                print('Cross-checking for existing agents...')
                target_agent_dir = Path(f'agents/{self.name.lower()}')
                if target_agent_dir.exists():
                    print('----------------------------------------')
                    print(f'Agent ({self.name}) already exists. Pleae choose another name.')
                    return None
                else:
                    print('Agent does not exist, creating...')
                    target_agent_dir.mkdir(parents=True, exist_ok=True)
                    print('----------------------------------------')
                    print(f'Agent Directory ({self.name}) created')

                for directory in directories:
                    Path(f'agents/{self.name.lower()}/{directory}').mkdir(parents=True, exist_ok=True)
                    print('----------------------------------------')
                    print(f'Agent Sub-Directory ({self.name}/{directory}) created')
                
                # Copy the project template files
                for template in templates:
                    shutil.copy(template, f"{agents_dir}/{self.name.lower()}/{template.name}")
                    print(f"Copied {template} to {agents_dir}/{self.name.lower()}/{template.name}")

                print('----------------------------------------')
                print("All template files copied to new project")
                print('----------------------------------------')

                self.save_to_yaml()

            except Exception as e:
                print(e)
                return


    def to_dict(self) -> dict:
        """
        Export config class to a base dict

        :returns: Base dictionary for the config class.
        """
        return asdict(self)
    
    def print_model_instructions(self) -> None:
        """
        Print the config to the terminal.

        :returns: Prints a pre-defined config string to the terminal.
        """
        print(f"Agent Configuration:\n{self.to_dict()}")
    
    def to_prompt_script(self) -> str:
        """
        Export config class to a base dict

        :returns: Base dictionary for the config class.
        """
        return (
            f"{self.start_token}System: \n"
            f"{self.system_message}{self.end_token}\n"
            f"{self.start_token}Assistant: \n"
            f"{self.assistant_intro}{self.end_token}\n"
            f"{self.start_token}User: \n"
            f"Your current focus should be: {self.assistant_focus}{self.end_token}\n"
            f"{self.mem_start_token}Context from memory: "
            f"$context{self.mem_end_token}\n"
            f"Chat History: \n"
            f"$history\n"
            f"{self.start_token}$username: \n"
            f"$user_input{self.end_token}\n"
            f"{self.start_token}{self.name}: \n"
        )
    
    def update_model_instructions(self) -> None:
        """
        Iterate through the config and update the values or keep current.
        """
        instructions = self.to_dict()
        for key, value in instructions.items():
            new_value = input(f"\n{key} ({value}): Press enter to keep current value or enter a new one: ").strip()
            if new_value:
                setattr(self, key, new_value)
        self.save_to_yaml()
    
    def load_defaults_from_yaml(self) -> None:
        """
        Load the agent instructions config from a yaml file.
        """
        model_instructions = Path(f"agent-templates/instructions.yaml")
        if model_instructions.exists():
            with model_instructions.open('r') as file:
                instructions = yaml.safe_load(file)
                for key, value in instructions.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

    def load_from_yaml(self, assistant_name: str) -> None:
        """
        Load the agent instructions config from a yaml file.
        """
        model_instructions = Path(f"agents/{assistant_name.lower()}/instructions.yaml")
        if model_instructions.exists():
            with model_instructions.open('r') as file:
                instructions = yaml.safe_load(file)
                for key, value in instructions.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def save_to_yaml(self) -> None:
        """
        Save the agent config to a yaml file.

        :returns: Saves the agent config to a yaml file.
        """
        data = self.to_dict()
        with open(f"agents/{self.name.lower()}/instructions.yaml", "w") as f:
            yaml.safe_dump(data, f)


@dataclass
class ParamsConfig:
    """
    Agent configuration dataclass for tweaking completion parameters. More are available through Ollama's API, I will build this out to cover it all eventually. Parameter definitions from Ollama and their defaults values are given in params. Class field defaults are values that I have found to work well for my use cases.

    :param temperature: The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
    :param num_ctx: Sets the size of the context window used to generate the next token. (Default: 4096)
    :param num_gpu: The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable. (Default: 50)
    :param num_thread: Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). (Default: 8, I run 16 on a core i9)
    :param top_k: Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
    :param top_p: Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
    :param num_predict: The number of tokens to generate. (Default: 128)
    :param seed: The seed to use for random number generation. (Default: 0)
    :param mirostat: Enables the Mirostat algorithm. (Default: 0)
    :param mirostat_eta: The learning rate for the Mirostat algorithm. (Default: 0.1)
    :param mirostat_tau: The temperature for the Mirostat algorithm. (Default: 5.0)
    :param repeat_last_n: The number of tokens to repeat at the end of the context. (Default: 64)
    :param completions_url: The URL of the Ollama API endpoint.
    :param completion_headers: The headers to send with the request to the Ollama API.
    :param start_token: The token to use to start the prompt.
    :param end_token: The token to use to end the prompt.
    :param tfs_z: The number of tokens to use for the TFS-Z algorithm. (Default: 0)
    :creates: Param config object for the agent.
    """
    temperature: float = None
    num_ctx: int = None
    num_gpu: int = None
    num_thread: int = None
    top_k: int = None
    top_p: float = None
    num_predict: int = None
    seed: int = None
    mirostat: int = None
    mirostat_eta: float = None
    mirostat_tau: float = None
    repeat_last_n: int = None
    tfs_z: int = None
    assistant_name: str = None

    def __init__(self, method: str, assistant_name: str) -> None:
        """
        Model Params Config init takes a method param as ['create', 'load'] to determine if the instructions should be loaded from a yaml file or created from the CLI.

        :param method: The method to use to create the instructions.
        """
        self.assistant_name = assistant_name
        if method == 'load':
            self.load_from_yaml()
            print(f"Loaded param config for {assistant_name}")
            print(asdict(self))
        elif method == 'create':
            self.load_defaults_from_yaml()
            print("Creating new completion parameters configuration...")
            customize = input("Would you like to customize the model params? (y/n): ").strip()
            if customize == 'y':
                self.update_model_params()
                print(asdict(self))
            else:
                print("Using default completion parameters. You can customize these later.")
                print(asdict(self))
        else:
            print("Error: Invalid method. Please use 'create' or 'load'.")
    
    def to_dict(self) -> dict:
        """
        Export config class to a base dict

        :returns: Base dictionary for the config class.
        """
        return asdict(self)
    
    def update_model_params(self) -> None:
        """
        Iterate through the config and update the values or keep current.
        """
        params = self.to_dict()
        for key, value in params.items():
            new_value = input(f"\n{key} ({value}): Press enter to keep current value or enter a new one: ").strip()
            if new_value:
                setattr(self, key, new_value)
        self.save_to_yaml(self.assistant_name)
    
    def print_config(self) -> None:
        """
        Print the config to the terminal.

        :returns: Prints a pre-defined config string to the terminal.
        """
        print(f"Agent Configuration:\n{self.to_dict()}")
    
    def cli_create_config(self) -> None:
        """
        Create a config from the CLI.
        """
        params = self.to_dict()
        for key, value in params.items():
            new_value = input(f"\n{key} ({value}): Press enter to keep current value or enter a new one: ").strip()
            if new_value:
                params[key] = new_value
    
    def load_from_yaml(self) -> None:
        """
        Load the agent instructions config from a yaml file.
        """
        params_config = Path(f"agents/{self.assistant_name.lower()}/params_config.yaml")
        if params_config.exists():
            with params_config.open('r') as file:
                completion_params = yaml.safe_load(file)
                for key, value in completion_params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def load_defaults_from_yaml(self) -> None:
        """
        Load the agent instructions config from a yaml file.
        """
        params_config = Path(f"agent-templates/params_config.yaml")
        if params_config.exists():
            with params_config.open('r') as file:
                completion_params = yaml.safe_load(file)
                for key, value in completion_params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def save_to_yaml(self) -> None:
        """
        Save the agent config to a yaml file.

        :returns: Saves the agent config to a yaml file.
        """
        data = self.to_dict()
        with open(f"/agents/{self.assistant_name.lower()}/params_config.yaml", "w") as f:
            yaml.safe_dump(data, f)


class Agent:
    name: str
    params_config: ParamsConfig
    instructions: ModelInstructions
    message_cache: MessageCache
    last_response: str

    def __init__(self, params_config: ParamsConfig, instructions: ModelInstructions) -> None:
        """
        Agent init takes a params_config and instructions object to create the agent.

        :param params_config: The parameters configuration for the agent.
        :param instructions: The instructions for the agent.
        """
        self.params_config = params_config
        self.instructions = instructions
        self.name = self.instructions.name
        self.message_cache = MessageCache(20)
        self.last_response = None

    def build_prompt(self, user_input: str, username: str, agent_agent: bool) -> str:
        """
        Builds a prompt dynamically based on a template and user input.Parses a predefined prompt template to identify placeholders as $param. then substitute these placeholders with corresponding values from the class's instructions or other relevant sources. 

        :param user_input: (str) The user's input text to be included in the prompt.
        :returns: (str) A formatted prompt string with the necessary substitutions made.
        """
        # Pull the prompt template
        prompt_template = self.instructions.to_prompt_script()
        collection = chroma_get_or_create_collection(f"{self.name}-{username}")

        if agent_agent == True:
            formatted_chroma_results = None
        else:
            chroma_results = chroma_query_collection(collection, user_input, 5)
            formatted_chroma_results = chroma_results_format_to_prompt(chroma_results)

        message_history = self.message_cache.get_message_cache()
        message_cache_formatted = message_cache_format_to_prompt(self, message_history)
        print(f"Message Cache Formatted: {message_cache_formatted}")

        # all possible substitutions
        substitutions = {
            "history": message_cache_formatted,
            "user_input": user_input,
            "username": username,
            "context": formatted_chroma_results,
        }
        # Identify the parameters in the template
        params_in_template = set(re.findall(r'\$(\w+)', prompt_template))

        # Create a dictionary with only the necessary substitutions
        required_substitutions = {key: substitutions[key] for key in params_in_template if key in substitutions}

        # Substitute the values into the template
        template = Template(prompt_template)

        return template.safe_substitute(required_substitutions)
    
    def generate_response(self, prompt: str, server_port: int) -> str:
        """
        Generates an http request to the ollama server and returns the response.

        :param prompt: The prompt to send to the model.
        :return: The response from the model.
        """
        data = {
            "model": self.instructions.llm_model,
            "stream": False,
            "prompt": prompt,
            "options": {
                "temperature": self.params_config.temperature,
                "num_ctx": self.params_config.num_ctx,
                "num_gpu": self.params_config.num_gpu,
                "num_thread": self.params_config.num_thread,
                "top_k": self.params_config.top_k,
                "top_p": self.params_config.top_p,
            }
        }

        completion_headers = {
            'Content-Type': 'application/json',
        }

        url = f"http://127.0.0.1:{server_port}/api/generate"
        try:
            response = requests.post(url, headers=completion_headers, data=json.dumps(data))
            # print(f"Response: {response}")
            if response.status_code == 200:
                response_text = response.text
                data = json.loads(response_text)
                response_content = data["response"]

                return response_content
        except Exception as e:
            print(f'Error: {e}')


class ChatHandler:
    """
    With multiple chat formats, it makes sense to kick this to it's own class to keep things tidy. Supports User>Agent chat and Agent>Agent chat currently. Looking at integrating a pub sub library so num of participants is arbitrary. The logic for round robin with agents is a little trickier to flesh out and maintain a consistent flow. Agent>Agent chat still tends to convert to mimicry after 12 to 15 rounds but i am hoping improvements in source will fix this along with logic to filter, limit or remove chroma results from prompt which has shown good results in testing but limits the functionality and overall scope.
    """

    def chat_with_agent(self, assistant_name: str) -> None:
        """
        Opens a chat session and starts a new conversation with the selected agent. Chroma collection is created with agent:user nomencalture to refine results. 

        :param project: The name of the project to chat about. If None, chat about all projects.
        """
        instructions = ModelInstructions(method='load', assistant_name=assistant_name)
        
        config = ParamsConfig(method='load', assistant_name=assistant_name)
        
        agent = Agent(params_config=config, instructions=instructions)

        # Start a new conversation for chat logging. TODO: ability to check existing conversations and load OR new
        conversation = start_new_conversation(host=agent.name, 
                                              host_is_bot=True, 
                                              guest=os.environ.get('USER') or os.environ.get('USERNAME'), 
                                              guest_is_bot=False)
        
        collection=chroma_get_or_create_collection(f"{agent.name}-{conversation.guest}")

        server = OllamaServer()
        available_port = server.find_available_port()
        print(f"This session will use port: {available_port}")
        server.start_server(available_port)

        try:
            while True:
                # Get the user's request
                request = input("User>> ")

                if request == 'exit' or request == 'quit':   # Check if the user wants to exit
                    print("Exiting chat...\n\n")
                    break
                elif request == '!focus':
                    input(f"Current focus: {agent.instructions.assistant_focus}. Type a new focus message or press enter to keep this one.")
                    if input == '':
                        continue
                    else:
                        agent.instructions.assistant_focus = input
                        continue
                
                # Convert to Message class
                request_message = Message(
                    uuid=str(uuid4()),
                    timestamp=str(datetime.now().strftime('%Y-%m-%d @ %H:%M')),
                    role='user',
                    speaker=conversation.guest,
                    content=request
                )

                # Build the prompt
                username = os.environ.get('USER') or os.environ.get('USERNAME')
                prompt = agent.build_prompt(request_message.content, username=username, agent_agent=False)

                #####  DEBUG: PROMPT  #####
                debug_print_function_return('Prompt', prompt)
                #####  DEBUG END  #####

                # Get the response and stream it to the terminal
                response_content = agent.generate_response(prompt=prompt, server_port=available_port)

                # Convert response to message class and pull the message string
                response_message = Message(
                    uuid=str(uuid4()),
                    timestamp=str(datetime.now().strftime('%Y-%m-%d @ %H:%M')),
                    role='assistant',
                    speaker=conversation.host,
                    content=response_content
                )

                stream_agent_response(agent.name, text=response_content, delay=.05)

                # Create turn and add to chat history
                convo_turn = Turn(
                    uuid=str(uuid4()),
                    request=request_message,
                    response=response_message
                )

                agent.message_cache.add_message(convo_turn)

                # Chroma Upsert
                document = convo_turn.request.to_memory_string()
                document += convo_turn.response.to_memory_string()
                #print(f"Documents: {document}")
                chroma_upsert_to_collection(collection=collection, metadata=None, document=document, id=convo_turn.uuid)
                

                ###  DEBUG: TURN BASE DICT  ###
                #debug_print_function_return('Turn Base Dict', convo_turn.to_dict())
                ###  DEBUG END  ###

                # Add Turn to Conversation YAML
        except KeyboardInterrupt:
            print("Interrupted by user...\n")
        
        finally:
            print("Chat session ended.")
            server.stop_server()


    def multi_agent_chat(self, host_agent_name: str, guest_agent_name: str) -> None:
        """
        Puts two agents into a chat session together. Super fun.

        :param host_agent: The name of the agent to host the chat.
        :param guest_agent: The name of the agent to join the chat.
        :returns: Hours of enjoyment if you know how to prompt..
        """
        # Load the host agent from file
        host_agent_model_instructions = ModelInstructions(method='load', assistant_name=host_agent_name)
        host_agent_params_config = ParamsConfig(method='load', assistant_name=host_agent_name)
        host_agent = Agent(host_agent_params_config, host_agent_model_instructions)
        # Load the guest agent from file
        guest_agent_model_instructions = ModelInstructions(method='load', assistant_name=guest_agent_name)
        guest_agent_params_config = ParamsConfig(method='load', assistant_name=guest_agent_name)
        guest_agent = Agent(guest_agent_params_config, guest_agent_model_instructions)

        # Print the banners
        toilet_banner_metal(host_agent.name)
        toilet_banner_plain('welcomes')
        toilet_banner_metal(guest_agent.name)

        # Start a new conversation for chat logging. TODO: ability to check existing conversations and load OR new
        conversation = start_new_conversation(host_agent, 
                                              host_is_bot=True, 
                                              guest=guest_agent, 
                                              guest_is_bot=True)
        
        server = OllamaServer()
        available_port = server.find_available_port()
        print(f"This session will use port: {available_port}")
        server.start_server(available_port)
        
        host_collection = chroma_get_or_create_collection(f"{host_agent.name}-{guest_agent.name}")
        guest_collection = chroma_get_or_create_collection(f"{guest_agent.name}-{host_agent.name}")


        # Get the guest's first message before entering the chat to give the while loop a little better progression.
        host_agent.last_response = f"Hello, I'm {host_agent.name}, welcome to my room! People describe me as: {host_agent.instructions.description}. Please first tell me a little bit about yourself, and then give me 2 topics that you may be interested in speaking with me about. As your host, I will choose our first subject from your list."

        try:
            while True:
                guest_prompt = guest_agent.build_prompt(host_agent.last_response, username=host_agent.name, agent_agent=True)
                debug_print_function_return('Guest Prompt', guest_prompt)
                guest_agent.last_response = guest_agent.generate_response(prompt=guest_prompt, server_port=available_port)

                guest_request_message = Message(
                    uuid=str(uuid4()),
                    timestamp=str(datetime.now().strftime('%Y-%m-%d @ %H:%M')),
                    role='user',
                    speaker=guest_agent.name,
                    content=guest_agent.last_response
                )

                # Stream the guest request to the terminal chat
                stream_agent_response(guest_agent.name, guest_agent.last_response, 0.05)
                
                # Request to Hosting Agent
                host_agent_prompt = host_agent.build_prompt(guest_agent.last_response, username=guest_agent.name, agent_agent=True)
                debug_print_function_return('Host Prompt', host_agent_prompt)
                host_agent.last_response = host_agent.generate_response(prompt=host_agent_prompt, server_port=available_port)

                host_response_message = Message(
                    uuid=str(uuid4()),
                    timestamp=str(datetime.now().strftime('%Y-%m-%d @ %H:%M')),
                    role='assistant',
                    speaker=host_agent.name,
                    content=host_agent.last_response
                )

                # Stream the host response to the terminal chat
                stream_agent_response(host_agent.name, host_agent.last_response, 0.05)

                # Create turn and add to chat history
                message_turn = Turn(
                    uuid=str(uuid4()),
                    request=guest_request_message,
                    response=host_response_message
                )

                # Add Turn to each agents' message cache for prompt context
                host_agent.message_cache.add_message(message_turn)
                guest_agent.message_cache.add_message(message_turn)

                # Add Turn to Conversation
                conversation.create_turn(guest_request_message, host_response_message)
                document = message_turn.request.to_memory_string()
                document += message_turn.response.to_memory_string()
                chroma_upsert_to_collection(collection=host_collection, metadata=None, document=document, id=message_turn.uuid)
                chroma_upsert_to_collection(collection=guest_collection, metadata=None, document=document, id=message_turn.uuid)
        except KeyboardInterrupt:
            print("Interrupted by user...\n")
        
        finally:
            print("Chat session ended.")
            server.stop_server()
