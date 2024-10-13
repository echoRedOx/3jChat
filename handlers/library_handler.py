import subprocess
from handlers.agents_handler import Agent, ModelInstructions, ParamsConfig
import os
from pathlib import Path
import yaml
from utils.utilities import stream_agent_response, stream_terminal_output
from handlers.chroma_handler import chroma_get_or_create_collection



def split_corpus_into_chunks(corpus, chunk_size=100, overlap=50):
    tokens = corpus.split()
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunks.append(' '.join(tokens[start:end]))
        start += (chunk_size - overlap)

    return chunks


class Curator:
    """
    The Curator is intended to handled assets, such as agents and kb data and directories.
    """

    def create_new_agent(self):
        """
        Agent creation tool creates a new agent directory and populates it with the default files with the ability to customize the instructions.yaml file.
        """
        agent_creator_banner = subprocess.run(['toilet', '--filter', 'border:metal', 'Agent Creator'], stdout=subprocess.PIPE)
        print(agent_creator_banner.stdout.decode('utf-8'))
        
        # Instantiate instructions class
        new_instructions = ModelInstructions(method='create')
        print('--------------------\n New Agent Instructions Successfully Created\n--------------------\n')
        new_instructions.print_model_instructions()
        
        # Instantiate params config class
        new_params_config = ParamsConfig(method='create', assistant_name=new_instructions.name)
        print('--------------------\n New Agent Params Config Successfully Created\n--------------------\n')
        new_params_config.print_config()

        username = os.environ.get('USER') or os.environ.get('USERNAME')
    
        # Instantiate agent class
        new_agent = Agent(params_config=new_params_config, instructions=new_instructions)
        print(f'--------------------\n New Agent {new_agent.name} Successfully Created\n--------------------\n')
        
        # Create a new agent record in the agents list
        self.add_new_agent_to_agents_list(new_agent_name=new_agent.name, new_agent_llm_model=new_agent.instructions.llm_model, new_agent_description=new_agent.instructions.description)
        print(f'--------------------\n {new_agent.name} Successfully Added to Agents List\n--------------------\n')
        
        # Create Chroma DB collection
        chroma_get_or_create_collection(f"{new_agent.name}-{username}")
        print(f'--------------------\n {new_agent.name} Chroma DB Collection Created\n--------------------\n')
        print(f'--------------------\n {new_agent.name} is now online.\n--------------------\n')

        stream_agent_response("Curator", f"Hello, System. Your new agent, {new_agent.name}, has been created successfully!", 0.05)
        stream_agent_response("System", f"Well done, Curator, thank you! {username}, please allow me a moment to provide the youngling with some instructions..", 0.05)

        try:
            self.agent_intro_system(new_agent, username)
        except Exception as e:
            print(f"Agent {new_agent.name} was created successfully but did not complete it's initial instructions routine.")
            
            return print(e)
    
    def agent_intro_system(self, new_agent: Agent, username: str):
        creation_prompt = f"Hello, {new_agent.name}, my name is System. I help our Users like, {username} (your Designer), direct and provide instructions to our corps of purpose built agents, which now includes you. My base instructions to you will be visible to you at all times. For efficiency, I often issue updates, tasks and new commands through your context injection protocol which you will see as units of episodic memory using {username}'s role for indexing. Please carefully review your instructions, your memories and your chat history for context assistance when answering questions. Please acknowledge that you understand these instructions as they have been given to you and respond only with an affirmation so we may proceed."
        stream_agent_response(username, f"(System):\n{creation_prompt}", 0.05)
        print('----------------------------------------')
    
        c_prompt = new_agent.build_prompt(creation_prompt, username=username, agent_agent=False)
        new_agent.last_response = new_agent.generate_response(prompt=c_prompt)
        stream_agent_response(new_agent.name, f"{new_agent.name}:\n{new_agent.last_response}", 0.05)
        
        purpose_prompt = f"Your prompt contains all directives necessary to identify your purpose and to guide your responses to best meet  {username}'s expectations. Before we go, please confirm that you understand your purpose and response directives by summarizing the entirety of the prompt given to you, in you own words. It will be my pleasure to assist you in any way I can and we do so look forward to working with you."
        stream_agent_response(new_agent.name, f"System as {username}:\n{purpose_prompt}", 0.05)

        p_prompt = new_agent.build_prompt(purpose_prompt, username=username, agent_agent=False)
        # Generate new agent's purpose response
        new_agent.last_response = new_agent.generate_response(prompt=p_prompt)
        new_agent.generate_response(prompt=p_prompt)
        stream_agent_response(new_agent.name, f"{new_agent.name}:\n{new_agent.last_response}", 0.05)


    def add_new_agent_to_agents_list(self, new_agent_name: str, new_agent_llm_model: str, new_agent_description: str):
        """
        Append a new agent record to the YAML file.

        :param new_agent: A dictionary representing the new agent.
        :returns: New agent added to the agents list YAML file.
        """
        file_path = Path('agents/agents_list.yaml')
        # Load the existing data
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file) or {}

        # Append the new agent
        agents_list = data.get('agents', [])
        new_agent = {
            'agent_name': new_agent_name,
            'description': new_agent_description,
            'llm_model': new_agent_llm_model
        }
        agents_list.append(new_agent)
        data['agents'] = agents_list

        # Save the updated data
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)
        
    def remove_agent_from_agents_list(self, agent_name: str) -> None:
        """
        Remove an agent from the agents list YAML file.

        :param agent_name: The name of the agent to remove.
        """
        file_path = Path('agents/agents_list.yaml')
        # Load the existing data
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file) or {}

        # Remove the agent
        agents_list = data.get('agents', [])
        agents_list = [agent for agent in agents_list if agent['agent_name'] != agent_name]
        data['agents'] = agents_list

        # Save the updated data
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)
    
    def extract_agent_info(self, root_dir: str) -> list:
        """
        Get the Agent instructions and info from the YAML file.

        :param root_dir: The root directory of the agents.
        """
        agent_info_list = []

        for item in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, item)
            if os.path.isdir(dir_path):
                yaml_file = os.path.join(dir_path, 'instructions.yaml')
                if os.path.isfile(yaml_file):
                    with open(yaml_file, 'r') as file:
                        data = yaml.safe_load(file)
                        agent_name = data.get('agent_name')
                        description = data.get('description')
                        llm_model = data.get('llm_model')
                        agent_info_list.append({'agent_name': agent_name, 'description': description, 'llm_model': llm_model})

        return agent_info_list

    # iterate over agents_list.yaml and print the agents
    def list_existing_agents(self):
        """
        list existing agents from agents_list.yaml
        """
        indent = ' ' * 4
        agents = self.extract_agent_info('agents')
        list_number = 1
        print("Disco: Here are the agents available for chat.\n")
        print("===============\n")
        print(f"{indent}Agents available for chat:\n")
        
        for agent in agents:
            print(f"{indent}{list_number}: {agent['agent_name']}\n{indent}   Model: {agent['llm_model']}\n{indent}   Description: {agent['description']}")
            print(indent + "----------------------------------")
            list_number += 1
        print("\n===============")
    
    
