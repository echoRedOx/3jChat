
from dataclasses import asdict, dataclass, field

import yaml


@dataclass
class SystemInstructions:
    description: str = None
    llm_model: str = None
    system_message: str = None
    assistant_intro: str = None
    assistant_focus: str = None
    commands: dict = None
    prompt_script: str = None
    start_token: str = field(default="<|im_start|>", metadata={"help": "Start token for message content"})
    end_token: str = field(default="<|im_end|>", metadata={"help": "End token for message content"})
    mem_start_token: str = field(default="<|mem_start|>", metadata={"help": "Start token for context memories"})
    mem_end_token: str = field(default="<|mem_end|>", metadata={"help": "End token for context memories"})
    chat_start_token: str = field(default="<|chat_start|>", metadata={"help": "Start token for chat history"})
    chat_end_token: str = field(default="<|chat_end|>", metadata={"help": "End token for chat history"})
    name: str = field(default="System", metadata={"help": "Name of the assistant"})

    def to_dict(self):
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
        self.save_to_yaml(self.name)

    def save_to_yaml(self) -> None:
        """
        Save the agent config to a yaml file.

        :returns: Saves the agent config to a yaml file.
        """
        data = self.to_dict()
        with open(f"/agents/{self.assistant_name.lower()}/params_config.yaml", "w") as f:
            yaml.safe_dump(data, f)


@dataclass
class SystemParams_Config:
    temperature: float = field(default=0.5, metadata={"help": "0 implies determinism -> >0 implies creativity scaled"})
    num_ctx: int = field(default=4096, metadata={"help": "Context window size"})
    num_gpu: int = field(default=50, metadata={"help": "Number of layers for GPU"})
    num_thread: int = field(default=16, metadata={"help": "Number of threads for computation"})
    top_k: int = field(default=42, metadata={"help": "Limits token generation"})
    top_p: float = field(default=0.42, metadata={"help": "Diversity of text generation"})
    num_predict: int = field(default=512, metadata={"help": "Number of tokens to generate"})
    seed: int = field(default=0, metadata={"help": "Seed for RNG"})
    mirostat: int = field(default=0, metadata={"help": "Enables Mirostat algorithm"})
    mirostat_eta: float = field(default=0.1, metadata={"help": "Learning rate for Mirostat"})
    mirostat_tau: float = field(default=5.0, metadata={"help": "Temperature for Mirostat"})
    repeat_last_n: int = field(default=64, metadata={"help": "Tokens to repeat at end of context"})
    completions_url: str = field(default_factory=lambda: None, metadata={"help": "URL of Ollama API"})
    completion_headers: dict = field(default_factory=lambda: {"Content-Type": "application/json"}, metadata={"help": "Headers for Ollama API request"})
    tfs_z: int = field(default=0, metadata={"help": "Tokens for TFS-Z algorithm"})
    assistant_name: str = field(default="System", metadata={"help": "Name of the assistant"})


class SystemAgent:
    completions_url: str
    agents_list: dict
    agents_dir: str
    instructions_path: str
    instructions: dict
    model_params_path: str
    model_params: dict
    last_request: str
    last_response: str
    chroma_collection_name: str
    


    