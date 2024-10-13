from handlers.chroma_handler import chroma_delete_collection
from sys import argv


def main():
    agent_name = argv[1]
    chroma_delete_collection(agent_name)
    print(f"Deleted collection for {agent_name}.")