import json
from sys import argv


file_path = argv[1]
dest_dir = argv[2]


# Load the JSON data from the file
with open(file_path, 'r') as file:
    dataset = json.load(file)

requests = []
responses = []
turns = []
train_prompts = []
turn_count = 0
keys_skipped_count = 0
data = dataset

# Pretty print the JSON data
print(json.dumps(data, indent=4))

print(f"Number of Conversations in dataset: {len(data)}")

try:
    for conversation in data:
        for key, value in conversation['mapping'].items():
            if 'message' in value and value['message'] and 'content' in value['message'] and 'parts' in value['message']['content']:
                message_id = value['id']
                message_content = value['message']['content']['parts'][0] if 'content' in value['message'] else "NO CONTENT"
                message_role = value['message']['author']['role'] if 'author' in value['message'] else "NO ROLE"
                message_parent = value['parent']

                requests.append({"id": message_id, "role": message_role, "request": message_content}) if message_role == 'user' else None
                responses.append({"id": message_parent, "role": message_role, "response": message_content}) if message_role == 'assistant' else None
                turn_count += 1

            else:
                print(f"key: {key} skipped")
                keys_skipped_count += 1
            
    for request in requests:
        request['response'] = next((response['response'] for response in responses if response['id'] == request['id']), None)
        prompt_request = f"""<|im_start|>System: \nYour name is Juliet, an acronym for Junctive Unsupervised Learning for Incrementally Evolving Transformers. You are a helpful AI super-assistant and empathetic human companion. The User has made a request to you, Juliet. Please respond with accurate and meaningful information or a solution for the User.<|im_end|>\n<|im_start|>User: \n{request['request']}<|im_end|>\n"""

        prompt_response = f"""<|im_start|>Juliet: \n{request['response']}<|im_end|>\n"""

        prompt = f"{prompt_request}{prompt_response}"

        prompt_dict = {"request": prompt_request, "response": prompt_response, "prompt": prompt}

        turns.append(prompt_dict)
        train_prompts.append(prompt)

except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except FileNotFoundError:
    print(f"File not found: {file_path}")

#Export to json file
with open(f"{dest_dir}/conversation_turns.json", 'w') as outfile:
    json.dump(turns, outfile, indent=4)
print("-------------\nConversation Turns JSON file exported successfully.\n-------------\n")

#Export to json file
with open(f"{dest_dir}/training_prompts.json", 'w') as outfile:
    json.dump(train_prompts, outfile, indent=4)
print("-------------\nTraining Prompts JSON file exported successfully.\n-------------\n")

print(f"Number of keys skipped: {keys_skipped_count}")
print(f"Number of Request/Response pairs saved to the dataset: {turn_count}")

