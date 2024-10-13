import hashlib
import os


def calculate_file_checksum(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def calculate_directory_checksum(directory_path):
    hash_sha256 = hashlib.sha256()
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_checksum = calculate_file_checksum(file_path)
            hash_sha256.update(file_checksum.encode())
    return hash_sha256.hexdigest()

directory_path = '/agents/clappy/fine-tuning'
dir_checksum = calculate_directory_checksum(directory_path)
print(f"Checksum for directory {directory_path}: {dir_checksum}")
