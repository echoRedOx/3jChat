from sys import argv
import requests
import os


def main():
    """
    Query the wolfram alpha api and return the response. Currently just prints the response to stdout and runs as main.

    :param query: The query to use. ("This is a query")
    :return: The response from the api.
    """
    api_key = os.getenv("WOLFRAM_API_KEY")
    query = argv[1]
    response = f"http://api.wolframalpha.com/v2/query?input={query}&appid={api_key}"

    return print(response)


if __name__ == "__main__":
    main()