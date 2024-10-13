from json import detect_encoding
import requests
from bs4 import BeautifulSoup
import random
import time
from dotenv import load_dotenv
import re

# Purpose: This script is used to get random wikipedia articles and save them to file. The article content will be added to a custom training
# dataset for our in house LLM models. The script will run indefinitely if allowed and will eventually all of wikipedia. This is a base implementation,
# there will be updates for data cleaning, summarization, entity mapping and a rabbit hole function to dive deeper into links from article reference
# on articles within flagged topics or categories.  This is also fun for a random fact or trivia bot for discord or slack.

# Load environment variables
load_dotenv()

# Parse the wikipedia article
def parse_wikipedia_article(response: requests.Response) -> (str, str):
    """
    Parse http response for a wikipedia article.

    :param response: The http response to parse.
    :returns: The title and content of the wikipedia article.
    """
    soup = BeautifulSoup(response.text, 'html.parser')
    page_title = soup.find('title').text
    page_content = soup.find('div', {'id': 'mw-content-text'}).text
    return page_title, page_content

# Remove citation references [*]
def remove_references(text: str) -> (str):
    """
    Remove citation references from wikipedia article text by removing text between square brackets and the square brackets themselves.

    :param text: The text to remove references from.
    """
    # Remove the square brackets and numbers inside them
    text = re.sub(r'\[\d+\]', '', text)
    return text

def wikipedia_get_random_article() -> None:
    # run loop to get random wikipedia articles and save to txt files
    while True:
        response = requests.get(url="https://en.wikipedia.org/wiki/Special:Random")
        page_title, page_content = parse_wikipedia_article(response)
        
        # check if the text is English
        try:
            if detect_encoding(page_content) != 'en':
                print(f"Skipping non-English article: {page_title}")
                continue
        except Exception as e:
            print(f"Skipping article due to error during language detection: {page_title}. Error: {str(e)}")
            time.sleep(random.randint(1, 12))
            continue

        # Remove citation references [*]
        page_content = remove_references(page_content)

        with open(f"wikipedia_{page_title}.txt", "x") as f:
            f.write(page_content)
        
        # wait between 1 to 5 seconds before next request to respect Wikipedia's server
        time.sleep(random.randint(1,12))


def wikipedia_get_search_article(search_query: str) -> None:
    """
    Search Wikipedia for the query and get the first result's title and content.

    :param query: The query to search Wikipedia for.
    :returns: A .txt file containing the article content saved to agent directory.
    """
    response = requests.get(url="https://en.wikipedia.org/wiki/...search...")
    page_title, page_content = parse_wikipedia_article(response)
    
    # check if the text is English
    try:
        if detect_encoding(page_content) != 'en':
            print(f"Skipping non-English article: {page_title}")
            return
    except Exception as e:
        print(f"Skipping article due to error during language detection: {page_title}. Error: {str(e)}")
        return
        
    # Remove citation references [*]
    page_content = remove_references(page_content)

    with open(f"wikipedia_{page_title}.txt", "x") as f:
        f.write(page_content)


def o_wikipedia_get_search_article(query: str) -> None:
    """
    A slightly beefier version of wikipedia_get_search_article() that also saves the article content to a file.

    :param query: The query to search Wikipedia for.
    :returns: A .txt file containing the article content saved to agent directory.
    """
    # Search Wikipedia for the query and get the first result's title
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
    search_response = requests.get(search_url).json()
    search_results = search_response.get('query', {}).get('search', [])
    if not search_results:
        print(f"No results found for {query}")
        return

    page_title = search_results[0]['title']

    # Fetch the article content
    page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the main text content
    content = soup.find('div', {'class': 'mw-parser-output'})
    page_content = content.get_text() if content else ""

    # Optional: language detection and removing references can be implemented here
    
    # Save the content to a file
    with open(f"wikipedia_{page_title.replace('/', '_')}.txt", "w") as f:
        f.write(page_content)

    print(f"Article '{page_title}' saved to file.")