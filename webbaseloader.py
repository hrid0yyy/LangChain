from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from GroqCloud import llama

llm = llama()

def extract(content, keywords):
    """Extract lines containing any of the specified keywords (case-insensitive) from the content."""
    if not content:
        return "No content provided."
    if not keywords:
        return "No keywords provided."
    lines = content.split('\n')
    matching_lines = [line.strip() for line in lines if any(keyword.lower() in line.lower() for keyword in keywords)]
    return '\n'.join(matching_lines) if matching_lines else "No relevant information found."

# Load environment variables
load_dotenv()


url = "https://en.wikipedia.org/wiki/2025_Royal_Challengers_Bengaluru_season"
keywords = ['RCB', 'Royal Challengers Bengaluru']
loader = WebBaseLoader(url)
docs = loader.load()

if docs and docs[0].page_content:
    content = extract(docs[0].page_content, keywords=keywords)
    prompt = f"Summarize the following information:\n{content}"
    response = llm.invoke(prompt)
    print(response)
else:
    print("No content loaded from the URL.")