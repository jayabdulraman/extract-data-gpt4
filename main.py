from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import json
import pandas as pd
import time
import os
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Extract structured info from text via LLM
def extract_structured_data(content: str, data_points):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY)
    template = """
    You are an expert scholarship admin people who will extract core information from documents

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}

    Now please extract details from the content  and export in a JSON array format, 
    return ONLY the JSON array:
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.run(content=content, data_points=data_points)

    return results

def extract(texts):
    default_data_points = """{
            "name": "what is the name of the scholarship",
            "brief_description": "what is a brief description of the scholarship",
            "value": "what is the scholarship value or inclusion or duration?",
            "eligibility": "what is the eligibility of the scholarship?",
            "deadline": "what is the deadline of the scholarship?",
            "field_of_study": "what is the level or fields of study that the scholarship cover?",
            "application_instructions": "what are the application instructions",
            "official_scholarship_website": "what is the official scholarship website?",
        }"""
    results = []
    count = 1
    for text in texts:
        # remove unnecessary characters
        try:
            text = text.replace("\n", " ").replace("\xa0", " ").split("Related Scholarships:")[0]
            data = extract_structured_data(text, default_data_points)
            # transform to json
            json_data = json.loads(data)
            if isinstance(json_data, list):
                results.extend(json_data)  # Use extend() for lists
            else:
                results.append(json_data)  # Wrap the dict in a list
            # pause program for 8 seconds to prevent hitting TPM 
            time.sleep(8)
            print(f"Extraction {count} completed!")
        except Exception as e:
            print("Error: ", e)
        count += 1
    return results

if __name__ == "__main__":
    # List of texts to extract data from
    df = pd.read_json("./data/international-scholarships.json")
    df.rename(columns={0: "texts"}, inplace=True)
    texts = df.texts.tolist()
    # Scrape content from URLs
    results = extract(texts)
    # Write the list of dictionaries to a JSON file
    print("File:", results)
    with open("data/scholarships.json", "w") as file:
        json.dump(results, file, indent=2)
    # print progress
    print("Extraction and write successful!")
    print("Program Completed!")
