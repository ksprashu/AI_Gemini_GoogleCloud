# gets a list of cities guides from the dataset and stores it.

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

import os
import json
import dotenv
import re
import logging


from lxml import etree

# fetch project_id and location from environment
dotenv.load_dotenv()
project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("REGION")

_DATASET = '../datasets/enwikivoyage-latest-pages-articles.xml'

# init vertex ai with project_id and location
vertexai.init(project=project_id, location=location)
model = GenerativeModel(
    model_name="gemini-1.5-flash-002")


def select_cities(count: int) -> list[str]:
    """Ask the model to generate a list of cities"""
    response = model.generate_content(
        contents=f"""
        Generate a list of {count} cities that are popular tourist destinations. 
        Return the response as a json list of strings.
        """,
        generation_config=GenerationConfig(response_mime_type="application/json")
    )

    logging.info(f"Cities: {response.text}")
    return json.loads(response.text)


def get_revisions(cities: list[str]) -> dict[str, str]:
    """parse the xml file and fetch the revisions for the cities
    
        Returns a map of cityname and coresponding revision text
    """
    guides = {}

    # Define the correct namespace
    ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
    context = etree.iterparse(
        _DATASET,
        events=("end",), 
        tag="{http://www.mediawiki.org/xml/export-0.11/}page")

    for event, page in context:
        title = page.find("mw:title", ns).text
        if title in cities:
            # Get the <revision> element
            revision = page.find("mw:revision", ns) 
            # Extract the content from the <text> tag within <revision>
            content = revision.find("mw:text", ns).text

            # if there is a redirect, then log this.
            matches = re.findall(r"#REDIRECT \[\[(.*?)\]\]", content, re.IGNORECASE)
            if len(matches) > 0:
                # log a warning for the title
                logging.warning(f"Redirect found for title: {title}")
                continue

            # store the content against the city name
            logging.info(f"Found revision for title: {title}")
            guides[title] = content

    return guides
            

# main call
if __name__ == "__main__":
    cities = select_cities(50)
    guides = get_revisions(cities)

    # write in jsonl format
    with open('city_guides.jsonl', 'w') as f:
        for city, guide in guides.items():
            f.write(json.dumps({"city": city, "guide": guide}) + '\n')

