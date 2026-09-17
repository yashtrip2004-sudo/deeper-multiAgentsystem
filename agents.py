from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

#1st agent

def build_search_agent():
    return create_agent(
        model= llm,
        tools=[web_search]
    )

#2nd agent

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]

    )

#writer chain

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer. Write clear, structured and insightful reports based on the research provided."""
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""
    ),
])

writer_chain = writer_prompt | llm |StrOutputParser()

#critic_chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research critic. Evaluate the quality of the research report based on accuracy, relevance, completeness, source quality and clarity."""
    ),
    (
        "human",
        """Critically review the research report below.

Topic: {topic}

Research Report:
{research}

Evaluate the report on the following criteria:
- Accuracy
- Relevance
- Completeness
- Source Quality
- Clarity and Structure

Give an overall research score out of 10.

Provide your response in the following format:

Research Score: X/10

Review:
- Accuracy: ...
- Relevance: ...
- Completeness: ...
- Source Quality: ...
- Clarity and Structure: ...

Overall Review:
...

Suggestions for Improvement:
- ...
- ...
- ...

Be objective, factual and constructive."""
    ),
])

critic_chain = critic_prompt |llm |StrOutputParser()

