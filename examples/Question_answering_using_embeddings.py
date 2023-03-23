#!/usr/bin/env python
# coding: utf-8

# # Question Answering using Embeddings
# 
# Many use cases require GPT-3 to respond to user questions with insightful answers. For example, a customer support chatbot may need to provide answers to common questions. The GPT models have picked up a lot of general knowledge in training, but we often need to ingest and use a large library of more specific information.
# 
# In this notebook we will demonstrate a method for enabling GPT-3 to answer questions using a library of text as a reference, by using document embeddings and retrieval. We'll be using a dataset of Wikipedia articles about the 2020 Summer Olympic Games. Please see [this notebook](fine-tuned_qa/olympics-1-collect-data.ipynb) to follow the data gathering process.

# In[5]:


import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken

COMPLETIONS_MODEL = "gpt-3.5-turbo"
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


# In[7]:


# We have hosted the processed dataset, so you can download it directly without having to recreate it.
# This dataset has already been split into sections, one row for each section of the Wikipedia page.

# df = pd.read_csv('fine-tuned_qa\olympics-data\olympics_sections.csv')
df = pd.read_csv(r'fine-tuned_qa/porce-III-data/Estudio amenaza sismica Porce III.csv')
df = df.set_index(["page_number", "heading"])


# In[8]:


df = df[df.tokens>500]
print(f"{len(df)} rows in the data.")


# We preprocess the document sections by creating an embedding vector for each section. An embedding is a vector of numbers that helps us understand how semantically similar or different the texts are. The closer two embeddings are to each other, the more similar are their contents. See the [documentation on OpenAI embeddings](https://beta.openai.com/docs/guides/embeddings) for more information.
# 
# This indexing stage can be executed offline and only runs once to precompute the indexes for the dataset so that each piece of content can be retrieved later. Since this is a small example, we will store and search the embeddings locally. If you have a larger dataset, consider using a vector search engine like [Pinecone](https://www.pinecone.io/), [Weaviate](https://github.com/semi-technologies/weaviate) or [Qdrant](https://qdrant.tech) to power the search.

# In[9]:


def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]
# %%
def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }


# In[10]:


def load_embeddings(fname: str):
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


# Again, we have hosted the embeddings for you so you don't have to re-calculate them from scratch.

# In[11]:


# document_embeddings = load_embeddings("https://cdn.openai.com/API/examples/data/olympics_sections_document_embeddings.csv")

# ===== OR, uncomment the below line to recaculate the embeddings from scratch. ========
# Read api key from environment variable
import os

openai.api_key = "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"

document_embeddings = compute_doc_embeddings(df)
# %%
# Save the embeddings to a CSV so we can load them later.
pd.DataFrame(document_embeddings).T.to_csv("fine-tuned_qa/porce-III-data/Porce Embeddings.csv")
# In[12]:


# An example embedding:
example_entry = list(document_embeddings.items())[0]
print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")


# So we have split our document library into sections, and encoded them by creating embedding vectors that represent each chunk. Next we will use these embeddings to answer our users' questions.
# 
# # 2) Find the most similar document embeddings to the question embedding
# 
# At the time of question-answering, to answer the user's query we compute the query embedding of the question and use it to find the most similar document sections. Since this is a small example, we store and search the embeddings locally. If you have a larger dataset, consider using a vector search engine like [Pinecone](https://www.pinecone.io/), [Weaviate](https://github.com/semi-technologies/weaviate) or [Qdrant](https://qdrant.tech) to power the search.

# In[13]:


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


# In[14]:


order_document_sections_by_query_similarity("Cual es la fuente sismogenica de la presa", document_embeddings)[:5]


# In[15]:


order_document_sections_by_query_similarity("Cuantos escenarios de amenaza sismica se consideraron", document_embeddings)[:5]


# # 3) Add the most relevant document sections to the query prompt
# 
# Once we've calculated the most relevant pieces of context, we construct a prompt by simply prepending them to the supplied query. It is helpful to use a query separator to help the model distinguish between separate pieces of text.

# In[16]:


MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"


# In[17]:


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            print(f"Reached maximum section length of {MAX_SECTION_LEN} tokens.")
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


# In[18]:


prompt = construct_prompt(
    "Cuantos escenarios de amenaza sismica se consideraron?",
    document_embeddings,
    df
)

print("===\n", prompt)


# We have now obtained the document sections that are most relevant to the question. As a final step, let's put it all together to get an answer to the question.
# 
# # 4) Answer the user's question based on the context.
# 
# Now that we've retrieved the relevant context and constructed our prompt, we can finally use the Completions API to answer the user's query.

# In[29]:


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.5,
    "max_tokens": 200,
    "model": COMPLETIONS_MODEL,
}


# In[30]:


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    
    import textwrap

    def split_string(string):
        lines = textwrap.wrap(string, width=80)
        return '\n'.join(lines)
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    string = split_string(response["choices"][0]["text"].strip(" \n"))
    animate_string(string)
    return


# In[32]:


answer_query_with_context("Cual es la fuente sismogenica predominante para la presa Porce III? Responde en detalle", df, document_embeddings)

# %%
import time

def animate_string(long_string, time_step=0.1, line_length=80):
    words = long_string.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > line_length:
            print(current_line)
            current_line = word + " "
        else:
            current_line += word + " "
        time.sleep(time_step)
    print(current_line)

# Example usage
long_string = "This is a very long string that needs to be printed out word by word in an animation with a short time step and create a new line after about 80 characters."
animate_string(long_string)



# %%
query = "En qué consiste un análisis de amenaza sísmica probabilístico?"
answer_query_with_context(query, df, document_embeddings)
# %%
answer = answer_query_with_context("Cual es la fuente sismogenica predominante para la presa Porce III? Responde en detalle", df, document_embeddings)
print(split_string(answer))
# %%
answer_query_with_context("Cuales son los escenarios de amenaza sismica considerados?", df, document_embeddings)
# %%
answer_query_with_context("Cual es el pga para la presa Porce III? Responde detalladamente.", df, document_embeddings)
# %%
answer_query_with_context("Qué programa de computador se utilizó para calcular el efecto de las fuentes sísmicas?", df, document_embeddings)
# %%

answer = answer_query_with_context("Cómo funciona el programa EZ-FRISK?", df, document_embeddings)

# Split the answer into multiple lines



# In[22]:


query = "Why was the 2020 Summer Olympics originally postponed?"
answer = answer_query_with_context(query, df, document_embeddings)

print(f"\nQ: {query}\nA: {answer}")


# In[23]:


query = "In the 2020 Summer Olympics, how many gold medals did the country which won the most medals win?"
answer = answer_query_with_context(query, df, document_embeddings)

print(f"\nQ: {query}\nA: {answer}")


# In[24]:


query = "What was unusual about the men’s shotput competition?"
answer = answer_query_with_context(query, df, document_embeddings)

print(f"\nQ: {query}\nA: {answer}")


# In[25]:


query = "In the 2020 Summer Olympics, how many silver medals did Italy win?"
answer = answer_query_with_context(query, df, document_embeddings)

print(f"\nQ: {query}\nA: {answer}")


# Our Q&A model is less prone to hallucinating answers, and has a better sense of what it does or doesn't know. This works when the information isn't contained in the context; when the question is nonsensical; or when the question is theoretically answerable but beyond GPT-3's powers!

# In[26]:


query = "What is the total number of medals won by France, multiplied by the number of Taekwondo medals given out to all countries?"
answer = answer_query_with_context(query, df, document_embeddings)

print(f"\nQ: {query}\nA: {answer}")


# In[27]:


query = "What is the tallest mountain in the world?"
answer = answer_query_with_context(query, df, document_embeddings)

print(f"\nQ: {query}\nA: {answer}")


# In[28]:


query = "Who won the grimblesplatch competition at the 2020 Summer Olympic games?"
answer = answer_query_with_context(query, df, document_embeddings)

print(f"\nQ: {query}\nA: {answer}")

