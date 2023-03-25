#!/usr/bin/env python
# coding: utf-8

# # Convert PDF to raw text

# In[1]:


pwd


# In[2]:


file_path =r"Estudio amenaza sismica Porce III.pdf"


# In[4]:


get_ipython().run_line_magic('pip', 'install PyPDF2')


# In[5]:


import PyPDF2

pdf_file = open(file_path, 'rb')
read_pdf = PyPDF2.PdfReader(pdf_file)
number_of_pages = len(read_pdf.pages)
page = read_pdf.pages[7]
page_content = page.extract_text()

print(page_content)


# In[14]:


import re
from typing import Set
from transformers import GPT2TokenizerFast

import numpy as np
from nltk.tokenize import sent_tokenize

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text

discard_categories = ['See also', 'References', 'External links', 'Further reading', "Footnotes",
    "Bibliography", "Sources", "Citations", "Literature", "Footnotes", "Notes and references",
    "Photo gallery", "Works cited", "Photos", "Gallery", "Notes", "References and sources",
    "References and notes",]


def extract_sections(
    pdf_text: str,
    page_number: int,
    max_len: int = 1500,
    discard_categories: Set[str] = discard_categories,):
    """
    Extract the sections of a PDF document, discarding the references and other low information sections
    """
    # Split the text into lines
    lines = pdf_text.split('\n')
    sections = []
    current_section = {'heading': '', 'content': '', 'tokens': 0}
    for line in lines:
        # Check if the line matches the heading marker convention
        if re.match('^\d+\. .+', line):
            # If so, add the current section to the list and start a new section
            if current_section['tokens'] > 0:
                sections.append((
                    page_number,
                    current_section['heading'],
                    current_section['content'],
                    current_section['tokens']
                ))
            current_section = {'heading': line.strip(), 'content': '', 'tokens': 0}
        else:
            # Otherwise, add the line to the current section's content
            current_section['content'] += line + '\n'
            current_section['tokens'] += count_tokens(line) + 1  # Add 1 for the newline character
    # Add the last section to the list
    if current_section['tokens'] > 0:
        sections.append((
            page_number,
            current_section['heading'],
            current_section['content'],
            current_section['tokens']
        ))
    # Filter out low-information sections and truncate long sections
    outputs = []
    for section in sections:
        if section[3] < 40 or section[1] in discard_categories:
            continue
        elif section[3] > max_len:
            reduced_content = reduce_long(section[2], long_text_tokens=section[3], max_len=max_len)
            outputs.append((section[0], section[1], reduced_content, count_tokens(reduced_content)))
        else:
            outputs.append(section)
    return outputs



# In[ ]:


### 1.2.1 We create a dataset and filter out any sections with fewer than 40 tokens, as those are unlikely to contain enough context to ask a good question.
res = []
for page_number, page in enumerate(read_pdf.pages):
    res += extract_sections(page.extract_text(), page_number)
    


# In[20]:


import pandas as pd
df = pd.DataFrame(res, columns=["page_number", "heading", "content", "tokens"])
df = df[df.tokens>40]
# df = df.drop_duplicates(['heading'])
df = df.reset_index().drop('index',axis=1) # reset index
df.head(25)

# In[ ]:

