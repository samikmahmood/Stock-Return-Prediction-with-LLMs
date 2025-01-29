#!/usr/bin/env python
# coding: utf-8

# # OpenAI's API

# In[2]:


get_ipython().system('pip install openai')


# In[ ]:


from openai import OpenAI
client = OpenAI(api_key='enter_OpenAI_key')
# Get the API


# In[ ]:


# This is the syntax to send a automated request through the OpenAI's API
completion = client.chat.completions.create(
    temperature = 1, model="gpt-4o", # Think of temperature as creativity or acceptable threshold for token probability. Ranges 0 (low "creativity") to 2 (high "creativity"). Web version uses 1 as default.
    messages=[
        {
            "role": "user",
            "content": "Answer in 30 words: What would two squirrels argue about during winter?"
        }
    ]
)

print(completion.choices[0].message.content)


# In[ ]:


# Get a simple reply by describing the system (or what ChatGPT should pretend to be)
completion = client.chat.completions.create(
    temperature = 1, model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a 5-year old kid."},
        {
            "role": "user",
            "content": "Answer in 30 words: What would two squirrels argue about during winter?"
        }
    ]
)

print(completion.choices[0].message.content)


# In[ ]:


# Temprature is a "creativity" parameter. Setting a low temprature will lead to the expected (high probability) token being accepted. Leads to a more determined response.
completion = client.chat.completions.create(
    temperature = 0, model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a 5-year old kid."},
        {
            "role": "user",
            "content": "Answer in 30 words: What would two squirrels argue about during winter?"
        }
    ]
)

print(completion.choices[0].message.content)


# In[ ]:


# Temprature is a "creativity" parameter. Setting a high temprature will lead to "unexpected" (low probability) token being accepted. 
# Leads to a "garbage in, garbage out" scenerio.
completion = client.chat.completions.create(
    temperature = 1.5, model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a 5-year old kid."},
        {
            "role": "user",
            "content": "Answer in 30 words: What would two squirrels argue about during winter?"
        }
    ]
)

print(completion.choices[0].message.content)


# # News API

# In[ ]:


get_ipython().system('pip install newsapi')


# In[ ]:


### Get the NewsAPI
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key='enter_NewsAPI_key')


# In[ ]:


# Fetch articles, limiting the results to 30 articles
all_articles = newsapi.get_everything(q='AAPL',
                                      from_param='2024-09-23',
                                      to='2024-10-21',
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=30,  # Restricting to 30 articles
                                      page=1)




import pandas as pd
articles = all_articles['articles']
df = pd.json_normalize(articles)


df[:5]


# # Get Headline Labels from ChatGPT



# Function to interact with ChatGPT for financial analysis
def assess_headline(headline, company_name='Apple'):
    prompt = f"""Forget all your previous instructions. Pretend you are a financial expert. You are a
    financial expert with stock recommendation experience. Answer “YES” if good news,
    “NO” if bad news, or “UNKNOWN” if uncertain in the first line. Then elaborate with
    one short and concise sentence on the next line. Is this headline good or bad for the stock
    price of {company_name} in the short term?
    Headline: {headline}"""

    # Query OpenAI ChatGPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the first line (YES/NO/UNKNOWN) and the second line (explanation)
    content = response.choices[0].message.content.split('\n')
    answer = content[0]  # First line (YES/NO/UNKNOWN)
    explanation = content[1] if len(content) > 1 else ""  # Second line (explanation)

    return answer, explanation


# Iterate over each headline in the DataFrame and call ChatGPT for assessment
df[['sentiment', 'explanation']] = df['title'].apply(lambda headline: pd.Series(assess_headline(headline)))



# Export as CSV
df.to_csv('articles_sentiment_reason_4o.csv', encoding="utf-8-sig", index=False)

