import sys
import os

import numpy as np

import ollama

import feedparser

import diskcache

import requests
import requests_cache
import sqlitedict

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


rss_url = 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml'
# https://rss.nytimes.com/services/xml/rss/nyt/Americas.xml
requests_cache.install_cache(expire_after=3600, allowable_methods=('GET', 'POST'))
cache = diskcache.Cache('ollama_cache')

@cache.memoize()
def calc_embedding(s: str) -> list[float]:
  return ollama.embeddings(model='llama2', prompt=s)['embedding']

@cache.memoize()
def calc_embedding2(s: str) -> np.array:
  e = calc_embedding(s)
  e2 = e / np.linalg.norm(e)
  return np.array(e2, dtype=np.float32)

embeddings = []
titles = []

for rss_url in [
    'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Americas.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/US.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Science.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Health.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/MostEmailed.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/MostShared.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/MostViewed.xml',
    ]:
  feed = feedparser.parse(requests.get(rss_url).text)

  for i, entry in enumerate(feed.entries):
    title = entry.title
    url = entry.link
    summary = entry.summary
    titles.append(title + ' ' + summary)
    e2 = calc_embedding2(f'{title} {summary}')
    print(i, title)
    embeddings.append(e2)


# q = embeddings[0]
# distances = euclidean_distances(embeddings, [q])

# # Find the closest vector
# closest_index = np.argmin(distances)
# closest_vector = embeddings[closest_index]

# print('CLOSE: ', closest_index)
# print(closest_vector)
# print(np.linalg.norm(closest_vector))

# print()
# print('#')
# print()

for qi, q in enumerate(embeddings):
  distances = cosine_similarity(embeddings, [q])

  print("Q: ", titles[qi])
  i2 = np.squeeze(np.argsort(distances, axis=-2))[-2]
  print("\t", i2, titles[i2])
  print()
#  for i in np.squeeze(np.argsort(distances, axis=-2)):
#    print(i, titles[i])
#  print()
