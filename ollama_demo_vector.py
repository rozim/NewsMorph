import sys
import os
import time

import numpy as np

import ollama

import feedparser

import diskcache

import requests
import requests_cache
import sqlitedict

from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import euclidean_distances


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

def get_url(entry):
  if 'links' in entry:
    return entry.links[0].href
  return entry.link


embeddings = []
titles = []

already = set()
for rss_url in [
    'https://feeds.bbci.co.uk/news/rss.xml',
    'http://rss.cnn.com/rss/cnn_topstories.rss',
    'http://rss.cnn.com/services/podcasting/cnn10/rss.xml',
    'https://nypost.com/feed/',
    'https://www.chicagotribune.com/arc/outboundfeeds/rss/section/news_breaking/range/display_date/now-5d/now/?outputType=xml&size=50',
    'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
    'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml',
    'http://america.aljazeera.com/content/ajam/articles.rss',
    'https://www.washingtonpost.com/arcio/rss/category/politics/',
    'https://www.washingtonpost.com/arcio/rss/category/opinions/',
    'https://feeds.washingtonpost.com/rss/national',
    'http://feeds.washingtonpost.com/rss/world',
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
    url = get_url(entry)
    summary = entry.get('summary', '')

    print(i, '#', url, '#', title, '#', summary)
    if title in already:
      continue
    already.add(title)

    titles.append(title + ' ' + summary)
    e2 = calc_embedding2(f'{title} {summary}')
    print(i, title)
    embeddings.append(e2)


t1 = time.time()
mega_distances = cosine_similarity(embeddings, embeddings)
t2 = time.time()
print('dt', t2-t1, 'e', len(embeddings))

print()
for qi, q in enumerate(embeddings):
  #distances = cosine_similarity(embeddings, [q])
  distances = mega_distances[qi]
  print('ds', distances.shape)

  print("Q: ", qi, "     ", titles[qi])
#  i2 = np.squeeze(np.argsort(distances, axis=-2))[-2]
  i2 = np.argsort(distances, axis=-1)[-2]
  print("\t", i2, titles[i2])
  print()
#  for i in np.squeeze(np.argsort(distances, axis=-2)):
#    print(i, titles[i])
#  print()
