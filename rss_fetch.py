import json
import time

from absl import app
from absl import flags
from absl import logging

import feedparser
import requests
import requests_cache

import sqlitedict


def main(_):
  with open('rss_feeds.txt', 'r') as f:
    lines = f.read().split('\n')
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    print(lines)

  sess = requests_cache.CachedSession('rss_fetch_cache_hour',
                                      expire_after=3600,
                                      allowable_methods=('GET', 'POST'))

  with sqlitedict.open(filename='rss_fetch.sqlite',
                       flag='c',
                       encode=json.dumps,
                       decode=json.loads) as db:
    for url in lines:
      key = time.time()
      logging.info("key %s, url %s", key, url)
      res = feedparser.parse(sess.get(url).text)

      db[key] = res
    db.commit()


if __name__ == '__main__':
  app.run(main)
