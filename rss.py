
from PIL import Image

import base64
import dataclasses
import html
import itertools
import os
import re
import secrets
import shutil
import time
import traceback

import sys

from absl import app
from absl import flags
from absl import logging

import feedparser

import requests
import requests_cache

import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_integer('max_entries', 15, '')
flags.DEFINE_integer('num_prompts', 6, '')
flags.DEFINE_string('outdir', 'site', 'Output directory')

# Docs
# https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage

RSS_FEEDS = [
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
  'https://rss.nytimes.com/services/xml/rss/nyt/MostViewed.xml'
]



# url = 'World.xml'

# Stability.ai
ENGINE_ID = "stable-diffusion-v1-6"
API_HOST = os.getenv('API_HOST', 'https://api.stability.ai')
API_KEY = None

HTML5 = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{title}</title>
    <link rel="stylesheet" href="style.css">
  </head>
  <body>
    {body}
  </body>
</html>
"""

requests_rss = None
requests_stability_ai = None

def strip_dir_prefix(s, pre):
  assert s.startswith(pre)
  return s[len(pre) + 1:] # +1 -> slash

def shrink_image(path):
  # path = path[:200] # Avoid path limit.
  goal = '_512x512.png'
  if not path.endswith(goal):
    assert False, (path, goal)
    return None
  img = Image.open(path)
  path2 = path[:-len(goal)] + "_256x256.png"
  img2 = img.resize((256, 256))
  img2.save(path2)
  return path2

def normalize_filename(string):
  """Normalizes a string to a valid file name, ensuring only a single underscore between words."""

  # Replace invalid characters with underscores
  pattern = r"[^\w\s\.\-]+"
  normalized = re.sub(pattern, "_", string)

  # Remove leading and trailing spaces
  normalized = normalized.strip()

  # Convert to lowercase
  normalized = normalized.lower()

  normalized = re.sub(r" +", "_", normalized)
  # Replace repeated underscores with a single one
  normalized = re.sub(r"_+", "_", normalized)

  return normalized[:200] # avoid file len problems


def generate_images(prompt: str, samples=1):
  """Generate image from prompt, return path."""
  global API_KEY
  global requests_stability_ai

  response = requests_stability_ai.post(
    f"{API_HOST}/v1/generation/{ENGINE_ID}/text-to-image",
    headers={
      "Content-Type": "application/json",
      "Accept": "application/json",
      "Authorization": f"Bearer {API_KEY}",
      "Stability-Client-ID": "newsmorph",
      "Stability-Client-Version": "0.01a",
    },
    json={
        "text_prompts": [
            {
                "text": prompt,
            }
        ],
        "cfg_scale": 7,
        "height": 512,
        "width": 512,
        "samples": samples,
        "steps": 30,
    },
  )

  if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text) + "response: " + str(response) + " prompt: " + prompt)

  data = response.json()

  paths = []
  for i, image in enumerate(data["artifacts"]):
    path = os.path.join(FLAGS.outdir, "images", normalize_filename(prompt) + f"_{i}_512x512.png")
    paths.append(path)
    with open(path, "wb") as f:
      f.write(base64.b64decode(image["base64"]))
  return paths


def do_feed_entry(entry, logf):
  global API_KEY

  rnd = secrets.SystemRandom()

  title, summary = entry.title, entry.summary

  prompts = [f'{title} {summary}']
  for what in [title, summary]:
    prompts.extend([
      what,
      f"Black and white sketch for news article with title: {what}",
      f"Low angle, wide angle lens, photo of: {what}",
      f"High angle overview photo of: {what}",
      f"Artist sketch of news story: {what}",
      f"Editorial cartoon of news story: {what}",
      f"Charcoal sketch of news story: {what}",
      f"Pencil sketch of news story: {what}",
      f"Artist rendering of news story: {what}",
      f"Watercolor of news story: {what}",
      f"Dramatic painting of news story: {what}",
      f"Close up image of of news story: {what}",
      f"Map summarizing news story: {what}",
      f"Highly detailed rendering of news story: {what}",
      f"DLSR photo of: {what}",
      f"Surrealist painting of {what}",
      f"Gritty photography of {what}",
      f"Impressionist painting of {what}",
      f"Pop art painting of {what}",
      f"Street photography of {what}",
      f"Aerial photography of {what}",
      f"Vintage photography of {what}",
      f"Documentary-style photography photography of {what}",
      f"Isometric digital art of {what}",
      f"Steampunk digital art of {what}",
      f"Diagram for news: {what}",
    ])

  for prompt in tqdm.tqdm(rnd.sample(prompts, FLAGS.num_prompts), leave=False):
    t1 = time.time()
    paths = generate_images(prompt)
    logf.write(f'\t\tdt: {time.time() - t1:.1f}s\n')
    yield (prompt, paths)



def escape_url(url):
  return html.escape(url, quote=True)


def escape_str(s):
  assert s is not None
  return html.escape(s)


@dataclasses.dataclass
class Entry:
  title: str
  summary: str
  url: str

def get_all_feed_entries():
  global requests_rss

  entries = []
  for url in tqdm.tqdm(RSS_FEEDS):
    feed = feedparser.parse(requests_rss.get(url).text)
    for entry in feed.entries:
      entries.append(Entry(title=entry.title,
                           summary=entry.summary,
                           url=entry.link # TBD, may need to look at alt href for washpost
                           ))
  return entries


def do_feed(entries : list[Entry], logf):
  global requests_rss

  body = []
  for i, entry in enumerate(tqdm.tqdm(entries)):
    title = entry.title
    url = entry.url
    summary = entry.summary
    logf.flush()
    logf.write(f'\tENTRY {i}. "{title}" | "{summary}" | {url}\n')

    body.append(f'<a href="{escape_url(url)}">{escape_str(title)}</a>')
    body.append('<br/>')
    body.append(f'<a href="{escape_url(url)}"><em class=sm>{escape_str(summary)}</em></a>')
    body.append('<br/>')

    # Generate images
    prompt, paths = None, None
    for (prompt, paths) in do_feed_entry(entry, logf):
      logf.write(f'\t\tIMAGE "{prompt}" -> {paths}\n')
      for p in paths:
        sm = shrink_image(p)
        assert prompt is not None
        assert sm is not None
        p = strip_dir_prefix(escape_str(p), FLAGS.outdir)
        body.append(f'<a href="{p}"><img width=256 height=256 border=0 title="{escape_str(prompt)}" src="{strip_dir_prefix(escape_str(sm), FLAGS.outdir)}"></a>')
    logf.write('\n')
    body.append('<br/>')
    body.append('<p>')
    body.append("")
  return '\n'.join(body)


def install_caches():
  global requests_rss, requests_stability_ai
  hour_session = requests_cache.CachedSession('http_cache_hour',
                                              expire_after=3600,
                                              allowable_methods=('GET', 'POST'))

  month_session = requests_cache.CachedSession('http_cache_month',
                                              expire_after=(30 * 24 * 3600),
                                              allowable_methods=('GET', 'POST'))

  requests_rss = hour_session
  requests_stability_ai = month_session

def main(_):
  global API_KEY

  assert os.path.isdir(FLAGS.outdir)
  install_caches()
  shutil.copy2('style.css', FLAGS.outdir)
  entries = get_all_feed_entries()
  entries = secrets.SystemRandom().sample(entries, FLAGS.max_entries)


  # requests_cache.install_cache(expire_after=3600, allowable_methods=('GET', 'POST'))
  with open(os.path.expanduser("~/.stability.ai.secret.txt"), 'r') as f:
    API_KEY = f.read().strip()

  with open(os.path.join(FLAGS.outdir, 'log.txt'), 'w') as logf:
    body = do_feed(entries, logf)
    with open(os.path.join(FLAGS.outdir, 'index.html'), 'w') as fp:
      fp.write(HTML5.format(title='newsmorph', body=body))


if __name__ == '__main__':
  app.run(main)
