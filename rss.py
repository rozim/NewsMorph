
from PIL import Image

import base64
import html
import os
import re
import secrets

import sys

from absl import app
from absl import flags
from absl import logging

import feedparser

import requests
import requests_cache
import sqlitedict

FLAGS = flags.FLAGS

flags.DEFINE_integer('max_entries', 15, '')
flags.DEFINE_integer('num_prompts', 6, '')

# Docs
# https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage

rss_url = 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml'
# url = 'World.xml'

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

def shrink_image(path):
  path = path[:200] # Avoid path limit.
  goal = '_512x512.png'
  if not path.endswith(goal):
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

  return normalized


def generate_images(prompt: str, samples=1):
  """Generate image from prompt, return path."""
  global API_KEY

  response = requests.post(
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
    raise Exception("Non-200 response: " + str(response.text))

  data = response.json()

  paths = []
  for i, image in enumerate(data["artifacts"]):
    path = "images/" + normalize_filename(prompt) + f"_{i}_512x512.png"
    paths.append(path)
    with open(path, "wb") as f:
      f.write(base64.b64decode(image["base64"]))
  return paths


def do_feed_entry(entry):
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

  for prompt in rnd.sample(prompts, FLAGS.num_prompts):
    print(prompt)
    paths = generate_images(prompt)
    yield (prompt, paths)



def escape_url(url):
  return html.escape(url, quote=True)

def escape_str(s):
  return html.escape(s)

def do_feed(url: str):
  feed = feedparser.parse(requests.get(rss_url).text)
  print("Feed title:", feed.feed.title)
  body = []
  for i, entry in enumerate(feed.entries):
    # print("Title:", entry.title)
    # print("Link:", entry.link)
    # print("Summary:", entry.summary)

    title = entry.title
    url = entry.link
    summary = entry.summary
    body.append(f'<a href="{escape_url(url)}">{escape_str(title)}</a>')
    body.append('<br/>')
    body.append(f'<a href="{escape_url(url)}"><em class=sm>{escape_str(summary)}</em></a>')
    body.append('<br/>')
    try:
      for (prompt, paths) in do_feed_entry(entry):
        for p in paths:
          sm = shrink_image(p)
          body.append(f'<a href="{p}"><img width=256 height=256 border=0 title="{escape_str(prompt)}" src="{escape_str(sm)}"></a>')
    except Exception as e:
      print("OUCH: ", e)
      pass
    body.append('<br/>')
    body.append('<p>')
    body.append("")
    if i >= FLAGS.max_entries:
      break
  return '\n'.join(body)


def main(_):
  global API_KEY
  requests_cache.install_cache(expire_after=3600, allowable_methods=('GET', 'POST'))
  with open(os.path.expanduser("~/.stability.ai.secret.txt"), 'r') as f:
    API_KEY = f.read().strip()

  body = do_feed(rss_url)
  with open('index.html', 'w') as fp:
    fp.write(HTML5.format(title='newsmorph', body=body))




if __name__ == '__main__':
  app.run(main)
