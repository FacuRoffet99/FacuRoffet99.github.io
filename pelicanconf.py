# pelican content
# pelican --listen
# pelican content -s publishconf.py -o output

# VER IDIOMAS
# PUBLICAR

import os
import sys
sys.path.append(os.curdir)
from pelicanconf import *

AUTHOR = 'Facundo Roffet'
SITENAME = 'Facundo Roffet'
# SITEURL = "http://127.0.0.1:8000"
SITEURL = "https://FacuRoffet99.github.io"
RELATIVE_URLS = False

PATH = "content"

TIMEZONE = 'America/Argentina/Buenos_Aires'

DEFAULT_LANG = 'es'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/{slug}.atom.xml'
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (
    ("Inicio", "/"),
    ("Notas de cursos", "/notes/"),
)

DEFAULT_PAGINATION = False

ARCHIVES_URL = 'notes/'
ARCHIVES_SAVE_AS = 'notes/index.html'
ARTICLE_PATHS = ['notes']
ARTICLE_SAVE_AS = 'notes/{slug}.html'
ARTICLE_URL = 'notes/{slug}.html'
COPYRIGHT_NAME = AUTHOR
COPYRIGHT_YEAR = '2025'
CUSTOM_CSS = 'static/css/custom.css'
DISQUS_SITENAME = "tu-shortname-de-disqus"
EXTRA_PATH_METADATA = {'images/icon.ico': {'path': 'favicon.ico'}}
FAVICON = '/favicon.ico'
INDEX_SAVE_AS = ''
PAGE_SAVE_AS = '{slug}.html'
PAGE_URL = '{slug}.html'
PAGE_PATHS = ['pages']
SITELOGO = '/images/profile.png'
STATIC_PATHS = ['images', 'static']
THEME = 'themes/Flex'
THEME_COLOR = 'dark'
THEME_TEMPLATES_OVERRIDES = ['templates']