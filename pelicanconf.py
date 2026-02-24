# pelican content
# pelican --listen

import os
import sys
sys.path.append(os.curdir)
# from pelicanconf import *


# =============================================================================
# SITE IDENTITY
# =============================================================================

AUTHOR = 'Facundo Roffet'
SITENAME = 'Facundo Roffet'
SITETITLE = 'Facundo Roffet'
SITESUBTITLE = 'Investigador en Deep Learning<br>UNS–CONICET'
SITEDESCRIPTION = 'Portfolio de Facundo Roffet, Ingeniero Electrónico e investigador doctoral (CONICET). Especialista en Inteligencia Artificial, Deep Learning y Computer Vision.'
SITELOGO = 'https://FacuRoffet99.github.io/images/profile.png'
FAVICON = '/favicon.ico'


# =============================================================================
# URLs AND PATHS
# =============================================================================

# SITEURL = "http://127.0.0.1:8000"
SITEURL = "https://FacuRoffet99.github.io"
RELATIVE_URLS = False

PATH = "content"
STATIC_PATHS = ['images', 'static', 'extra']
EXTRA_PATH_METADATA = {
    'images/icon.ico': {'path': 'favicon.ico'},
    'extra/robots.txt': {'path': 'robots.txt'},
    'extra/llms.txt': {'path': 'llms.txt'},
}

# Article URLs and save paths
ARCHIVES_URL = 'notes/'
ARCHIVES_SAVE_AS = 'notes/index.html'
ARTICLE_PATHS = ['notes']
ARTICLE_URL = 'notes/{slug}.html'
ARTICLE_SAVE_AS = 'notes/{slug}.html'

# Page URLs and save paths
PAGE_URL = '{slug}.html'
PAGE_SAVE_AS = '{slug}.html'
INDEX_SAVE_AS = ''


# =============================================================================
# THEME AND APPEARANCE
# =============================================================================

THEME = 'themes/Flex'
THEME_COLOR = 'dark'
THEME_TEMPLATES_OVERRIDES = ['templates']
CUSTOM_CSS = 'static/css/custom.css'


# =============================================================================
# NAVIGATION
# =============================================================================

LINKS = (
    ("Notas de cursos", "/notes/"),
)


# =============================================================================
# LOCALIZATION AND INTERNATIONALIZATION
# =============================================================================

TIMEZONE = 'America/Argentina/Buenos_Aires'
DEFAULT_LANG = 'es'
OG_LOCALE = 'es_AR'

LOCALE = [
    'es_AR.UTF-8', 'es_ES.UTF-8', 'es.UTF-8',
    'en_US.UTF-8', 'en_GB.UTF-8', 'en.UTF-8',
    '' #  fallback
]

# Translation settings
LANGUAGES = (('es', 'Español'), ('en', 'English'))
ARTICLE_TRANSLATION_ID = 'slug'
PAGE_TRANSLATION_ID = 'slug'
ARTICLE_LANG_URL = '{slug}-{lang}.html'
ARTICLE_LANG_SAVE_AS = '{slug}-{lang}.html'
PAGE_LANG_URL = '{slug}-{lang}.html'
PAGE_LANG_SAVE_AS = '{slug}-{lang}.html'

# English subsite overrides
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}
I18N_SUBSITES = {
    'en': {
        'LOCALE': 'en_US.UTF-8',
        'SITESUBTITLE': 'Deep Learning Researcher<br>UNS–CONICET',
        'SITEDESCRIPTION': 'Portfolio of Facundo Roffet, Electronic Engineer and doctoral researcher (CONICET). Specialist in Artificial Intelligence, Deep Learning and Computer Vision.',
        'OG_LOCALE': 'en_US',
        'LINKS': (
            ("Course notes", "/notes/"),
        ),
    },
}


# =============================================================================
# PLUGINS
# =============================================================================

PLUGINS = [
    'pelican.plugins.i18n_subsites',  # Multilingual subsites
    'pelican.plugins.render_math',    # LaTeX math rendering
    'pelican.plugins.sitemap',        # XML sitemap generation
]

SITEMAP = {
    'format': 'xml',
    'priorities': {'articles': 0.8, 'pages': 1.0, 'indexes': 0.5},
    'changefreqs': {'articles': 'monthly', 'pages': 'monthly', 'indexes': 'weekly'},
}

MATH_JAX = {
    'auto_insert': True,
    'responsive': True,
    'message_style': 'none'
}


# =============================================================================
# SEO AND METADATA
# =============================================================================

ROBOTS = 'index, follow'
REL_CANONICAL = True


# =============================================================================
# FEEDS
# =============================================================================

FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None


# =============================================================================
# MISCELLANEOUS
# =============================================================================

DEFAULT_PAGINATION = False
TYPOGRIFY = False
COPYRIGHT_NAME = AUTHOR
COPYRIGHT_YEAR = '2025'