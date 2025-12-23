# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

import os
import sys

sys.path.append(os.curdir)
from pelicanconf import *

# If your site is available via HTTPS, make sure SITEURL begins with https://
SITEURL = "https://FacuRoffet99.github.io"
RELATIVE_URLS = False

FEED_ALL_ATOM = "feeds/all.atom.xml"
CATEGORY_FEED_ATOM = "feeds/{slug}.atom.xml"

DELETE_OUTPUT_DIRECTORY = True

# 2. FORCE these settings again for Production
TYPOGRIFY = False
# 3. Ensure Plugin Paths are absolute to avoid build errors
PLUGIN_PATHS = [os.path.abspath('./plugins')]
PLUGINS = ['i18n_subsites', 'render_math']