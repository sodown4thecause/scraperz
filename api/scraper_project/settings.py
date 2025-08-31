BOT_NAME = 'scraper_project'

SPIDER_MODULES = ['scraper_project.spiders']
NEWSPIDER_MODULE = 'scraper_project.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 1

# Set a crawl depth limit
DEPTH_LIMIT = 2

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# DefaultRequestHeaders setting is now located in the spider

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
