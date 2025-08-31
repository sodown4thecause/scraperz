import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapegraphai.graphs import SmartScraperGraph

class IntelligentSpider(scrapy.Spider):
    name = 'intelligent'

    def __init__(self, *args, **kwargs):
        super(IntelligentSpider, self).__init__(*args, **kwargs)
        self.start_urls = [kwargs.get('start_url')]
        self.prompt = kwargs.get('prompt')
        self.llm_config = kwargs.get('llm_config')
        self.results_list = kwargs.get('results_list')
        self.link_extractor = LinkExtractor()

    async def parse(self, response):
        # Use ScrapeGraphAI on the current page
        smart_scraper_graph = SmartScraperGraph(
            prompt=self.prompt,
            source=response.body.decode('utf-8'), # Pass the HTML content
            config=self.llm_config
        )
        result = smart_scraper_graph.run()
        self.results_list.append({'url': response.url, 'data': result})

        # Follow links to next pages
        links = self.link_extractor.extract_links(response)
        for link in links:
            yield response.follow(link, self.parse)
