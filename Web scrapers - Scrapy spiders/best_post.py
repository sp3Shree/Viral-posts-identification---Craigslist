# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request


class BestPostSpider(scrapy.Spider):
    name = 'best_post'
    allowed_domains = ['craigslist.org']
    start_urls = ['https://www.craigslist.org/about/best/all/']

    def parse(self, response):
        post = {}
        for row in response.xpath('//tr'):
            post['date'] = row.xpath('td[1]//text()').get()
            post['title'] = row.xpath('td[2]//text()').get()
            post['title_link'] = row.xpath('td[2]/a/@href').get()
            post['category'] = row.xpath('td[3]//text()').get()
            post['region'] = row.xpath('td[4]//text()').get()
            low_rel_url = row.xpath('td[2]/a/@href').get()
            low_url = response.urljoin(low_rel_url)
            yield Request(low_url, callback=self.parse_lower, meta=post)
        next_rel_url = response.xpath('//a[@class="button next"]/@href').get()
        next_url = response.urljoin(next_rel_url)
        yield Request(next_url, callback=self.parse)

    def parse_lower(self, response):
        text = "".join(line for line in response.xpath('//*[@id="postingbody"]/text()').getall())
        response.meta['Text'] = text
        yield response.meta