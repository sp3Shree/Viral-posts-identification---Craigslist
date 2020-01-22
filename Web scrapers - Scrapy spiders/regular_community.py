# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request


class RegularCommunitySpider(scrapy.Spider):
    name = 'regular_community'
    allowed_domains = ['craigslist.org']
    start_urls = ["https://orlando.craigslist.org/search/ccc?", "https://columbus.craigslist.org/search/ccc?",
                  "https://columbia.craigslist.org/search/ccc?", "https://madison.craigslist.org/search/ccc?",
                  "https://longisland.craigslist.org/search/ccc?", "https://tippecanoe.craigslist.org/search/ccc?",
                  "https://houston.craigslist.org/search/ccc?", "https://honolulu.craigslist.org/search/ccc?",
                  "https://cosprings.craigslist.org/search/ccc?", "https://cleveland.craigslist.org/search/ccc?",
                  "https://cincinnati.craigslist.org/search/ccc?", "https://philadelphia.craigslist.org/search/ccc?",
                  "https://palmsprings.craigslist.org/search/ccc?", "https://nashville.craigslist.org/search/ccc?",
                  "https://chicago.craigslist.org/search/ccc?", "https://centralmich.craigslist.org/search/ccc?",
                  "https://york.craigslist.org/search/ccc?", "https://tucson.craigslist.org/search/ccc?",
                  "https://syracuse.craigslist.org/search/ccc?", "https://susanville.craigslist.org/search/ccc?",
                  "https://stjoseph.craigslist.org/search/ccc?", "https://stgeorge.craigslist.org/search/ccc?",
                  "https://springfield.craigslist.org/search/ccc?", "https://spokane.craigslist.org/search/ccc?",
                  "https://spacecoast.craigslist.org/search/ccc?", "https://swva.craigslist.org/search/ccc?",
                  "https://bigbend.craigslist.org/search/ccc?", "https://swmi.craigslist.org/search/ccc?",
                  "https://swv.craigslist.org/search/ccc?", "https://smd.craigslist.org/search/ccc?",
                  "https://carbondale.craigslist.org/search/ccc?", "https://semo.craigslist.org/search/ccc?",
                  "https://seks.craigslist.org/search/ccc?", "https://ottumwa.craigslist.org/search/ccc?",
                  "https://juneau.craigslist.org/search/ccc?", "https://southjersey.craigslist.org/search/ccc?",
                  "https://miami.craigslist.org/search/ccc?", "https://sd.craigslist.org/search/ccc?",
                  "https://sandiego.craigslist.org/search/ccc?", "https://saltlakecity.craigslist.org/search/ccc?",
                  "https://providence.craigslist.org/search/ccc?", "https://puertorico.craigslist.org/search/ccc?",
                  "https://portland.craigslist.org/search/ccc?", "https://muncie.craigslist.org/search/ccc?",
                  "https://indianapolis.craigslist.org/search/ccc?", "https://grandrapids.craigslist.org/search/ccc?",
                  "https://charleston.craigslist.org/search/ccc?", "https://pittsburgh.craigslist.org/search/ccc?",
                  "https://northmiss.craigslist.org/search/ccc?", "https://newjersey.craigslist.org/search/ccc?",
                  "https://nd.craigslist.org/search/ccc?", "https://losangeles.craigslist.org/search/ccc?",
                  "https://lafayette.craigslist.org/search/ccc?", "https://charlottesville.craigslist.org/search/ccc?",
                  "https://wyoming.craigslist.org/search/ccc?", "https://wichitafalls.craigslist.org/search/ccc?",
                  "https://wichita.craigslist.org/search/ccc?", "https://westernmass.craigslist.org/search/ccc?",
                  "https://westmd.craigslist.org/search/ccc?", "https://westky.craigslist.org/search/ccc?",
                  "https://washingtondc.craigslist.org/search/ccc?", "https://virgin.craigslist.org/search/ccc?",
                  "https://toledo.craigslist.org/search/ccc?", "https://sfbay.craigslist.org/search/ccc?",
                  "https://maine.craigslist.org/search/ccc?", "https://lawrence.craigslist.org/search/ccc?",
                  "https://lasvegas.craigslist.org/search/ccc?", "https://charlotte.craigslist.org/search/ccc?",
                  "https://baltimore.craigslist.org/search/ccc?", "https://atlanta.craigslist.org/search/ccc?"]

    def parse(self, response):
        post = {}
        for row in response.xpath('//p[@class="result-info"]'):
            post['date'] = row.xpath('time[@class="result-date"]//text()').get()
            post['title'] = row.xpath('a//text()').get()
            post['title_link'] = row.xpath('a/@href').get()
            post['region'] = row.xpath('span[@class="result-meta"]/span[@class="result-hood"]//text()').get("")[2:-1]
            low_rel_url = row.xpath('a/@href').get()
            low_url = response.urljoin(low_rel_url)
            yield Request(low_url, callback=self.parse_lower, meta=post)
        next_rel_url = response.xpath('//a[@class="button next"]/@href').get()
        next_url = response.urljoin(next_rel_url)
        yield Request(next_url, callback=self.parse)

    def parse_lower(self, response):
        text = "".join(line for line in response.xpath('//*[@id="postingbody"]/text()').getall())
        category = response.xpath('//li[@class="crumb category"]/p/a//text()').get()
        location = response.xpath('//div[@id="map"]').getall()
        response.meta['Text'] = text
        response.meta['category'] = category
        response.meta['location'] = location
        yield response.meta