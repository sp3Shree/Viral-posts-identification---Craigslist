# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request


class RegularForsaleSpider(scrapy.Spider):
    name = 'regular_forsale'
    allowed_domains = ['craigslist.org']
    start_urls = ["https://orlando.craigslist.org/search/sss", "https://columbus.craigslist.org/search/sss",
                  "https://columbia.craigslist.org/search/sss", "https://madison.craigslist.org/search/sss",
                  "https://longisland.craigslist.org/search/sss", "https://tippecanoe.craigslist.org/search/sss",
                  "https://houston.craigslist.org/search/sss", "https://honolulu.craigslist.org/search/sss",
                  "https://cosprings.craigslist.org/search/sss", "https://cleveland.craigslist.org/search/sss",
                  "https://cincinnati.craigslist.org/search/sss", "https://philadelphia.craigslist.org/search/sss",
                  "https://palmsprings.craigslist.org/search/sss", "https://nashville.craigslist.org/search/sss",
                  "https://chicago.craigslist.org/search/sss", "https://centralmich.craigslist.org/search/sss",
                  "https://york.craigslist.org/search/sss", "https://tucson.craigslist.org/search/sss",
                  "https://syracuse.craigslist.org/search/sss", "https://susanville.craigslist.org/search/sss",
                  "https://stjoseph.craigslist.org/search/sss", "https://stgeorge.craigslist.org/search/sss",
                  "https://springfield.craigslist.org/search/sss", "https://spokane.craigslist.org/search/sss",
                  "https://spacecoast.craigslist.org/search/sss", "https://swva.craigslist.org/search/sss",
                  "https://bigbend.craigslist.org/search/sss", "https://swmi.craigslist.org/search/sss",
                  "https://swv.craigslist.org/search/sss", "https://smd.craigslist.org/search/sss",
                  "https://carbondale.craigslist.org/search/sss", "https://semo.craigslist.org/search/sss",
                  "https://seks.craigslist.org/search/sss", "https://ottumwa.craigslist.org/search/sss",
                  "https://juneau.craigslist.org/search/sss", "https://southjersey.craigslist.org/search/sss",
                  "https://miami.craigslist.org/search/sss", "https://sd.craigslist.org/search/sss",
                  "https://sandiego.craigslist.org/search/sss", "https://saltlakecity.craigslist.org/search/sss",
                  "https://providence.craigslist.org/search/sss", "https://puertorico.craigslist.org/search/sss",
                  "https://portland.craigslist.org/search/sss", "https://muncie.craigslist.org/search/sss",
                  "https://indianapolis.craigslist.org/search/sss", "https://grandrapids.craigslist.org/search/sss",
                  "https://charleston.craigslist.org/search/sss", "https://pittsburgh.craigslist.org/search/sss",
                  "https://northmiss.craigslist.org/search/sss", "https://newjersey.craigslist.org/search/sss",
                  "https://nd.craigslist.org/search/sss", "https://losangeles.craigslist.org/search/sss",
                  "https://lafayette.craigslist.org/search/sss", "https://charlottesville.craigslist.org/search/sss",
                  "https://wyoming.craigslist.org/search/sss", "https://wichitafalls.craigslist.org/search/sss",
                  "https://wichita.craigslist.org/search/sss", "https://westernmass.craigslist.org/search/sss",
                  "https://westmd.craigslist.org/search/sss", "https://westky.craigslist.org/search/sss",
                  "https://washingtondc.craigslist.org/search/sss", "https://virgin.craigslist.org/search/sss",
                  "https://toledo.craigslist.org/search/sss", "https://sfbay.craigslist.org/search/sss",
                  "https://maine.craigslist.org/search/sss", "https://lawrence.craigslist.org/search/sss",
                  "https://lasvegas.craigslist.org/search/sss", "https://charlotte.craigslist.org/search/sss",
                  "https://baltimore.craigslist.org/search/sss", "https://atlanta.craigslist.org/search/sss"]

    def parse(self, response):
        post = {}
        for row in response.xpath('//p[@class="result-info"]'):
            post['date'] = row.xpath('time[@class="result-date"]//text()').get()
            post['title'] = row.xpath('a//text()').get()
            post['title_link'] = row.xpath('a/@href').get()
            post['region'] = row.xpath('span[@class="result-meta"]/span[@class="result-hood"]//text()').get("")[2:-1]
            post['price'] = row.xpath('span[@class="result-meta"]/span[@class="result-price"]//text()').get()
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