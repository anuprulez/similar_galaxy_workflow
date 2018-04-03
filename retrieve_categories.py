#"//div[@class = 'toolSectionWrapper']/div[@class = 'toolSectionTitle']/text()") # takes the category
#"//div[@class = 'toolSectionWrapper']//div[@class = 'toolTitle']//a/text()") # takes a tool's label
#"//div[@class = 'toolSectionWrapper']//div[@class = 'toolTitle']//a/@href") # takes a tool's URL

import dryscrape
import urllib
from bs4 import BeautifulSoup
from lxml import etree
from StringIO import StringIO

USEGALAXY_EU = "https://usegalaxy.eu"

# visit usegalaxy.eu and handle its javascript calls
dryscrape.start_xvfb()
session = dryscrape.Session()
session.visit(USEGALAXY_EU)

if session.status_code() == 200:

    # correct ieventually broken HTML opening/closing tags
    response = BeautifulSoup(session.body(), "lxml")
    html = response.prettify().encode('utf-8')

    # parse the retrieved HML
    parser = etree.HTMLParser()
    tree = etree.parse(StringIO(html), parser)

    # retrieve each tool's URL
    tools = tree.xpath("//div[@class = 'toolSectionWrapper']//div[@class = 'toolTitle']//a/@href")


    for tool in tools:
        t.append(urllib.unquote(tool))

