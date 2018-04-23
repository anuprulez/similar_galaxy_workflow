import dryscrape
import re
import urllib
from bs4 import BeautifulSoup
from lxml import etree
from StringIO import StringIO

USEGALAXY_EU = "https://usegalaxy.eu"
CATEGORY = "category"
LABEL    = "label"
NAME     = "name"
URL      = "url"
result = {}

pp = pprint.PrettyPrinter(indent=4)

# visit usegalaxy.eu and handle its javascript calls
dryscrape.start_xvfb()
session = dryscrape.Session()
session.visit(USEGALAXY_EU)

if session.status_code() == 200:

    # correct eventually broken HTML opening/closing tags
    response = BeautifulSoup(session.body(), "lxml")
    html = response.prettify().encode('utf-8')

    # parse the retrieved HML
    parser = etree.HTMLParser()
    tree = etree.parse(StringIO(html), parser)

    # retrieve each tool section
    tool_sections = tree.xpath("//div[@class = 'toolSectionWrapper']")

    for tool_section in tool_sections:

        category = None
        label    = None
        name     = None
        url      = None

        # retrieve the tool section's category
        category = tool_section.xpath("./div[@class = 'toolSectionTitle']/a/span/text()")[0].rstrip().lstrip()

        # retrieve the tool list
        tools = tool_section.xpath("./div[@class = 'toolSectionBody']/div[@class = 'toolTitle']")

        for tool in tools:

            # retrieve the tool's label
            label = tool.xpath(".//a/span[@class = 'tool-old-link']/text()")[0].rstrip().lstrip()

            # retrieve the tool's URL
            url = re.sub("^.*tool_id=", "", urllib.unquote(tool.xpath(".//a/@href")[0].rstrip().lstrip()) )

            # retrieve the tool ONLY if its URL refers to the toolshed
            if re.match("^.*toolshed", url) :

                # retrieve the tool's name
                name = url.split("/")[::-1][1]

                # track the new tool entry
                if name not in result.keys():
                    result[name] = { url: {LABEL: label, CATEGORY: category}}
                else:
                    result[name][url] = {LABEL: label, CATEGORY: category}

