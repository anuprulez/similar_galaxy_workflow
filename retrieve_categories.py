# module for the retrieval of tools and tools' attributes from
# usegalaxy.eu web interface

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

# XPATH queries for retrieving tool attributes from the usegalaxy.eu
# web interface
XPATH_SECTION  = "//div[@class = 'toolSectionWrapper']"
XPATH_CATEGORY = "./div[@class = 'toolSectionTitle']/a/span/text()"
XPATH_TOOLS    = "./div[@class = 'toolSectionBody']/div[@class = 'toolTitle']"
XPATH_LABEL    = ".//a/span[@class = 'tool-old-link']/text()"
XPATH_URL      = ".//a/@href"


# retrieve the tools catalog
#
def get_tool_catalog():
    tool_catalog = None

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
        tool_catalog = tree.xpath(XPATH_SECTION)

    return tool_catalog



# retrieve all attributes of each tool
#
def get_attributes():
    result = {}

    tool_catalog = get_tool_catalog()

    for section in tool_catalog:

        category = None
        label    = None
        name     = None
        url      = None

        # retrieve the tool section's category
        category = section.xpath(XPATH_CATEGORY)[0].rstrip().lstrip()

        # retrieve the tool list
        tools = section.xpath(XPATH_TOOLS)

        for tool in tools:

            # retrieve the tool's label
            label = tool.xpath(XPATH_LABEL)[0].rstrip().lstrip()

            # retrieve the tool's URL
            url = re.sub("^.*tool_id=", "", urllib.unquote(tool.xpath(XPATH_URL)[0].rstrip().lstrip()) )

            # retrieve the tool ONLY if its URL refers to the toolshed
            if re.match("^.*toolshed", url) :

                # retrieve the tool's name
                name = url.split("/")[::-1][1]

                # track the new tool entry
                if name not in result.keys():
                    result[name] = { url: {LABEL: label, CATEGORY: category}}
                else:
                    result[name][url] = {LABEL: label, CATEGORY: category}

    return result



# retrieve only the names and categories of each tool
#
def get_names_categories():
    result = {}

    tool_catalog = get_tool_catalog()

    for section in tool_catalog:

        category = None
        name     = None

        # retrieve the tool section's category
        category = section.xpath(XPATH_CATEGORY)[0].rstrip().lstrip()

        # retrieve the tool list
        tools = section.xpath(XPATH_TOOLS)

        for tool in tools:

            # retrieve the tool's URL
            url = re.sub("^.*tool_id=", "", urllib.unquote(tool.xpath(XPATH_URL)[0].rstrip().lstrip()) )

            # retrieve the tool ONLY if its URL refers to the toolshed
            if re.match("^.*toolshed", url) :

                # retrieve the tool's name
                name = url.split("/")[::-1][1]

                # track the new tool entry
                if name not in result.keys():
                    result[name] = [category]
                else:
                    categories = result[name]
                    categories.append(category)
                    result[name] = categories

    return result

if __name__ == '__main__':
    result = get_attributes()
    print result
