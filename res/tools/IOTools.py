import xml.dom.minidom


def write_pretty_xml(path):
    xml_pretty = xml.dom.minidom.parse(path).toprettyxml()
    with open(path, 'w') as file:
        file.write(xml_pretty)
