import xml.etree.cElementTree as ET
import numpy as np


def parse_xml(source):
    root = ET.parse(str(source)).getroot()
    return parse_xmlelem(root)


def parse_xmlelem(elem):
    results = {}
    children = set(c.tag for c in elem)
    for child in children:
        child_elems = elem.findall(child)
        child_results = []
        for child_elem in child_elems:
            if 'type' in child_elem.attrib:
                if 'size' in child_elem.attrib:
                    child_elem_results = parse_xmlarr(child_elem)
                else:
                    child_elem_results = float(child_elem.text)
            elif len(list(child_elem)):
                child_elem_results = parse_xmlelem(child_elem)
            else:
                child_elem_results = child_elem.text.strip()
            child_results.append(child_elem_results)
        if len(child_results) == 1:
            results[child] = child_results[0]
        else:
            results[child] = child_results
    return results


def parse_xmlarr(xmlarr, axis=None, typef=None):
    if axis is None:
        axis = len(xmlarr.attrib['size'].split())-1
    if not typef:
        typename = xmlarr.attrib['type']
        if typename == 'dble' or typename == 'real':
            typef = float
        elif typename == 'int':
            typef = int
        else:
            raise Exception('Unknown array type')
    if axis > 0:
        lst = [parse_xmlarr(v, axis-1, typef)[..., None]
               for v in xmlarr.findall('vector')]
        return np.concatenate(lst, axis)
    else:
        return np.array([typef(x) for x in xmlarr.text.split()])
