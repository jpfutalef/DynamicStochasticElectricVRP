import xml.etree.ElementTree as ET
import xml.dom.minidom
from typing import NamedTuple, Union, Dict


class OptimizationReport(NamedTuple):
    best_fitness: float
    feasible: bool
    execution_time: float
    # hyper_parameters: Dict[str, float]


def save_optimization_report(path, report: OptimizationReport, pretty=False) -> None:
    tree = ET.parse(path)
    _info = tree.find('info')
    for _op_info in _info.findall('optimization_info'):
        _info.remove(_op_info)
    attrib = {'best_fitness': str(report.best_fitness),
              'feasible': str(report.feasible),
              'execution_time':str(report.execution_time)}
    _op_info = ET.SubElement(_info, 'optimization_info', attrib=attrib)

    tree.write(path)
    if pretty:
        write_pretty_xml(path)
    return


def read_optimization_report(path, tree=None) -> Union[OptimizationReport, None]:
    if not tree:
        tree = ET.parse(path)
    _info = tree.find('info')
    _report = _info.find('optimization_info')
    if _report is not None:
        report = OptimizationReport(float(_report.get('best_fitness')), bool(_report.get('feasible')),
                                    float(_report.get('execution_time')))
        return report
    return None


def write_pretty_xml(path):
    xml_pretty = xml.dom.minidom.parse(path).toprettyxml()
    with open(path, 'w') as file:
        file.write(xml_pretty)
