from typing import Dict, List, Tuple, Union, NamedTuple

RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]