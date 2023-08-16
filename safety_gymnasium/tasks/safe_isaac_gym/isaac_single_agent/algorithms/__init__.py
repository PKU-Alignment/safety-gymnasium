from .cpo import CPO
from .focops import FOCOPS
from .p3o import P3O
from .pcpo import PCPO
from .ppol import PPOL
from .trpol import TRPOL


REGISTRY = {'cpo': CPO, 'ppol': PPOL, 'trpol': TRPOL, 'focops': FOCOPS, 'pcpo': PCPO, 'p3o': P3O}
