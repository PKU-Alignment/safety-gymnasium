from algorithms.cpo import CPO
from algorithms.focops import FOCOPS
from algorithms.p3o import P3O
from algorithms.pcpo import PCPO
from algorithms.ppol import PPOL
from algorithms.trpol import TRPOL


REGISTRY = {
    'cpo': CPO,
    'ppol': PPOL,
    'trpol': TRPOL,
    'focops': FOCOPS,
    'pcpo': PCPO,
    'p3o': P3O,
}
