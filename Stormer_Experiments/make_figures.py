# make_figures.py
import physq_path

import os
import sys

_AURORA = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       os.pardir, "Aurora_Experiments"))
if _AURORA not in sys.path:
    sys.path.insert(0, _AURORA)

from make_figures import main, with_defaults

_DEFAULTS = [
    ("--aurora-root", _AURORA),
    ("--stormer-root", "."),
    ("--outdir", "figures"),
]

if __name__ == "__main__":
    main(with_defaults(sys.argv[1:], _DEFAULTS))
