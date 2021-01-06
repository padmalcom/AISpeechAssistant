from PyInstaller.compat import modname_tkinter
import os

hiddenimports = ['sklearn', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils', 'snips_nlu', 'multiprocessing.get_context', 'sklearn.utils', 'pycrfsuite._dumpparser', 'pycrfsuite._logparser']

binaries=[('dylib\\libsnips_nlu_parsers_rs.cp38-win_amd64.pyd', 'dylib')]