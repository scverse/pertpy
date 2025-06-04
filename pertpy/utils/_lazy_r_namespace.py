from types import SimpleNamespace

def lazy_import_r_env():
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr, STAP
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter, pandas2ri, numpy2ri

    from pertpy.utils.rpy2_utils import _py_to_r, _r_to_py, lazy_import_r_packages  # moved inside function

    r = SimpleNamespace(
        ro=ro,
        importr=importr,
        STAP=STAP,
        localconverter=localconverter,
        default_converter=default_converter,
        pandas2ri=pandas2ri,
        numpy2ri=numpy2ri,
        lazy_import_r_packages=lazy_import_r_packages
    )

    return r, _py_to_r, _r_to_py