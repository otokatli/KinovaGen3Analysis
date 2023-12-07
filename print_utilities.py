import re
from sympy.printing.cxx import CXX11CodePrinter


def add_std(s: str) -> str:
    s_1 = s.replace('cos', 'std::cos')

    return s_1.replace('sin', 'std::sin')

def matrix_to_string(expr) -> str:
    return [r.strip("[]").strip() for r in re.findall(r'\[.*?\]', CXX11CodePrinter().doprint(expr))]

def matrix_to_eigen_stream(M, variable_name: str) -> str:

    ss = ',\n\t\t\t'
    M_str = [str(m) for m in M]
    
    return f'\t{variable_name} << {ss.join(M_str)};'
