import os
from sympy import cse
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.cxx import CXX11CodePrinter
from sympy.printing.julia import JuliaCodePrinter


class Function:
    def __init__(self, name, path, input_args, output_args, expr):
        self.function_name = name
        self.file_path = path
        self.input_args = input_args
        self.output_args = output_args
        self.expr = cse(expr)
        self.code_printer = None

        self.file_handle = None

    def open_file(self, file_extension):
        self.file_handle = open(os.path.join(self.file_path, self.function_name + f'.{file_extension}'), 'w')

    def close_file(self):
        self.file_handle.close()

    def _four_spaces(self):
        return '    '

    def _two_spaces(self):
        return '  '

    def _tab_space(self):
        return '\t'

    def _newline(self):
        return '\n'


class PythonFunction(Function):
    def __init__(self, name, path, input_args, output_args, expr):
        super().__init__(name, path, input_args, output_args, expr)
        self.code_printer = NumPyPrinter

    def print(self):
        self.open_file('py')
        print(f'Printing the function: {self.function_name}')
        print(f'{self._import_str()}', file=self.file_handle)
        print(f'def {self.function_name}({self._input_argument_str()}):', file=self.file_handle)
        # print(self._q_array_to_local_var(), file=self.file_handle)
        for input_arg in self.input_args:
            print(self._array_to_local_var(input_arg), file=self.file_handle)
        print(self._auto_x_to_code(), file=self.file_handle)
        print(self._output_statement(), file=self.file_handle)
        print(self._return_statement(), file=self.file_handle)
        self.close_file()

    def _import_str(self):
        imp_str = 'import numpy\n\n'

        return imp_str

    def _input_argument_str(self):
        return ', '.join(self.input_args)

    def _return_statement(self):
        return self._four_spaces() + 'return ' + ', '.join([out_args for out_args in self.output_args])

    def _array_to_local_var(self, var_name):
        local_var_str = ''

        for i in range(7):
            local_var_str += self._four_spaces() + f'{var_name}{i+1} = {var_name}[{i}]\n'

        return local_var_str

    def _output_statement(self):
        output_expression = self.expr[1]

        output_str = ''

        for oe, oa in zip(output_expression, self.output_args):
            output_str += self._four_spaces() + oa + ' = ' + self.code_printer().doprint(oe) + '\n'

        return output_str

    def _auto_x_to_code(self):
        auto_x = self.expr[0]

        auto_x_str = ''

        for ax in auto_x:
            auto_x_str += self._four_spaces() + self.code_printer().doprint(ax[0]) + ' = ' + self.code_printer().doprint(ax[1]) + '\n'

        return auto_x_str



class CppFunction(Function):
    def __init__(self, name, path, input_args, output_arg, aux_output_args, expr):
        super().__init__(name, path, input_args, output_arg, expr)
        self.code_printer = CXX11CodePrinter
        self.aux_output_args = aux_output_args

    def _array_to_local_var(self, var_name):
        local_var_str = ''

        for i in range(7):
            local_var_str += f'{self._tab_space()}const double {var_name}{i+1} = {var_name}[{i}];\n'

        return local_var_str

    def _output_statement(self):
        output_expression = self.expr[1]

        output_str = ''

        for oe, oa in zip(output_expression, self.output_args):
            output_str += self._tab_space() + oa + ' << '

            output_str +=  ', '.join([self.code_printer().doprint(e) for e in oe]) + ';\n'

        return output_str

    def _aux_output_statement(self):
        output_expression = self.expr[1]

        output_str = ''

        for oe, oa in zip(output_expression, self.aux_output_args):
            output_str += self._tab_space() + oa + ' << '

            output_str +=  ', '.join([self.code_printer().doprint(e) for e in oe]) + ';\n'

        return output_str

    def _auto_x_to_code(self):
        auto_x = self.expr[0]

        auto_x_str = ''

        for ax in auto_x:
            auto_x_str += self._tab_space() + 'const double ' + self.code_printer().doprint(ax[0]) + ' = ' + self.code_printer().doprint(ax[1]) + ';\n'

        return auto_x_str

    def print(self):
        self.open_file('cpp')
        print(f'Printing the function: {self.function_name}')
        print(f'{self._header_str()}', file=self.file_handle)
        print(f'{self.output_args[list(self.output_args.keys())[0]]} {self.function_name}({self._input_argument_str()})', file=self.file_handle)
        print('{\n', file=self.file_handle)
        for out_arg in self.aux_output_args:
            print(self._tab_space() + self.aux_output_args[out_arg] + ' ' + out_arg + ';', file=self.file_handle)
        print(f'{self._tab_space()}{self.output_args[list(self.output_args.keys())[0]]} {list(self.output_args.keys())[0]};', file=self.file_handle)
        print('\n', file=self.file_handle)
        for input_arg in self.input_args:
            print(self._array_to_local_var(input_arg), file=self.file_handle)
        print(self._auto_x_to_code(), file=self.file_handle)
        print(self._aux_output_statement(), file=self.file_handle)
        print(self._output_statement(), file=self.file_handle)
        print(self._return_statement(), file=self.file_handle)
        print('}\n', file=self.file_handle)
        self.close_file()

    def _header_str(self):
        header_str = '\\include "Eigen/Dense"\n' + '\\include <cmath>\n'

        return header_str

    def _input_argument_str(self):
        return ', '.join([f'{self.input_args[s]} {s}' for s in self.input_args])

    def _return_statement(self):
        return f'{self._tab_space()}return {list(self.output_args.keys())[0]};'

class JuliaFunction(Function):
    def __init__(self, name, path, input_args, output_arg, expr):
        super().__init__(name, path, input_args, output_arg, expr)
        self.code_printer = JuliaCodePrinter

    def print(self):
        self.open_file('jl')
        print(f'Printing the function: {self.function_name}')
        print(f'{self._import_str()}', file=self.file_handle)
        print(f'function {self.function_name}({self._input_argument_str()})', file=self.file_handle)
        for input_arg in self.input_args:
            print(self._array_to_local_var(input_arg), file=self.file_handle)
        print(self._auto_x_to_code(), file=self.file_handle)
        print(self._output_statement(), file=self.file_handle)
        print(self._return_statement(), file=self.file_handle)
        print('end', file=self.file_handle)
        self.close_file()

    def _import_str(self):
        imp_str = 'using StaticArrays\n\n'

        return imp_str

    def _input_argument_str(self):
        return ', '.join([f'{s}::{self.input_args[s]}' for s in self.input_args])

    def _array_to_local_var(self, var_name):
        local_var_str = ''

        for i in range(7):
            local_var_str += self._four_spaces() + f'{var_name}{i+1} = {var_name}[{i+1}]\n'

        return local_var_str

    def _auto_x_to_code(self):
        auto_x = self.expr[0]

        auto_x_str = ''

        for ax in auto_x:
            auto_x_str += self._four_spaces() + self.code_printer().doprint(ax[0]) + ' = ' + self.code_printer().doprint(ax[1]) + '\n'

        return auto_x_str

    def _output_statement(self):
        output_expression = self.expr[1]

        output_str = ''

        # for oe, oa in zip(output_expression, self.output_args):
        #     output_str += self._four_spaces() + oa + ' = ' + self.code_printer().doprint(oe) + '\n'

        # TODO the first index for matrix printing is not working, temporary solution is to set the index to 1 for all elements
        # TODO the second index for matrix printing is not working, temporary solution is to use an index starting from 1 and increment
        i = 1
        for oe, oa in zip(output_expression, self.output_args):
            # output_str += self._four_spaces() + oa + ' = ' + self.code_printer().doprint(oe) + '\n'

            j = 1

            # Check if the expression is (mathematically) a vector or a matrix
            if oe.shape[1] == 1:
                for oei in oe:
                    output_str += self._four_spaces() + f'{oa}[{j}] = ' + self.code_printer().doprint(oei) + '\n'
                    j += 1
            else:
                for oei in oe:
                    output_str += self._four_spaces() + f'{oa}[{i}, {j}] = ' + self.code_printer().doprint(oei) + '\n'
                    j += 1


        return output_str

    def _return_statement(self):
        return_statement = ', '.join([out_args for out_args in self.output_args])
        return f'{self._four_spaces()}return ({return_statement})'

if __name__ == '__main__':
    # import os
    # from pathlib import Path
    from sympy import Matrix, pi, symbols
    from sympy.physics.mechanics import dynamicsymbols, mechanics_printing, Point, ReferenceFrame, \
        RigidBody, inertia, KanesMethod
    # from function import PythonFunction
    # from function import CppFunction
    # from function import JuliaFunction

    mechanics_printing(pretty_print=True)

    q1, q2, q3, q4, q5, q6, q7 = dynamicsymbols("q1 q2 q3 q4 q5 q6 q7")
    q1p, q2p, q3p, q4p, q5p, q6p, q7p = dynamicsymbols("q1 q2 q3 q4 q5 q6 q7", 1)
    u1, u2, u3, u4, u5, u6, u7 = dynamicsymbols("u1 u2 u3 u4 u5 u6 u7")
    u1p, u2p, u3p, u4p, u5p, u6p, u7p = dynamicsymbols("u1 u2 u3 u4 u5 u6 u7", 1)

    # Lists of generalized coordinates and speeds
    q = [q1, q2, q3, q4, q5, q6, q7]
    qp = [q1p, q2p, q3p, q4p, q5p, q6p, q7p]
    u = [u1, u2, u3, u4, u5, u6, u7]

    dummy_dict = dict(zip(
                          q + qp + u,
                          ["q1", "q2", "q3", "q4", "q5", "q6", "q7"] +
                          ["qp1", "qp2", "qp3", "qp4", "qp5", "qp6", "qp7"] +
                          ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
                         )
                     )

    N = ReferenceFrame("N")
    A = N.orientnew("A", "Body", [pi, 0, q1], "123")
    B = A.orientnew("B", "Body", [pi / 2, 0, q2], "123")
    C = B.orientnew("C", "Body", [-pi / 2, 0, q3], "123")
    D = C.orientnew("D", "Body", [pi / 2, 0, q4], "123")
    E = D.orientnew("E", "Body", [-pi / 2, 0, q5], "123")
    F = E.orientnew("F", "Body", [pi / 2, 0, q6], "123")
    G = F.orientnew("G", "Body", [-pi / 2, 0, q7], "123")
    H = G.orientnew("H", "Body", [pi, 0, 0], "123")

    P0 = Point("O")
    P1 = P0.locatenew("P1", 0.15643 * N.z)
    P2 = P1.locatenew("P2", 0.005375 * A.y - 0.12838 * A.z)
    P3 = P2.locatenew("P3", -0.21038 * B.y - 0.006375 * B.z)
    P4 = P3.locatenew("P4", 0.006375 * C.y - 0.21038 * C.z)
    P5 = P4.locatenew("P5", -0.20843 * D.y - 0.006375 * D.z)
    P6 = P5.locatenew("P6", 0.00017505 * E.y - 0.10593 * E.z)
    P7 = P6.locatenew("P7", -0.10593 * F.y - 0.00017505 * F.z)
    P8 = P7.locatenew("P8", -0.0615 * G.z)

    x = P8.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)

    R = N.dcm(H).subs(dummy_dict)

    forward_kinematics_printer_julia = JuliaFunction('forward_kinematics', '.', {'q':'SVector{7, Float64}'}, {'x': 'SVector{3, Float64}', 'R': 'SMatrix{3, 3, Float64}'}, [x, R])
    forward_kinematics_printer_julia.print()

