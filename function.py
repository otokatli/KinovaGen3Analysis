import atexit
import os
from sympy import cse
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.cxx import CXX11CodePrinter


class Function:
    def __init__(self, name, path, input_args, output_args, expr):
        self.function_name = name
        self.file_path = path
        self.input_args = input_args
        self.output_args = output_args
        self.expr = cse(expr)
        self.code_printer = None

        self.file_handle = None

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

    def open_file(self):
        self.file_handle = open(os.path.join(self.file_path, self.function_name + '.py'), 'w')


    def print(self):
        self.open_file()
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

    def open_file(self):
        self.file_handle = open(os.path.join(self.file_path, self.function_name + '.cpp'), 'w')

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
        self.open_file()
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
