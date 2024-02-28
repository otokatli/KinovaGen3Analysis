import atexit
import os
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.cxx import CXX11CodePrinter


class Function:
    def __init__(self, name, path, input_args, output_args, expr):
        self.function_name = name
        self.file_path = path
        self.input_args = input_args
        self.output_args = output_args
        self.expr = expr
        self.code_printer = None

        self.file_handle = open(os.path.join(self.file_path, self.function_name + '.py'), 'w')

        # Close the output file at exit
        atexit.register(self.cleanup)

        # Open the file for writing
    def cleanup(self):
        print('Closing the file')
        self.file_handle.close()

    def _four_spaces(self):
        return '    '

    def _q_array_to_local_var(self, is_cpp=False):
        local_var_str = ''

        if is_cpp:
            for i in range(7):
                local_var_str += f'{self._four_spaces()}const double q{i+1} = q[{i}]\n'
        else:
            for i in range(7):
                local_var_str += f'{self._four_spaces()}q{i+1} = q[{i}]\n'

        return local_var_str

    def _newline(self):
        return '\n'

    def _auto_x_to_code(self, is_cpp=False):
        auto_x = self.expr[0]

        auto_x_str = ''

        if is_cpp:
            pass
        else:
            for ax in auto_x:
                # auto_x_str += self.code_printer().doprint(ax[0]) + ' = ' + self.code_printer().doprint(ax[1]) + '\n'
                auto_x_str += self.code_printer().doprint(ax[0])# + ' = ' + self.code_printer().doprint(ax[1]) + '\n'

        return auto_x_str

class PythonFunction(Function):
    def __init__(self, name, path, input_args, output_args, expr):
        super().__init__(name, path, input_args, output_args, expr)
        self.code_printer = NumPyPrinter

    def print(self):
        print(f'{self._import_str()}', file=self.file_handle)
        print(f'def {self.function_name} ({self._input_argument_str()}):', file=self.file_handle)
        print(self._q_array_to_local_var(), file=self.file_handle)
        print(self._auto_x_to_code(), file=self.file_handle)

    def _import_str(self):
        imp_str = 'import numpy\n\n'

        return imp_str

    def _input_argument_str(self):
        arg_str = ''
        for key in self.input_args:
            arg_str += f'{key}: {self.input_args[key]}'

        return arg_str


class CppFunction(Function):
    def print(self):
        print(f'{self._header_str()}', file=self.file_handle)
        print(f'def {self.function_name} ({self._input_argument_str()}):', file=self.file_handle)
        print(f'{self._q_array_to_local_var(is_cpp=True)}', file=self.file_handle)

    def _header_str(self):
        header_str = '\include "Eigen/Dense"\n'

        return header_str


if __name__ == '__main__':
    function_name = 'forward_kinematics'
    file_path = os.path.join('.')
    print(file_path)
    py_fun_input_args = {'q': 'numpy.ndarray'}
    py_fun_output_args = {'x': 'numpy.ndarray',
                          'R': 'numpy.ndarray'}
    py_fun = PythonFunction(function_name, file_path, py_fun_input_args, py_fun_output_args, None)

    py_fun.print()

    # cpp_fun_input_args = {'q': 'Eigen::Vector<double, 7>'}
    # cpp_fun_output_args = {'T': 'Eigen::Affine3d<double, 7>'}
    # cpp_fun = CppFunction(function_name, cpp_fun_input_args, cpp_fun_output_args)

    # cpp_fun.print()
