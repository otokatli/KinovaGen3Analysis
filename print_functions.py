import os
import print_utilities as pu


def _include(file: str) -> None:
    print('#include <cmath>', file=file)
    print('#include <Eigen/Dense>', file=file)
    print('', file=file)

def _q(file: str) -> None:
    for i in range(7):
        print(f'\tconst double q{i+1} = q({i});', file=file)
    print('', file=file)

def forward_kinematics(output_folder: str, expr_list: list) -> None:
    with open(os.path.join(output_folder, 'forwardKinematics.cpp'), "w") as f:
        # Print the preamble
        _include(f)
        
        print('Eigen::Transform<double, 3, Eigen::Affine> forwardKinematics(const Eigen::Vector<double, 7>& q)', file=f)
        print('{', file=f)
        print('\tEigen::Vector<double, 3> x;', file=f)
        print('\tEigen::Matrix<double, 3, 3> R;', file=f)
        print('\tEigen::Transform<double, 3, Eigen::Affine> transform = Eigen::Transform<double, 3, Eigen::Affine>::Identity();', file=f)
        print('', file=f)

        # Print the joint variables
        _q(f)

        for i in expr_list[0]:
            print(f'\tconst double {str(i[0])} = {pu.add_std(str(i[1]))};', file=f)
        print('', file=f)

        print(pu.matrix_to_eigen_stream(expr_list[1][0], 'x'), file=f)
        print(pu.matrix_to_eigen_stream(expr_list[1][1], 'R'), file=f)
        print('', file=f)
        
        print('\ttransform.translate(x);', file=f)
        print('\ttransform.rotate(R);', file=f)
        print('', file=f)
        
        print('\treturn transform;', file=f)
        print('}', file=f)

def jacobian(output_folder: str, expr_list: list) -> None:
    with open(os.path.join(output_folder, 'jacobian.cpp'), "w") as f:
        # Print the preamble
        _include(f)

        print('Eigen::Matrix<double, 6, 7> jacobian(const Eigen::Vector<double, 7>& q)', file=f)
        print('{', file=f)
        print('\tEigen::Matrix<double, 6, 7> J;', file=f)
        print('', file=f)

        # Print the joint variables
        _q(f)

        for i in expr_list[0]:
            print(f'\tconst double {str(i[0])} = {pu.add_std(str(i[1]))};', file=f)
        print('', file=f)

        print(pu.matrix_to_eigen_stream(expr_list[1][0], 'J'), file=f)
        print('', file=f)
        
        print('\treturn J;', file=f)
        print('}', file=f)

def mass_matrix(output_folder: str, expr_list: list) -> None:
    with open(os.path.join(output_folder, 'massMatrix.cpp'), "w") as f:
        # Print the preamble
        _include(f)

        print('Eigen::Matrix<double, 7, 7> massMatrix(const Eigen::Vector<double, 7>& q)', file=f)
        print('{', file=f)
        print('\tEigen::Matrix<double, 7, 7> M;', file=f)
        print('', file=f)

        # Print the joint variables
        _q(f)

        for i in expr_list[0]:
            print(f'\tconst double {str(i[0])} = {pu.add_std(str(i[1]))};', file=f)
        print('', file=f)

        print(pu.matrix_to_eigen_stream(expr_list[1][0], 'M'), file=f)
        print('', file=f)
        
        print('\treturn M;', file=f)
        print('}', file=f)

def coriolis(output_folder: str, expr_list: list) -> None:
    with open(os.path.join(output_folder, 'coriolis.cpp'), "w") as f:
        # Print the preamble
        _include(f)

        print('Eigen::Vector<double, 7> coriolis(const Eigen::Vector<double, 7>& q, const Eigen::Vector<double, 7>& qp)', file=f)
        print('{', file=f)
        print('\tEigen::Vector<double, 7> C;', file=f)
        print('', file=f)

        # Print the joint variables
        _q(f)

        for i in expr_list[0]:
            print(f'\tconst double {str(i[0])} = {pu.add_std(str(i[1]))};', file=f)
        print('', file=f)

        print(pu.matrix_to_eigen_stream(expr_list[1][0], 'C'), file=f)
        print('', file=f)
        
        print('\treturn C;', file=f)
        print('}', file=f)

def gravity(output_folder: str, expr_list: list) -> None:
    with open(os.path.join(output_folder, 'gravity.cpp'), "w") as f:
        # Print the preamble
        _include(f)

        print('Eigen::Vector<double, 7> gravity(const Eigen::Vector<double, 7>& q)', file=f)
        print('{', file=f)
        print('\tEigen::Vector<double, 7> G;', file=f)
        print('', file=f)

        print('\t// Gravitational acceleration constant', file=f)
        print('\tconst double g{ 9.80665 };', file=f)
        print('', file=f)

        print('// Unit vector in the gravity direction', file=f)
        print('\tconst double xg{ 1.0 };', file=f)
        print('\tconst double yg{ 0.0 };', file=f)
        print('\tconst double zg{ 0.0 };', file=f)
        print('', file=f)

        # Print the joint variables
        _q(f)

        for i in expr_list[0]:
            print(f'\tconst double {str(i[0])} = {pu.add_std(str(i[1]))};', file=f)
        print('', file=f)

        print(pu.matrix_to_eigen_stream(expr_list[1][0], 'G'), file=f)
        print('', file=f)
        
        print('\treturn G;', file=f)
        print('}', file=f)