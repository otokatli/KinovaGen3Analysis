from sympy import cse, Matrix, pi, symbols
from sympy.physics.mechanics import dynamicsymbols, Point, ReferenceFrame
from sympy.printing.pycode import NumPyPrinter
from sympy.printing.cxxcode import CXX17CodePrinter


printerPy = NumPyPrinter()
printerCpp = CXX17CodePrinter()

q1, q2, q3, q4, q5, q6, q7 = dynamicsymbols("q1 q2 q3 q4 q5 q6 q7")
q1p, q2p, q3p, q4p, q5p, q6p, q7p = dynamicsymbols("q1 q2 q3 q4 q5 q6 q7", 1)
u1, u2, u3, u4, u5, u6, u7 = dynamicsymbols("u1 u2 u3 u4 u5 u6 u7")
u1p, u2p, u3p, u4p, u5p, u6p, u7p = dynamicsymbols("u1 u2 u3 u4 u5 u6 u7", 1)

# Lists of generalized coordinates and speeds
q = [q1, q2, q3, q4, q5, q6, q7]
u = [u1, u2, u3, u4, u5, u6, u7]

# Torques associated with each joint
TA, TB, TC, TD, TE, TF, TG = symbols("TA, TB, TC, TD, TE, TF, TG")

# Reference frames
N = ReferenceFrame("N")
A = N.orientnew("A", "Body", [pi, 0, q1], "123")
B = A.orientnew("B", "Body", [pi / 2, 0, q2], "123")
C = B.orientnew("C", "Body", [-pi / 2, 0, q3], "123")
D = C.orientnew("D", "Body", [pi / 2, 0, q4], "123")
E = D.orientnew("E", "Body", [-pi / 2, 0, q5], "123")
F = E.orientnew("F", "Body", [pi / 2, 0, q6], "123")
G = F.orientnew("G", "Body", [-pi / 2, 0, q7], "123")
# The interface module's reference frame
H = G.orientnew("H", "Body", [pi, 0, 0], "123")

# Unit for distance: meter
P0 = Point("O")
P1 = P0.locatenew("P1", 0.1564 * N.z)
P2 = P1.locatenew("P2", 0.0054 * A.y - 0.1284 * A.z)
P3 = P2.locatenew("P3", -0.2104 * B.y - 0.0064 * B.z)
P4 = P3.locatenew("P4", 0.0064 * C.y - 0.2104 * C.z)
P5 = P4.locatenew("P5", -0.2084 * D.y - 0.0064 * D.z)
P6 = P5.locatenew("P6", -0.1059 * E.z)
P7 = P6.locatenew("P7", -0.1059 * F.y)
P8 = P7.locatenew("P8", -0.0615 * G.z)
# End-effector position  (mid-point of the gripper)
P9 = P8.locatenew("P9", 0.12 * H.z)

# Mass centers
Ao = P1.locatenew("Ao", -0.000023 * A.x - 0.010364 * A.y - 0.07336 * A.z)
Bo = P2.locatenew("Bo", -0.000044 * B.x - 0.09958 * B.y - 0.013278 * B.z)
Co = P3.locatenew("Co", -0.000044 * C.x - 0.006641 * C.y - 0.117892 * C.z)
Do = P4.locatenew("Do", -0.000018 * D.x - 0.075478 * D.y - 0.015006 * D.z)
Eo = P5.locatenew("Eo", 0.000001 * E.x - 0.009432 * E.y - 0.063883 * E.z)
Fo = P6.locatenew("Fo", 0.000001 * F.x - 0.045483 * F.y - 0.009650 * F.z)
Go = P7.locatenew("Go", -0.000281 * G.x - 0.011402 * G.y - 0.029798 * G.z)
Ho = P8.locatenew("Ho", 0.058 * H.z)

# Velocities
P0.set_vel(N, 0 * N.x + 0 * N.y + 0 * N.z)
P1.v2pt_theory(P0, N, N)
P2.v2pt_theory(P1, N, A)
P3.v2pt_theory(P2, N, B)
P4.v2pt_theory(P3, N, C)
P5.v2pt_theory(P4, N, D)
P6.v2pt_theory(P5, N, E)
P7.v2pt_theory(P6, N, F)
P8.v2pt_theory(P7, N, G)
P9.v2pt_theory(P8, N, H)

Ao.v2pt_theory(P1, N, A)
Bo.v2pt_theory(P2, N, B)
Co.v2pt_theory(P3, N, C)
Do.v2pt_theory(P4, N, D)
Eo.v2pt_theory(P5, N, E)
Fo.v2pt_theory(P6, N, F)
Go.v2pt_theory(P7, N, G)
Ho.v2pt_theory(P8, N, H)

kd = [q1p - u1, q2p - u2, q3p - u3, q4p - u4, q5p - u5, q6p - u6, q7p - u7]

# Replace the time varying function of generalized coordinates, speed
# dummy_symbols = [Dummy() for i in q + u]
# dummy_dict = dict(zip(q + u, dummy_symbols))
dummy_dict = dict(
    zip(
        q + u,
        [
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "u1",
            "u2",
            "u3",
            "u4",
            "u5",
            "u6",
            "u7",
        ],
    )
)


# End-effector position
x = P9.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)

# End-effector orientation
R = N.dcm(H).subs(dummy_dict)

# Compact (cse processed) version of the end-effector position and orientation
pose_compact = cse([x, R])

# Write Jacobian to file
# Translational part of the Jacobian
Jt = (
    Matrix(
        [P9.pos_from(P0).dot(N.x),
         P9.pos_from(P0).dot(N.y),
         P9.pos_from(P0).dot(N.z)]
    )
    .jacobian(Matrix([q1, q2, q3, q4, q5, q6, q7]))
    .subs(dummy_dict)
)

# Rotational part of the Jacobian
# TODO: the code below is probably resulting in a wrong jacobian
Jr = Matrix(
    [
        [
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q1p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q2p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q3p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q4p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q5p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q6p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q7p).subs(dummy_dict),
        ],
        [
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q1p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q2p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q3p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q4p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q5p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q6p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q7p).subs(dummy_dict),
        ],
        [
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q1p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q2p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q3p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q4p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q5p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q6p).subs(dummy_dict),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q7p).subs(dummy_dict),
        ],
    ]
)

# Complete Jacobian of the robot
# J = Jt.col_join(Jr)
J = Jt

J_compact = cse(J)

# Write to files (python, cpp)
with open("./src/python/forwardKinematics.py", "w") as f:
    for i in pose_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(printerPy.doprint(pose_compact[1][0]), file=f)
    print(printerPy.doprint(pose_compact[1][1]), file=f)

with open("./src/cpp/forwardKinematics.cpp", "w") as f:
    for i in pose_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(printerCpp.doprint(pose_compact[1][0]), file=f)
    print(printerCpp.doprint(pose_compact[1][1]), file=f)

with open("./src/python/jacobian.py", "w") as f:
    for i in J_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(printerPy.doprint(J_compact[1][0]), file=f)

with open("./src/cpp/jacobian.cpp", "w") as f:
    for i in J_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(printerCpp.doprint(J_compact[1][0]), file=f)
