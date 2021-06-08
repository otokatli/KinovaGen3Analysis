import os
from pathlib import Path
from sympy import cse, Matrix, pi, symbols
from sympy.physics.mechanics import dynamicsymbols, Point, ReferenceFrame, \
    RigidBody, inertia, KanesMethod
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.cxx import CXX17CodePrinter


code_printer = NumPyPrinter()
# code_printer = CXX17CodePrinter()

# Whether to include the gripper into the calculation or not
gripper = False

q1, q2, q3, q4, q5, q6, q7 = dynamicsymbols("q1 q2 q3 q4 q5 q6 q7")
q1p, q2p, q3p, q4p, q5p, q6p, q7p = dynamicsymbols("q1 q2 q3 q4 q5 q6 q7", 1)
u1, u2, u3, u4, u5, u6, u7 = dynamicsymbols("u1 u2 u3 u4 u5 u6 u7")
u1p, u2p, u3p, u4p, u5p, u6p, u7p = dynamicsymbols("u1 u2 u3 u4 u5 u6 u7", 1)

# Lists of generalized coordinates and speeds
q = [q1, q2, q3, q4, q5, q6, q7]
u = [u1, u2, u3, u4, u5, u6, u7]

# Torques associated with each joint
TA, TB, TC, TD, TE, TF, TG = symbols("TA TB TC TD TE TF TG")
Fx, Fy, Fz = symbols('Fx Fy Fz')

# Gravitational constant
g = symbols('g')

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

# Mid-point of each link
Ao_half = P1.locatenew("P1_half", 0.0054 / 2 * A.y - 0.1284 / 2 * A.z)
Bo_half = P2.locatenew("P2_half", -0.2014 / 2 * B.y - 0.0064 / 2 * B.z)
Co_half = P3.locatenew("P3_half", 0.0064 / 2 * C.y - 0.2104 / 2 * C.z)
Do_half = P4.locatenew("P4_half", -0.2084 / 2 * D.y - 0.0064 / 2 * D.z)
Eo_half = P5.locatenew("P5_half", -0.1059 / 2 * E.z)
Fo_half = P6.locatenew("P6_half", -0.1059 / 2 * F.y)
Go_half = P7.locatenew("P7_half", -0.0615 / 2 * G.z)

# End-effector position  (tcp)
if gripper:
    P9 = P8.locatenew("P9", 0.12 * H.z)

# Mass centers
Ao = P1.locatenew("Ao", -0.000023 * A.x - 0.010364 * A.y - 0.073360 * A.z)
Bo = P2.locatenew("Bo", -0.000044 * B.x - 0.099580 * B.y - 0.013278 * B.z)
Co = P3.locatenew("Co", -0.000044 * C.x - 0.006641 * C.y - 0.117892 * C.z)
Do = P4.locatenew("Do", -0.000018 * D.x - 0.075478 * D.y - 0.015006 * D.z)
Eo = P5.locatenew("Eo", 0.000001 * E.x - 0.009432 * E.y - 0.063883 * E.z)
Fo = P6.locatenew("Fo", 0.000001 * F.x - 0.045483 * F.y - 0.009650 * F.z)
Go = P7.locatenew("Go", -0.000281 * G.x - 0.011402 * G.y - 0.029798 * G.z)

# Center of mass for Robotiq 2F-85 gripper
if gripper:
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

# Velocity of the gripper
if gripper:
    P9.v2pt_theory(P8, N, H)

Ao.v2pt_theory(P1, N, A)
Bo.v2pt_theory(P2, N, B)
Co.v2pt_theory(P3, N, C)
Do.v2pt_theory(P4, N, D)
Eo.v2pt_theory(P5, N, E)
Fo.v2pt_theory(P6, N, F)
Go.v2pt_theory(P7, N, G)

# Velocity of the com of the gripper
if gripper:
    Ho.v2pt_theory(P8, N, H)

kd = [q1p - u1, q2p - u2, q3p - u3, q4p - u4, q5p - u5, q6p - u6, q7p - u7]

# Dummy variables for substitution
dummy_dict = dict(zip(q + u, ["q1", "q2", "q3", "q4", "q5", "q6", "q7",
                              "u1", "u2", "u3", "u4", "u5", "u6", "u7"]))

# End-effector position
x = P9.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N) if gripper \
    else P8.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)

# Link centre of mass positions
x_com_1 = Ao.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_com_2 = Bo.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_com_3 = Co.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_com_4 = Do.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_com_5 = Eo.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_com_6 = Fo.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_com_7 = Go.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)

x_com = Matrix([[x_com_1[0], x_com_1[1], x_com_1[2]],
                [x_com_2[0], x_com_2[1], x_com_2[2]],
                [x_com_3[0], x_com_3[1], x_com_3[2]],
                [x_com_4[0], x_com_4[1], x_com_4[2]],
                [x_com_5[0], x_com_5[1], x_com_5[2]],
                [x_com_6[0], x_com_6[1], x_com_6[2]],
                [x_com_7[0], x_com_7[1], x_com_7[2]]])

# Mid-point position of each link
x_mid_1 = Ao_half.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_mid_2 = Bo_half.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_mid_3 = Co_half.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_mid_4 = Do_half.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_mid_5 = Eo_half.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_mid_6 = Fo_half.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
x_mid_7 = Go_half.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)

x_mid = Matrix([[x_mid_1[0], x_mid_1[1], x_mid_1[2]],
                [x_mid_2[0], x_mid_2[1], x_mid_2[2]],
                [x_mid_3[0], x_mid_3[1], x_mid_3[2]],
                [x_mid_4[0], x_mid_4[1], x_mid_4[2]],
                [x_mid_5[0], x_mid_5[1], x_mid_5[2]],
                [x_mid_6[0], x_mid_6[1], x_mid_6[2]],
                [x_mid_7[0], x_mid_7[1], x_mid_7[2]]])

# Position of each point on the links
joint_pos_1 = P1.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
joint_pos_2 = P2.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
joint_pos_3 = P3.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
joint_pos_4 = P4.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
joint_pos_5 = P5.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
joint_pos_6 = P6.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)
joint_pos_7 = P7.pos_from(P0).express(N).subs(dummy_dict).to_matrix(N)

joint_pos = Matrix([[joint_pos_1[0], joint_pos_1[1], joint_pos_1[2]],
                    [joint_pos_2[0], joint_pos_2[1], joint_pos_2[2]],
                    [joint_pos_3[0], joint_pos_3[1], joint_pos_3[2]],
                    [joint_pos_4[0], joint_pos_4[1], joint_pos_4[2]],
                    [joint_pos_5[0], joint_pos_5[1], joint_pos_5[2]],
                    [joint_pos_6[0], joint_pos_6[1], joint_pos_6[2]],
                    [joint_pos_7[0], joint_pos_7[1], joint_pos_7[2]]])

# End-effector orientation
R = N.dcm(H).subs(dummy_dict)

# Write Jacobian to file
# Translational part of the Jacobian
Jt = Matrix([P9.pos_from(P0).dot(N.x),
             P9.pos_from(P0).dot(N.y),
             P9.pos_from(P0).dot(N.z)])\
    .jacobian(Matrix([q1, q2, q3, q4, q5, q6, q7])).subs(dummy_dict) if gripper \
    else Matrix([P8.pos_from(P0).dot(N.x),
                 P8.pos_from(P0).dot(N.y),
                 P8.pos_from(P0).dot(N.z)])\
    .jacobian(Matrix([q1, q2, q3, q4, q5, q6, q7])).subs(dummy_dict)

# Rotational part of the Jacobian
Jr = Matrix(
    [
        [
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q1p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q2p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q3p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q4p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q5p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q6p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[0].coeff(q7p).subs(dummy_dict).trigsimp(),
        ],
        [
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q1p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q2p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q3p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q4p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q5p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q6p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[1].coeff(q7p).subs(dummy_dict).trigsimp(),
        ],
        [
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q1p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q2p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q3p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q4p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q5p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q6p).subs(dummy_dict).trigsimp(),
            H.ang_vel_in(N).to_matrix(N)[2].coeff(q7p).subs(dummy_dict).trigsimp(),
        ],
    ]
)

# Complete Jacobian of the robot
J = Jt.col_join(Jr)

# Dynamic analysis

# Link mass [kg]
mass_A = 1.3770
mass_B = 1.1636
mass_C = 1.1636
mass_D = 0.9300
mass_E = 0.6780
mass_F = 0.6780
mass_G = 0.5000

# Mass of the gripper [kg]
if gripper:
    mass_H = 0.925

inertia_A = inertia(A, 0.004570, 0.004831, 0.001409, 0.000001, 0.000448,
                    0.000002)
inertia_B = inertia(B, 0.011088, 0.001072, 0.011255, 0.000005, -0.000691,
                    0.000000)
inertia_C = inertia(C, 0.010932, 0.011127, 0.001043, 0.000000, 0.000606,
                    -0.000007)
inertia_D = inertia(D, 0.008147, 0.000631, 0.008316, -0.000001, -0.000500,
                    0.000000)
inertia_E = inertia(E, 0.001596, 0.001607, 0.000399, 0.000000, 0.000256,
                    0.000000)
inertia_F = inertia(F, 0.001641, 0.000410, 0.001641, 0.000000, -0.000278,
                    0.000000)
inertia_G = inertia(G, 0.000587, 0.000369, 0.000609, 0.000003, 0.000118,
                    0.000003)

# Inertia of the gripper
if gripper:
    inertia_H = inertia(H, 4180e-6, 5080e-6, 1250e-6)

body_A = RigidBody('body_A', Ao, A, mass_A, (inertia_A, Ao))
body_B = RigidBody('body_B', Bo, B, mass_B, (inertia_B, Bo))
body_C = RigidBody('body_C', Co, C, mass_C, (inertia_C, Co))
body_D = RigidBody('body_D', Do, D, mass_D, (inertia_D, Do))
body_E = RigidBody('body_E', Eo, E, mass_E, (inertia_E, Eo))
body_F = RigidBody('body_F', Fo, F, mass_F, (inertia_F, Fo))
body_G = RigidBody('body_G', Go, G, mass_G, (inertia_G, Go))

if gripper:
    body_H = RigidBody('body_H', Ho, H, mass_H, (inertia_H, Ho))

body_list = [body_A, body_B, body_C, body_D, body_E, body_F, body_G, body_H] \
    if gripper else [body_A, body_B, body_C, body_D, body_E, body_F, body_G]

force_list = [(Ao, N.z * g * mass_A),
              (Bo, N.z * g * mass_B),
              (Co, N.z * g * mass_C),
              (Do, N.z * g * mass_D),
              (Eo, N.z * g * mass_E),
              (Fo, N.z * g * mass_F),
              (Go, N.z * g * mass_G),
              (Ho, N.z * g * mass_H),
              (A, TA * A.z),
              (B, TB * B.z),
              (C, TC * C.z),
              (D, TD * D.z),
              (E, TE * E.z),
              (F, TF * F.z),
              (G, TG * G.z),
              (P9, Fx * N.x + Fy * N.y + Fz * N.z)] if gripper \
    else [(Ao, N.z * g * mass_A),
          (Bo, N.z * g * mass_B),
          (Co, N.z * g * mass_C),
          (Do, N.z * g * mass_D),
          (Eo, N.z * g * mass_E),
          (Fo, N.z * g * mass_F),
          (Go, N.z * g * mass_G),
          (A, TA * A.z),
          (B, TB * B.z),
          (C, TC * C.z),
          (D, TD * D.z),
          (E, TE * E.z),
          (F, TF * F.z),
          (G, TG * G.z),
          (P8, Fx * N.x + Fy * N.y + Fz * N.z)]

KM = KanesMethod(N, q_ind=q, u_ind=u, kd_eqs=kd)

(fr, frstar) = KM.kanes_equations(body_list, force_list)

# Mass matrix
M = KM.mass_matrix

# Rest of the eom
forcing = KM.forcing

# Gravity term
G = Matrix([-forcing[0].coeff(g) * g, -forcing[1].coeff(g) * g,
            -forcing[2].coeff(g) * g, -forcing[3].coeff(g) * g,
            -forcing[4].coeff(g) * g, -forcing[5].coeff(g) * g,
            -forcing[6].coeff(g) * g])

C = -(forcing + G).subs({TA: 0, TB: 0, TC: 0, TD: 0, TE: 0, TF: 0, TG: 0, g: 0,
                         Fx: 0, Fy: 0, Fz: 0})

# Auto-z long expressions
pose_compact = cse([x, R])
com_position_compact = cse(x_com)
mid_position_compact = cse(x_mid)
joint_pos_compact = cse(joint_pos)
J_compact = cse(J)
M_compact = cse(M.subs(dummy_dict))
G_compact = cse(G.subs(dummy_dict))
C_compact = cse(C.subs(dummy_dict))

# Write to files (python, cpp)
# Create output folder
src_folder = os.path.join('.', 'src')
Path(src_folder).mkdir(exist_ok=True)

# Write end-effector position to file
with open(os.path.join(src_folder, 'forward_kinematics.py'), "w") as f:
    for i in pose_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(code_printer.doprint(pose_compact[1][0]), file=f)
    print(code_printer.doprint(pose_compact[1][1]), file=f)

# Write link com position to file
with open(os.path.join(src_folder, 'com_positions.py'), "w") as f:
    for i in com_position_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(code_printer.doprint(com_position_compact[1][0]), file=f)

# Write link mid-point position to file
with open(os.path.join(src_folder, 'mid_positions.py'), "w") as f:
    for i in mid_position_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(code_printer.doprint(mid_position_compact[1][0]), file=f)

# Write joint positions to file
with open(os.path.join(src_folder, 'joint_positions.py'), "w") as f:
    for i in joint_pos_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(code_printer.doprint(joint_pos_compact[1][0]), file=f)

# Write Jacobian matrix to file
with open(os.path.join(src_folder, 'jacobian.py'), "w") as f:
    for i in J_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(code_printer.doprint(J_compact[1][0]), file=f)

# Write mass matrix to file
with open(os.path.join(src_folder, 'mass_matrix.py'), "w") as f:
    for i in M_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(code_printer.doprint(M_compact[1][0]), file=f)

# Write gravity term to file
with open(os.path.join(src_folder, 'gravity.py'), "w") as f:
    for i in G_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(code_printer.doprint(G_compact[1][0]), file=f)

# Write Coriolis term to file
with open(os.path.join(src_folder, 'coriolis.py'), "w") as f:
    for i in C_compact[0]:
        print(str(i[0]) + " = " + str(i[1]), file=f)
    print(code_printer.doprint(C_compact[1][0]), file=f)

