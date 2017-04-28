N Queens Problem:
File eightQueens.assigment and eightQueens.csp include domains, constraints for eight queens problem.
File eightQueens.test and eightQueens.solution are designed for eight queens problem.

Constraints:
1. Vertically, if they are in the same column---this is impossible as Qi and Qj are placed in
different columns.
2. Horizontally, if they are in the same row---we need the constraint Qi ¡Ù Qj.
3. Along a diagonal---they cannot be the same number of columns apart as they are rows apart:
we need the constraint |i-j| ¡Ù |Qi-Qj| (|.| is absolute value)

Directly run "python autograder.py -t Extra\eightQueens.test"
