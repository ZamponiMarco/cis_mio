import time
from queue import Empty, Queue
from typing import Optional, List

from pathos.threading import ThreadPool
from system.solvable import Solvable
from tree.node import TreeNode, VerificationType
from util.data_util import class_to_label
from z3 import Real, Context, Solver, z3


def _check(solver: tuple[Context, Solver, list[Real]], queue: Queue):
    val = solver[1].check()
    queue.put((solver[1], solver[2], val))
    return

def check(node: TreeNode, problem: Solvable, verbose: bool = False) -> Optional[List]:
    if not node.verified == VerificationType.UNVERIFIED:
        return None

    pool = ThreadPool()
    output_size = problem.get_output_size()

    fixed_delta = class_to_label(node.value, output_size)

    if verbose:
        print(f'Verifying: Node ({node.height}) ', fixed_delta, end=' ')
    solvers: list[tuple[Context, Solver, list[Real]]] = \
        problem.get_solver(node.polyhedron, fixed_delta)

    queue = Queue()

    if verbose:
        print(f"Time: ", end='')
    start = time.time()
    for solver in solvers:
        pool.apipe(_check, solver, queue)

    timeout = 5
    try:
        while True:
            timeout_start = time.time()
            solution = queue.get(block=True, timeout=timeout)

            if solution[2] == z3.unknown:
                timeout -= time.time() - timeout_start
            else:
                end = time.time()
                if verbose:
                    print(end - start, end=' ')
                break
    except Empty:
        if verbose:
            print('Timeout')
        node.verified = VerificationType.UNVERIFIED
        for solver in solvers:
            solver[0].interrupt()
        return None

    to_return = None
    solver = solution[0]
    theta_symbols = solution[1]
    val = solution[2]

    if verbose:
        print("Solution:", val, end='')
    if val == z3.unsat:
        node.verified = VerificationType.VERIFIED
    elif val == z3.sat:
        node.verified = VerificationType.UNVERIFIED
        to_return = [solver.model().eval(theta_symbols[i])
                     for i in range(node.polyhedron.dim)]
        if verbose:
            print(' Cex:', to_return, end='')
    elif val == z3.unknown:
        node.verified = VerificationType.UNVERIFIED

    if verbose:
        print()
    for solver in solvers:  # Interrupt Solvers
        solver[0].interrupt()
    return to_return