import os
import random
import time
import uuid
from argparse import Namespace, ArgumentParser

import joblib
from tqdm import tqdm

from multiprocess import Process, Queue, Manager, cpu_count  # noqa
from set.polyhedron import Polyhedron, from_interval
from system.solvable import Solvable
from tree.experiment_results import ExperimentResult
from tree.generator.worker import worker
from tree.node import VerificationType
from tree.tree import BinaryTree
from util.data_util import *
from util.sample_util import sample_points_rejection, sample_points_lhs
from util.tree_util import find_best_split_decision_tree
from z3 import z3


def parse_arguments() -> Namespace:
    """
    Parses command-line arguments.
    """
    parser = ArgumentParser(description='Process input arguments.')
    parser.add_argument('--seed', type=int, default=random.randint(0, 2000000000),
                        help='Seed value for random generation.')
    parser.add_argument('--initial_height', type=int, default=4, help='Initial tree height (default: 4).')
    parser.add_argument('--max_height', type=int, default=32, help='Maximum height for the tree.')
    parser.add_argument('--initial_points', type=int, default=100, help='Initial number of points.')
    parser.add_argument('--points', type=int, default=40, help='Number of points for each iteration.')
    parser.add_argument('--cores', type=int, default=cpu_count(), help='Number of cores to use.')
    parser.add_argument('--output', type=str, default='resources/', help='Output Folder')
    parser.add_argument('--zone', type=str, default=None, help='Verification Zone')
    return parser.parse_args()


def run_experiment(args: Namespace, solvable: Solvable, parameter_domain: Polyhedron, max_tasks: int) -> BinaryTree:
    """
    Runs the tree generation experiment.

    Args:
        args (Namespace): containing command-line arguments.
        solvable (Solvable): the optimization problem.
        parameter_domain (Polyhedron): the domain of the parameter space.
        max_tasks (int): the maximum number of tasks that can be executed during an experiment.
    """
    chosen_splitter = find_best_split_decision_tree(args.seed, solvable)
    chosen_sampler = sample_points_lhs(args.seed, solvable)

    if parameter_domain is None:
        parameter_domain = from_interval(
            solvable.get_parameter_domain_interval()[0],
            solvable.get_parameter_domain_interval()[1],
        )

    tree = get_tree_multiprocess(
        solvable, parameter_domain,
        args.seed,
        args.initial_height,
        args.max_height,
        args.initial_points,
        args.points,
        splitter=chosen_splitter,
        sampler=chosen_sampler,
        cores=args.cores,
        max_tasks=max_tasks
    )
    return tree


def save_tree(tree: BinaryTree, output_dir: str, seed: int):
    """
    Saves the generated tree to a file.

    Args:
        tree (BinaryTree): the generated tree.
        output_dir (str): the output folder.
        seed (int): the random seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(tree, f'{output_dir}/tree_{seed}.pkl')


def get_tree_multiprocess(
        model: Solvable,
        verification_domain: Polyhedron,
        seed: int,
        initial_height: int = 4,
        max_height: int = 16,
        initial_points: int = 100,
        refinement_points: int = 40,
        splitter=None,
        sampler=None,
        cores: int = 16,
        max_tasks: int = 500
):
    """
    Algorithm execution, initializes the problem, spawns worker processes, handles execution and gather metrics.

    Args:
        model (Solvable): the optimization problem.
        verification_domain (Polyhedron): the domain of the parameter space.
        seed (int): the random seed.
        initial_height (int): the tree height after initialization (CART algorithm is executed until this height).
        max_height (int): the maximum tree height, after which the algorithm will stop refining nodes.
        initial_points (int): the number of initial points to use for tree initialization.
        refinement_points (int): the number of refinement points to use for each node refinement.
        splitter (Callable): the splitting strategy to refine nodes.
        sampler (Callable): the sampling strategy for points inside unverified nodes.
        cores (int): the number of worker processes to spawn.
        max_tasks (int): the maximum number of tasks that can be executed during an experiment.
    """
    print(f'Generating Tree, seed: {seed}, max height: {max_height}')

    if not splitter:
        splitter = find_best_split_decision_tree(seed)
    if not sampler:
        sampler = sample_points_rejection(seed)

    # Set up configuration
    z3.set_param('parallel.enable', True)
    np.random.seed(seed)

    with Manager() as manager:

        start_time = time.time()

        # region TREE INITIALIZATION

        tree: BinaryTree = BinaryTree(verification_domain, model.get_output_size())

        X, y = sampler(model, initial_points, verification_domain)

        tree.root.add_data(X, labels_to_classes(y))
        tree.fit(initial_height, splitter=splitter)

        # endregion

        # region REFINEMENT PROCEDURE

        # region SETUP PROCESSES

        task_queue = manager.Queue()
        result_queue = manager.Queue()
        utilization_queue = Queue()

        num_tasks_submitted = 0
        num_results_received = 0

        submission_times = []
        completion_times = []

        id_to_node = {}

        num_workers = cores
        workers = []
        for _ in range(num_workers):
            p = Process(
                target=worker(model, seed, max_height, refinement_points, splitter, sampler),
                args=(task_queue, result_queue, utilization_queue)
            )
            workers.append(p)
            p.start()

        # endregion

        # Process Initial Leaves
        initial_nodes = tree.depth_first_leaves()
        for leaf in initial_nodes:
            node_id = str(uuid.uuid4())
            id_to_node[node_id] = leaf
            task_queue.put((node_id, leaf))
            num_tasks_submitted += 1

        # Refinement Loop

        progress_bar = tqdm(desc="Tree Generation", unit="node", ncols=100, ascii=True)

        while True:
            # Get Task Result
            node_id, leaf, to_verify, submission, completion = result_queue.get()
            num_results_received += 1
            submission_times.append(submission)
            completion_times.append(completion)

            original_node = id_to_node[node_id]
            original_node.update_from_node(leaf)

            new_tasks = []

            # Send New Tasks
            if to_verify and num_tasks_submitted < max_tasks:
                if to_verify == 'self':
                    new_tasks += [original_node]
                elif to_verify == 'children':
                    new_tasks += [original_node.left, original_node.right]

            for unverified_node in new_tasks:
                node_id = str(uuid.uuid4())
                id_to_node[node_id] = unverified_node
                task_queue.put((node_id, unverified_node))
                num_tasks_submitted += 1

            # Update progress bar
            progress_bar.total = num_tasks_submitted
            progress_bar.n = num_results_received
            progress_bar.refresh()
            progress_bar.set_postfix_str(f'Tasks Received/Submitted: {num_results_received}/{num_tasks_submitted}')

            # Check if all tasks have been processed and no more tasks to add
            if num_tasks_submitted == num_results_received:
                # Send termination signals to workers
                for _ in workers:
                    task_queue.put(None)
                break  # Exit the main loop
            if num_tasks_submitted >= max_tasks:
                break

        for p in workers:
            p.terminate()

        # endregion

        end_time = time.time()
        progress_bar.close()
        print('Tree Generation Completed.')

        # region METRICS GATHERING
        utilizations = []
        while not utilization_queue.empty():
            utilizations.append(utilization_queue.get())

        leaves = tree.depth_first_leaves()
        total_leaves = len(leaves)

        total_nodes = len(tree.depth_first_nodes())

        verified_leaves = 0
        verified_volume = 0
        data_points = 0
        for leaf in leaves:
            if leaf.verified == VerificationType.VERIFIED or leaf.verified == VerificationType.TOLERANCE_VERIFIED:
                verified_leaves += 1
                verified_volume += leaf.polyhedron.volume()
                data_points += len(leaf.data[0])

        real_max_height = 0
        for leaf in tree.depth_first_leaves():
            if leaf.height > real_max_height:
                real_max_height = leaf.height

        results: ExperimentResult = ExperimentResult(
            seed, end_time - start_time, real_max_height, sum(utilizations),
            submission_times, completion_times, verified_leaves, total_leaves, total_nodes, verified_volume,
            verification_domain, data_points
        )
        tree.experiment_results = results

        tree.remove_data()
        # endregion

    return tree
