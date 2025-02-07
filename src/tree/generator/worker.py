import os
import time
import traceback

import numpy as np
import psutil

import util.tree_util
from set.polyhedron import from_interval
from system.solvable import Solvable
from tree.generator.logger import setup_logger
from tree.node import TreeNode, VerificationType
from util.data_util import labels_to_classes
from util.optimization_util import sequential_solve
from util.z3_util import check


def process_node(node: TreeNode, model: Solvable, seed,
                 max_height: int, refinement_points: int=40, splitter=None, sampler=None, logger=None):
    """
    Main node refinement process.

    Args:
        node (TreeNode): the node being checked and eventually refined.
        model (Solvable): the optimization problem.
        seed (int): the random seed.
        max_height (int): the maximum height of the tree.
        refinement_points (int): the amount of refinement points to generate.
        splitter (callable): splitting hyperplane strategy.
        sampler (callable): sampling strategy from node domain.
        logger (Logger): logger object.
    """
    np.random.seed(seed)
    logger.info(f"Processing Node ({node.height}), {len(node.data[0])} Data Points")
    to_verify = None
    cex = check(node, model)
    logger.info(f'Verification executed, result: {node.verified}')
    lb, ub = node.polyhedron.bounding_box()
    if node.verified == VerificationType.UNVERIFIED and node.height < max_height \
            and not model.same_discretization(lb, ub) and len(node.data[0]) < refinement_points * 20:
        logger.info(f'Start Sampling Points From Area')
        X, y = sampler(model, int(refinement_points/2), node.polyhedron)
        logger.info(f'End Sampling Points From Area')

        if cex:
            logger.info(f'Start Sampling Points From Counterexample Neighborhood')
            cex = [el.as_fraction() for el in cex]
            out = sequential_solve(model, [cex])
            X = np.vstack([
                X,
                cex
            ])
            y = np.vstack([
                y,
                out
            ])

            bbox_length = node.polyhedron.bounding_box_lengths()
            logger.info(f'Construct Neighborhood, {bbox_length}')
            bad_zone = node.polyhedron.intersect_polyhedra(
                from_interval(
                    np.array([el - (bbox_length[i] / 16) for i, el in enumerate(cex)]),
                    np.array([el + (bbox_length[i] / 16) for i, el in enumerate(cex)])
                )
            )
            X_cex, y_cex = sampler(model, int(refinement_points/2) - 1, bad_zone)
            X = np.vstack([
                X,
                X_cex
            ])
            y = np.vstack([
                y,
                y_cex
            ])
            logger.info(f'End Sampling Points From Counterexample Neighborhood')
        else:
            logger.info('No CEX')

        # Add the new data
        node.add_data(X, labels_to_classes(y))

        logger.info(f'Start Splitting Procedure')
        if node.split_once(
                splitter=splitter,
                fallback_splitter=util.tree_util.half_split()
        ):
            to_verify = 'children'
            logger.info('Node successfully split')
        else:
            to_verify = 'self'
            logger.info('Could not split node')
        logger.info(f'End Splitting Procedure')
    elif node.verified == VerificationType.UNVERIFIED and model.same_discretization(lb, ub):
        node.verified = VerificationType.TOLERANCE_VERIFIED
    logger.info(f'Terminated Node Processing, Returning Results')
    return node, to_verify


def worker(model: Solvable, seed, max_height, refinement_points=40,
           splitter=None, sampler=None):
    """
    Entry point of a node refinement worker.

    Args:
        model (Solvable): the optimization problem.
        seed (int): the random seed.
        max_height (int): the maximum height of the tree.
        refinement_points (int): the amount of refinement points to generate.
        splitter (callable): splitting hyperplane strategy.
        sampler (callable): sampling strategy from node domain.
    """
    def work(task_queue, result_queue, utilization_queue):
        """
        Function being called at process start.

        Args:
            task_queue (Queue): the task queue, containing the node refinement tasks sent by the master.
            result_queue (Queue): the result queue, where task solutions are sent to the master.
            utilization_queue (Queue): utilization queue to send fraction of cpu time with respect to process lifetime.
        """
        logger = setup_logger()
        logger.propagate = False
        logger.info(f"Process {os.getpid()} started.")

        process = psutil.Process()
        start_time = time.time()
        start_cpu_times = process.cpu_times()

        while True:
            task = task_queue.get()
            submission = time.time()
            if task is None:
                break  # Exit if `None` is received
            node_id, node = task
            try:
                node, to_verify = process_node(node, model, seed, max_height, refinement_points,
                                               splitter, sampler, logger)
                completion = time.time()
                result_queue.put((node_id, node, to_verify, submission, completion))
            except Exception as e:
                logger.error(str(e))
                logger.error(str(traceback.format_exc()))
                break

        end_time = time.time()
        end_cpu_times = process.cpu_times()

        cpu_time_user = end_cpu_times.user - start_cpu_times.user
        cpu_time_system = end_cpu_times.system - start_cpu_times.system
        total_cpu_time = cpu_time_user + cpu_time_system

        elapsed_time = end_time - start_time

        # Put the CPU utilization info into the utilization_queue
        utilization_queue.put(total_cpu_time / elapsed_time)

    return work
