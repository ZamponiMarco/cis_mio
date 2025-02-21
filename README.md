[# Certified Inductive Synthesis for Online Mixed-Integer Optimization

## Installation

The package can be installed manually or using Docker.

### Manual Installation

The following instructions apply to an Ubuntu OS environment. The package requires installing some system dependencies:

```
apt-get update
apt-get install libcdd-dev libgmp-dev python3-dev python3-venv git build-essential
```

Next, create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Clone]() the repository and install dependencies:

```
git clone https://github.com/ZamponiMarco/cis_mio
cd cis_mio
pip install -r requirements.txt
```

At this point, the package is ready to use.

### Docker Image

The package can also be run using a pre-built Docker image.

To build the image locally, run:

```
docker image build . -t cis_mio:iccps
docker run -it cis_mio:iccps
```

Alternatively, the pre-built Docker image can be pulled directly from Docker Hub:

```
docker run -it zamponimarco/cis_mio:iccps
```

### Note

The example optimization problems presented in this package work fine using the size-limited Gurobi license automatically included in the installation of the `gurobipy` Python package. In order to run bigger optimization problems a [Gurobi license](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license) will be needed.

## Running the Software

The easiest way to run the software is by executing the scripts inside the `script` folder. The package contains multiple case studies, each with its own corresponding scripts.

Firstly, navigate to the chosen cas study folder:

```
cd script/<case_study>
```

### Constructing Trees

To generate decision trees for a particular case study execute:

```
bash generate_tree.sh
```

This script generates multiple trees for each subdomain and stores them in `resources/<case_study>`. Each tree is saved as `tree_<seed>.pkl`.

**Important:**
Not all generated trees may be fully certified, as the process stops at a predefined maximum depth. If no certified tree is available for a subdomain, run the script again until at least one certified tree exists per subdomain.
Without at least one certified tree per subdomain, the analysis pipeline **cannot** be executed.

To reproduce the exact trees used in the paper, execute:

```
bash generate_tree_seeded.sh
```

#### **Methodology Inputs**

In detail, the scripts presented here orchestrate the generation of trees by calling internally a Python method named `run_experiment`, which accepts the following inputs:

- **`args`:** A collection of command-line arguments that define the tree generation parameters. These arguments correspond to the possible command-line flags listed in the table below.
- **`solvable`:** The problem definition.
- **`parameter_domain`:** The domain over which the tree is constructed, defining the feasible region for parameter exploration.
- **`splitter`:** A function that determines how decision nodes in the tree are split. Defaults to a decision-tree-based heuristic.
- **`sampler`:** A function that defines how sample points are selected for exploration. Defaults to Latin Hypercube Sampling (LHS).

The tree generation script accepts the following command-line arguments:

| Argument           | Type    | Description |
|-------------------|--------|-------------|
| `--seed`          | `int`  | Seed value for random generation. |
| `--initial_height` | `int`  | Initial tree height. |
| `--max_height`    | `int`  | Maximum tree height. |
| `--initial_points` | `int`  | Initial number of points. |
| `--points`        | `int`  | Number of points per iteration. |
| `--cores`         | `int`  | Number of CPU cores used. |
| `--output`        | `str`  | Output directory. |
| `--zone`          | `str`  | Specific verification zone. |

### Performing Analysis

After generating a set of trees, execute the pipeline script:

```
bash pipeline.sh
```

This script:
- Selects the best certified trees (lowest height and node count) from each generated collection.
- Constructs a **unified certified tree**.
- Trains standard machine-learning-based trees for comparison.
- Executes comparisons between models.
- Generates results in the following format:

#### Pipeline Output Files
Upon completion, all results will be stored in `resources/verified_<case_study>`, containing:
- `tree_<zone>.pkl`: The best-certified trees found for each zone.
- `tree.pkl`: The final unified certified tree.
- `simple_tree.pkl` and `forest_tree.pkl`, the baseline machine learning trees.
- `timings_data.csv`: Feasibility and timing comparison data.
- `obstacle_comparison_time.pdf`: Graphical representations of timings results.
- `latex_tables.tex`: Feasibility and timings comparisons report in the form of LaTeX tables.

## Package Extensibility

### Constructing New Case Studies

The package allows for addition of new case studies by extending the `Solvable` abstract class. This class provides a structured interface for defining multi-parametric Mixed-Integer Optimization (mp-MIO) problems.

To implement a new case study, users must create a subclass of `Solvable` and provide implementations for the methods. At this point the instance of the newly created subclass can be passes as inputs to the `run_experiment` function. Examples of implementations of the interface for the presented case studies can be found in the `src/examples/models` folder. 

### Customizing Tree Construction

The package allows users to use custom splitters and samplers. In particular:
- A splitter is an instance of a `Callable[[np.ndarray, np.ndarray, Polyhedron], Optional[Tuple[np.ndarray, float]]]`, thus a function accepting an array of input data, of corresponding output labels and the polyhedral zone and returns a linear splitter composed of the linear coefficients of the hyperplane as well as the additive bias.
- A sampler is an instance of a `Callable[[Solvable, int, Polyhedron], Tuple[np.ndarray, np.ndarray]]`, thus a function accepting the optimization problem, the number of points to generate and the domain we want to sample and returns a list of input data with corresponding output labels.

To override these defaults, pass custom implementations as function arguments to the `run_experiment` function. Examples for various implementations of custom splitters and samplers can be found respectively in `src/util/tree_util.py` and `src/util/sample_util.py`.

### Example Custom Script

Here is example code for a script involving a custom case study and custom splitter and sampler implementations:

```
from argparse import Namespace
from typing import Callable, Optional, Tuple

import numpy as np

from set.polyhedron import Polyhedron
from system.solvable import Solvable
from tree.generator.tree_generator import parse_arguments, run_experiment, save_tree

solvable_instance: Solvable = ...
parameter_domain: Polyhedron = ...

custom_splitter: Callable[[np.ndarray, np.ndarray, Polyhedron], Optional[Tuple[np.ndarray, float]]] = ...
custom_sampler: Callable[[Solvable, int, Polyhedron], Tuple[np.ndarray, np.ndarray]] = ...

args: Namespace = parse_arguments()

# Run the experiment with custom splitter and sampler
tree = run_experiment(
    args=args,
    solvable=solvable_instance,
    parameter_domain=parameter_domain,
    max_tasks=500,
    splitter=custom_splitter,
    sampler=custom_sampler
)

save_tree(tree, args.output, args.seed)
```

This script will generate a tree for the custom case study, using the custom sampler and splitter and the given command line arguments and will consequently save the tree into the file specified by output file argument.

## Results Data

The folder `paper_data` contains the trees and the data regarding the comparison of the tree that has been presented inside the paper.