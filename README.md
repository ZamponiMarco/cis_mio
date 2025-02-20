# Certified Inductive Synthesis for Online Mixed-Integer Optimization

## Installation

It is possible to manually install the package or to run a container with the corresponding Docker image.

### Manual Installation

The following instructions regard the installation of the package in an Ubuntu OS environment.
To manually install the package it is necessary to install the following packages by running the following command from the root folder:

```
apt-get update
apt-get install libcdd-dev libgmp-dev python3-dev python3-venv git 
```

We can now create a virtual environment and activate it. In here, we call the such environment `venv`:

```
python -m venv venv
source venv/bin/activate
```

Subsequently, we clone the repository and install the package requirements from the `requirements.txt` file:

```
git clone https://github.com/ZamponiMarco/cis_mio
cd cis_mio
pip install -r requirements.txt
```

The package is now ready to be used.

### Docker Image

A ready to use environment with all the installed dependencies can be easily created by building the image described in the `Dockerfile` file and running a container with the given image running the following commands from the project root folder:

```
docker image build . -t cis_mio:iccps
docker run -it cis_mio:iccps
```

In this case you will be presented with a Bash shell where we can run the project scripts.

### Docker Hub

The Docker image is also hosted on a public image repository on Docker Hub, and is thus possible to directly run a container containing the such image by executing the command:

```
docker run -it zamponimarco/cis_mio:iccps
```

### Note

The example optimization problems presented in this package work fine using the size-limited Gurobi license automatically included in the installation of the `gurobipy` Python package. In order to run bigger optimization problems a [Gurobi license](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license) will be needed.

## Running the Software

The easiest way to run the software is executing the scripts inside the `script` folder. In here, there are two different case studies, and the corresponding scripts are contained inside the folder.

There are three scripts in each folder, which are divided into two main functions, certified tree construction and analysis of certified tree performance. 

### Constructing Trees

In order to construct a collection of trees for a particular case study, we navigate to the folder `script/<case_study>` for the corresponding case study of interest and we execute the command:

```
bash generate_tree.sh
```

In this case, a collection of trees for each subdomain will be created and stored into the `resources` folder. Note that inside the script we can configure the amount of trees to be generated and other hyperparameters such as tree initialization points, initialized tree height, points used at each refinement step and maximum tree height.

In order to provide reproducible results we also added an alternative version that allows to generate the trees used for the paper results. In this case we need to run the command:

Note that generating a collection of trees does not guarantee that they are completely certified, since we truncate the construction when a configured maximum height is reached, thus it may be needed to generate additional trees for specific subdomains.

```
bash generate_tree_seeded.sh
```

### Performing Analysis

After generating some trees we are interested in performing the analysis pipeline. The such pipeline is tasked with identifying the best trees between the generated ones, merge them into a single tree, constructing standard machine learning based trees, performing the comparison and reporting the results. In order to execute the pipeline we run the command:

```
bash pipeline.sh
```

After the pipeline execution all the results will be stored inside the folder `resources/verified_<case_study>`, which specifically will contain:

- The best certified trees found
- The unified certified tree
- The standard machine learning based trees used for comparison
- The data table containing timing and feasibility reports
- Generated images and latex tables for the result presentation

## Results Data

The folder `paper_data` contains the trees and the data regarding the comparison of the tree that has been presented inside the paper.