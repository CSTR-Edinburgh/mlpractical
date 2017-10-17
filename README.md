# Machine Learning Practical

This repository contains the code for the University of Edinburgh [School of Informatics](http://www.inf.ed.ac.uk) course [Machine Learning Practical](http://www.inf.ed.ac.uk/teaching/courses/mlp/).

This assignment-based course is focused on the implementation and evaluation of machine learning systems. Students who do this course will have experience in the design, implementation, training, and evaluation of machine learning systems.

The code in this repository is split into:

  *  a Python package `mlp`, a [NumPy](http://www.numpy.org/) based neural network package designed specifically for the course that students will implement parts of and extend during the course labs and assignments,
  *  a series of [Jupyter](http://jupyter.org/) notebooks in the `notebooks` directory containing explanatory material and coding exercises to be completed during the course labs.

## Getting set up

Detailed instructions for setting up a development environment for the course are given in [this file](notes/environment-set-up.md). Students doing the course will spend part of the first lab getting their own environment set up.

## IMPORTANT Coursework Setup

For coursework 1, 2 more libraries are required which add nice progress bar functionality.
To install them run in your conda mlp environment:

```conda install -c conda-forge ipywidgets```

and

```conda install tqdm```

Then your ipython notebook should be able to produce progress bar for your training and validation phases.

If you get javascript errors try running the following command in the terminal and restarting the notebook:

```jupyter nbextension enable --py --sys-prefix widgetsnbextension```
