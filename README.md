# Until Academy Launch

## What is this repo?

This repo contains a series of Learning Units. A Learning Unit is one hour of material
meant to convey a few related concepts in the form of

- a short (<15 min) presentation
- some example notebooks
- some practice exercises

During the academy, we will be going through these one-by-one listening
to the instructors give their short presentations, checking out the examples,
and doing the practice exercises.

### Setup

1. Do the [setup](https://github.com/LDSSA/setup).
1. Clone this repo.
    - Just do whatever you did to clone the setup repo except for this one instead.
1. Open a jupyter notebook and navigate to the `units` directory of this repo
    - If you used anaconda, just navigate to where you cloned this repo
    - If you're using Docker, run `run-jupyter-notebook.sh`

### Usage

#### Structure
Inside the `units` directory, there are 19 directories, each of them containing
a Learning Unit of roughly the following format:

```
├── README.md
├── data
├── Examples Notebook - <name of the Learning Unit> 
├── Exercise Notebook - <name of the Learning Unit> 
├── Learning Notebook - <name of the Learning Unit> 
```

- The README.md is meant to be viewed *in a browser on github.com* and contains.
    - The purpose of the unit.
    - The concepts being presented.
    - A link to the presentation.
    - The practice assignment.
- The data directory contains the datasets that the instructors have
  chosen for the examples and practice exerciess.
- The Learning Notebook is where you will do most of your learning. Read it carefully! 
- The Exercise Notebook is where you will execute what you've learned. 
- The Examples Notebook is a kind of "cheat sheet" that you can use to recall how  to use functions. It does **not** teach you any details. 
 
#### If (when) you have doubts about the materials  

#### If (when) you find bugs, typos or minor mistakes 
This repo is completely open source and is continuously improving over time. When you spot a mistake, please check whether it has been detected in the [issues](https://github.com/LDSSA/bootcamp/issues). If it hasn't, please open an issue, explaining in details where it is (e.g. in what notebook, and on what line), and how to reproduce the error. If it is an easy fix, feel free to make a pull request.  
