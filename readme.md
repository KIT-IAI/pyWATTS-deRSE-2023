# pyWATTS deRSE 2023

Welcome to the Python Workflow Automation Tool for Time-Series (pyWATTS) repository for the deRSE23.

This repository contains Jupyter notebooks to replicate the non-sequential machine learning pipelines demonstrated in
our talk and poster. You can also find the PDF slides of our talk and a copy of our poster.

## Getting Started
To run the jupyter notebooks:
* Create a new python environment.
* Install the requirements via ``pip install -r requirements.txt``.

You should now be able to run all the notebooks without any problems.

### A Special Note for Mac Users
If you want to run our notebooks on an Apple computer, you first have to install TensorFlow separately:
1. Install Apple TensorFlow dependencies: ``conda install -c apple tensorflow-deps``
2. Install TensorFlow for macOS: ``pip install tensorflow-macos``
3. Install TensorFlow metal for GPU usage: ``pip install tensorflow-metal``

More information regarding TensorFlow for macOS can be found through this helpful [installation guide](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706).

## Further Information on pyWATTS
Further information on pyWATTS can be found in the:
* [pyWATTS Documentation](https://pywatts.readthedocs.io/en/latest/)
* [pywatts-pipeline Repository](https://github.com/KIT-IAI/pywatts-pipeline)
* [pyWATTS Repository](https://github.com/KIT-IAI/pyWATTS)

## Funding
This project is supported by the Helmholtz Association under the Program “Energy System Design”, by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Joint Initiative "Energy System 2050 - A Contribution of the Research Field Energy", and by the German Research Foundation (DFG) Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".
