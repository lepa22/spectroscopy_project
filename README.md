# spectroscopy_project

This project is a set of Python 3 .py files containing classes and functions
that are meant to be used for the pre- and post-proccessing, as well as
statistical analysis, of Raman spectra and spectroscopic data in general.

The project is developed and meant to be used in Jupyter Notebooks, since it
makes use of the advanced file import, processing and display features of
pandas's DataFrames, although, if desired, it should be easy to be converted
for usage in Python 3 consoles.

## Requirements:

- Jupyter Notebook
- Matplotlib
- NumPy
- pandas
- scikit-learn
- PeakUtils
- silx

The Anaconda Python distribution is highly recommended, since it works on all
major platforms and contains most of the above packages by default. The
packages that are not contained by default in Anaconda is recommended to be
installed using conda.

## To-do:

**Caution:** This project is currently under heavy developement and is
considered to be in a pre-alpha stage. Currently it provides just a bunch of
functions for the most basic Raman spectra proccessing tasks. Many features
will be added during the next months.

Some of the things that are to be added or changed are:

- Name the project
- Documentation
- Docstrings for most functions
- Option to save multiple plots
- First and second order differeantiation functions
- Peak detection functions
- .spc files import and export to .csv
- Spectra normalization functions
- Scaling functions
- Plotting of DataFrames
- more...

For suggestions, bugs and ideas contact me via github or email.