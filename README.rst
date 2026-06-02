|title|_
========

.. |title| replace:: wdfkit
.. _title: https://dshirya.github.io/wdfkit

|PyPI| |Forge| |PythonVersion| |PR|

|Codecov| |Black| |Tracking|


.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |Codecov| image:: https://codecov.io/gh/dshirya/wdfkit/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/dshirya/wdfkit

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/wdfkit
        :target: https://anaconda.org/conda-forge/wdfkit

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff
        :target: https://github.com/dshirya/wdfkit/pulls

.. |PyPI| image:: https://img.shields.io/pypi/v/wdfkit
        :target: https://pypi.org/project/wdfkit/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/wdfkit
        :target: https://pypi.org/project/wdfkit/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/dshirya/wdfkit/issues

.. image:: https://raw.githubusercontent.com/dshirya/wdfkit/main/docs/source/img/logo.png
        :alt: wdfkit logo
        :width: 600px
        :align: center

About ``wdfkit``
----------------

``wdfkit`` is a Python toolkit for **Renishaw WiRE** ``.wdf`` spectroscopy data—especially **Raman** and **photoluminescence** work. It helps you bring measurements out of the instrument format so you can explore and analyse them in Python: load single spectra, line scans, and maps; interpret wavelength or Raman-shift axes consistently; and prepare data with everyday steps such as **normalization**, **cosmic-ray spike removal**, reduction of **laser-related spectral artefacts**, and **noise suppression** for stacks of spectra.

The project is **inspired by** `spectrapy <https://gitlab.in2p3.fr/dejan.skrelic/spectrapy>`__ by **Dejan Skrelic**—an earlier tool that shaped how spectroscopy users treat this kind of data.

For more information about the wdfkit library, please consult our `online documentation <https://dshirya.github.io/wdfkit>`_.

Citation
--------

If you use wdfkit in a scientific publication, we would like you to cite this package as

        wdfkit Package, https://github.com/dshirya/wdfkit

Installation
------------

The preferred method is to use `Miniconda Python
<https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_
and install from the "conda-forge" channel of Conda packages.

To add "conda-forge" to the conda channels, run the following in a terminal. ::

        conda config --add channels conda-forge

We want to install our packages in a suitable conda environment.
The following creates and activates a new environment named ``wdfkit_env`` ::

        conda create -n wdfkit_env wdfkit
        conda activate wdfkit_env

The output should print the latest version displayed on the badges above.

If the above does not work, you can use ``pip`` to download and install the latest release from
`Python Package Index <https://pypi.python.org>`_.
To install using ``pip`` into your ``wdfkit_env`` environment, type ::

        pip install wdfkit

If you prefer to install from sources, after installing the dependencies, obtain the source archive from
`GitHub <https://github.com/dshirya/wdfkit/>`_. Once installed, ``cd`` into your ``wdfkit`` directory
and run the following ::

        pip install .

This package also provides command-line utilities. To check the software has been installed correctly, type ::

        wdfkit --version

You can also type the following command to verify the installation. ::

        python -c "import wdfkit; print(wdfkit.__version__)"


To view the basic usage and available commands, type ::

        wdfkit -h

Getting Started
---------------

You may consult our `online documentation <https://dshirya.github.io/wdfkit>`_ for tutorials and API references.

Support and Contribute
----------------------

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/dshirya/wdfkit/issues>`_ and/or `submit a fix as a PR <https://github.com/dshirya/wdfkit/pulls>`_.

Feel free to fork the project and contribute. To install wdfkit
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/dshirya/wdfkit/blob/main/CODE-OF-CONDUCT.rst>`_.

Contact
-------

For more information on wdfkit please visit the project `web-page <https://dshirya.github.io/>`_ or email the maintainers ``Danila Shiryaev(danila.shiryaev@polytechnique.edu)``.

Acknowledgements
----------------

``wdfkit`` draws conceptual inspiration from `spectrapy <https://gitlab.in2p3.fr/dejan.skrelic/spectrapy>`__ by Dejan Skrelic.

``wdfkit`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.
