# Robox

A kit for robust optimization and risk averse stochastic programming.

Robox stands for Robust Optimization Box.
It is a Julia package designed to used general risk measures
for robust machine learning.

More precisely, the Robox project implements algorithms of the paper
[General risk measure for robust machine learning](https://arxiv.org/pdf/1904.11707.pdf)
and aims at providing a struture that is reliable for reproducing results
and flexible for applying it to a wide range of sectors.
This research was conducted during Henri Gérard's PhD under the supervision of
Jean-Christophe Pesquet and Émilie Chouzenoux.

See <https://www.linkedin.com/in/henri-g%C3%A9rard-phd-85826789/>
for details about Henri Gérard's career.


## Installation
------------

### Third-party dependencies
Note that third parties dependencies are not required for a working installation.
It is useful to benchmark our algorithm, reproduce and compare results.
#### Solver
Robox requires the installation of a convex solver. Several possibilities are available.
If you are looking for a free product, we advise you to install Ipopt for better performance.
Instructions are available on this site:
<https://www.coin-or.org/Ipopt/documentation/node10.html>

### Dependencies
Robox requires:


### User installation

## Development

We welcome new contributors of all experience levels. Contact the authors if you are
interested. Some information about contributing code, documentation, and tests are
included in this README.


## Testing

After installation, you can launch the test suite from the
tests directory:

    cd robox/tests/
    julia run_tests.py

By default, tests that require a solver are ignored.

## Examples

After installation, you can launch examples from outside the
source directory:

    cd examples/

By default, the solver parameter is set to use ipopt.

The actions of these examples are detailed in the comments of each files.

## Instructions for commits
Please follow the instructions below when you commit:

* begin your commit message by [DEV] if it is development,
[UPD] if you made some updates, [FIX] if you fixed some errors,
* Leave a space and start your message with a Upper Case letter,
* Example : "[UPD] Add a license file".

Following this instruction will improve a lot research in previous commits.


## Instructions for code style
Please follow the instructions of the PEP8 cheatsheet:
https://www.python.org/dev/peps/pep-0008/


## Documentation
