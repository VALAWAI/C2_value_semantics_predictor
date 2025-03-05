# C2 Value Semantics Predictor

Python software for identifying bioethical values with machine learning for the VALAWAI project.

The current code also includes an evaluation of this software, and an example application of it for an example patient.


# Execution

- Run Main.py:

This file contains the software that extracts the value semantics from a data set of synthetic patients.
This file also evaluates the learned value semantics models against a test set of 2000 synthetically created patients.
For the training data set, four placeholder value semantics were used as the ground truth (they are provisional)
It is possible to re-train the value semantics models.

After the evaluation is finished, the same file will proceed to show an example application of the learned model.
For an example synthetic patient and a prefixed action treatment, the value semantics models will try to predict its associated degree of value alignment.


# Dependencies

- Latest version of numpy
- Latest version of scikit-learn
