# ML Project 2023
Authors: Gianluca Panzani, Corrado Baccheschi, Emad Chelhi and coordinated by Prof. Alessio Micheli, University of Pisa

# The project
We had 4 tasks with a Dataset for each one:
- 3 of them were Binary Classification problems, as boolean expressions (Monks datasets),
- 1 of them was a Regression problem (CUP dataset).

# Implementation
We’ve compared three Neural Networks in Keras, Pytorch and Scikit-Learn, and one SVM, to see differences
both on models and tools.
For an effective comparison, we used the specific provided functions of each library without mixing them, when possible.
We made Model Selection and Model Assessment by using Grid-Search and K-fold Cross-Validation before testing all models, on an
Internal Test set for CUP, and on the given Test set (for Monks).

# Regression problem (CUP)
The selected model for the CUP is made with Pytorch because it retrieved the best results.
We focused on various levels of abstraction, flexibility and customizability offered by the different tools we investigated.

# Why we have chosen these frameworks?
Regarding the Neural Networks, we utilized:
- Keras for its higher-level functionalities and features,
- PyTorch for its greater flexibility and customizability.
Furthermore, regaring SVM's models, we used:
- Sklearn with those Neural Networks.
