# RTL Combinational Depth Predictor

An AI-powered tool to predict combinational logic depth in RTL designs without requiring synthesis, enabling early identification of potential timing violations in VLSI design projects.

## Problem Statement

Timing analysis is a crucial step in designing complex IP/SoC components. However, timing analysis reports are generated only after synthesis is complete, which is a very time-consuming process. This leads to overall delays in project execution, as timing violations often require architectural refactoring.

This tool creates an AI algorithm to predict the combinational logic depth of signals in behavioral RTL code, greatly speeding up the design process by identifying potential timing issues before synthesis.

## Key Features

- **Pre-Synthesis Prediction:** Predict combinational logic depth directly from RTL code.
- **Multiple ML Models:** Supports Random Forest, Gradient Boosting, SVM, and Neural Network models.
- **Advanced Feature Extraction:** Extracts 25+ relevant features from RTL code.
- **Technology Node Awareness:** Accounts for different semiconductor technology nodes.
- **Signal-Specific Analysis:** Examines signal characteristics for accurate predictions.
- **Visual Results:** Graphical representation of predicted depths and potential timing violations.
- **Synthetic Data Generation:** Generate realistic RTL datasets for training and testing.

## Table of Contents

1. [Installation](#installation)
2. [How to Run](#how-to-run)
3. [Dataset](#dataset)
4. [Understanding the Model](#understanding-the-model)
5. [How It Works](#how-it-works)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Contributing](#contributing)
9. [License](#license)

---

## Installation

How to Run
1. Clone the Repository:
Clone this repository to your local machine:

git clone https://github.com/Ramiksha24/Google--AI-algorithm-to-identify-timing-violations-Girl-Hackathon.git
cd Google--AI-algorithm-to-identify-timing-violations-Girl-Hackathon
2. Train the Model:
To train the model, run:

python model_training.py
3. Evaluate the Model:
Evaluate the trained model by running:

python evaluate_model.py
4. Make Predictions:
For predictions on new RTL module data, use the following command:
python predict.py

## Dataset

The dataset used in this project contains synthesis report data with the following features:

1. **module_name**: Name of the RTL module.
2. **module_type**: Type of the module.
3. **signal_name**: Name of the signal.
4. **signal_width**: Width of the signal.
5. **is_output**: Boolean flag indicating whether the signal is an output.
6. **is_registered**: Boolean flag indicating whether the signal is registered.
7. **fanin**: Number of input signals feeding into the signal.
8. **fanout**: Number of output signals driven by the signal.
9. **combinational_ops**: Number of combinational operations.
10. **arithmetic_ops**: Number of arithmetic operations.
11. **mux_ops**: Number of multiplexing operations.
12. **always_blocks**: Number of always blocks in the RTL code.
13. **case_statements**: Number of case statements.
14. **if_statements**: Number of if statements.
15. **loop_constructs**: Number of loop constructs.
16. **module_complexity**: Overall complexity of the module.
17. **technology_node_nm**: Technology node (in nanometers).
18. **clock_frequency_mhz**: Clock frequency (in MHz).
19. **optimization_level**: Level of optimization applied to the design.
20. **combinational_depth**: Target variable (combinational logic depth).

Understanding the Model
Input Features
The model uses the following features extracted from RTL code:

Module characteristics:

Module type (ALU, Multiplier, etc.)
Module complexity score
Architecture type (Pipeline, Parallel, etc.)
Signal characteristics:

Signal fan-in (number of signals that affect this signal)
Signal fan-out (number of other signals affected by this signal)
Signal width (number of bits)
Is output signal (boolean)
Is registered signal (boolean)
RTL complexity metrics:

Number of always blocks
Number of assignments
Number of case statements
Number of if statements
Number of loop constructs
Combinational logic operations count
Arithmetic operations count
Technology parameters:

Technology node (nm)
Clock frequency (MHz)
Output
Combinational Depth: Predicted number of logic levels in the critical path.
Interpreting Results
Depth < 5: Low complexity, likely to meet timing constraints.
Depth 5-10: Medium complexity, may need attention for high-frequency designs.
Depth > 10: High complexity, likely to cause timing violations.
How It Works
The RTL Combinational Depth Predictor follows these steps:

Feature Extraction: Analyzes RTL code to extract relevant features.

Parses Verilog/VHDL code using regex patterns.
Extracts module and signal characteristics.
Identifies control structures and operations.
Signal-Specific Analysis:

Identifies signal declarations and assignments.
Analyzes signal dependencies.
Calculates estimated fan-in/fan-out.
Prediction:

Preprocesses features (scaling, encoding).
Applies trained machine learning model.
Returns predicted combinational depth.
Risk Assessment:

Evaluates predicted depth against thresholds.
Identifies signals at risk of timing violations.
Provides recommendations.
Results
The trained model is capable of predicting the combinational depth of signals in RTL modules with a high degree of accuracy. A detailed report on model performance and results is included in the documentation.

Future Work
Incorporate additional features such as environmental factors like temperature and process variation impact.
Experiment with deep learning models to improve prediction accuracy.




