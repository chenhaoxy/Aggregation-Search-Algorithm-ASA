# High-Dimensional-Feature-Selection-based-on-Aggregation-Search-Algorithm

## Abstruct
Evolutionary feature selection methods with fixed length encoding and traditional search mechanism have the problems of high computational cost and long optimization time for high-dimensional data. To address these issues, this study proposed an Aggregation Search Algorithm (ASA) based feature selection method, which adopts a three-dimensional variable structure encoding to achieve a flexible representation of feature combinations. Based on it, the fusion and fission operators were designed to search the high-dimensional feature space. In this paper, we presented the framework of ASA while making a mathematical analysis of the core steps and discussing the effect of key parameters. Twelve publicly datasets were employed in the experiments to validate the algorithm. These datasets were categorized into three groups: three with dimensions below 1000, three with dimensions ranging from 1000 to 5000, and six with dimensions exceeding 5000. Furthermore, a comparative analysis was conducted, contrasting the ASA with seven typical and state-of-the-art methods in the same domain. In 9 out of 12 test cases, the results of ASA achieved the best accuracy against the 7 competitors. In particular, the best accuracy was achieved in 4 out of 6 test cases with more than 5000 dimensions. From the perspective of the number of features in output, the ASA achieves the best performance on all datasets with dimensions above 1000. Therefore, the ASA demonstrated superior performance in most of high-dimensional feature selection tasks. 

## How to use
```
python mainMethod.py
```
