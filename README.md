# Circuit_Performance_Prediction
This repository is the code for [Efficient Circuit Performance Prediction Using Machine Learning: From Schematic to Layout and Silicon Measurement With Minimal Data Input](https://ieeexplore.ieee.org/abstract/document/11114348). 

We present an ML-driven framework for predicting circuit performance metrics, bridging the gap between schematic and layout simulations, multi-process corner analysis, and measured silicon data. We demonstrate this using 14nm and 5nm FinFET-based ring oscillators, by collecting data across varying supply voltages, temperatures, and process corners. Using three baseline ML models—XGBoost, Random Forest, and a Neural Network—we simulate real-world design scenarios where parameter fine-tuning may not always be feasible. Key tasks include predicting layout performance from schematic data, performance prediction across process corners, and fabricated chip performance. Our results show that these models can achieve less than 5% mean absolute percentage error (MAPE) for power and frequency prediction while reducing required simulations by more than 2× . When migrating from 14nm to 5nm, XGBoost and Neural Network achieve high accuracy (>0.99 R<sup>2</sup>) using just 10% of the otherwise required 5nm simulations. We also present an extensive robustness analysis to demonstrate that our results are not limited to a single data split or initialization. By varying random seeds across multiple runs, we evaluate the stability of each model with respect to algorithm initialization and the selection of training data subsets. This demonstrates that the observed accuracy is consistent and not the result of a specific, favorable configuration. This framework offers a promising approach to accelerating circuit design across technology nodes by reducing simulation costs while maintaining accuracy in predicting performance.

The top-level directories consists of the models for each of the tasks described in the paper. We are unable to release the data due to NDA concerns.

If our work is useful to you, please cite our [paper]([https://ieeexplore.ieee.org/abstract/document/11114348]):

```
@article{kochar2025efficient,
  title={Efficient Circuit Performance Prediction Using Machine Learning: From Schematic to Layout and Silicon Measurement with Minimal Data Input},
  author={Kochar, Dimple Vijay and Ashok, Maitreyi and Cohn, John and Zhang, Xin and Chandrakasan, Anantha P},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers},
  year={2025},
  publisher={IEEE}
}

@article{kochar2025,
  title={Efficient Circuit Performance Prediction Using Machine Learning: From Schematic to Layout and Silicon Measurement with Minimal Data Input},
  author={Kochar, Dimple and Ashok, Maitreyi and Cohn, John and Chandrakasan, Anantha and Zhang, Xin},
  journal = {2025 IEEE International Conference Circuits and Systems (ISCAS)},
  year      = {2025},
  organization = {IEEE}
}


  ```


