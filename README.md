# McCaD: Multi-Contrast MRI Conditioned, Adaptive Adversarial Diffusion Model for Accurate Healthy and Tumor Brain MRI Synthesis
This  repo contains the supported pytorch code and configuration files to reproduce medical image synthesis results of McCaD.

![alt text](img/McCaD_architecture.png)

Network architecture of McCaD. **A**: Overall Architecture, **B**: Muti-scale Feature Guided Denosing Network, **C**: Adaptive Feature Maximizer, **D**: Feature Attentive Loss.

**Environment**
Please prepare an environment with python>=3.8, and then run the command "pip install -r requirements.txt" for the dependencies.

**Data Preparation**
For experiments, we used two datasets:
  * Tumor Dataset : [BRaTS Dataset](http://braintumorsegmentation.org/)
  * Healthy Dataset
    
Dataset structure
       ```
![file_structure](https://github.com/sanuwanihewa/McCaD/assets/124846386/66edb103-e29a-46ad-84d3-e9483832d105)


