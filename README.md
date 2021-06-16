# Satelite-image-road-extraction
The file is based on a published paper cited as: (Even though slight variations are there)
"Z. Zhang, Q. Liu and Y. Wang, "Road Extraction by Deep Residual U-Net," in IEEE Geoscience and Remote Sensing Letters, vol. 15, no. 5, pp. 749-753, May 2018, doi: 10.1109/LGRS.2018.2802944."
Variations include:
1. The loss function as changed into binary cross entropy rather mean squared error.
2. Iterations have been stopped at 15 rather at 50 since due to computation limit.
3. The output given below will be blurred as the training was stopped at 15 epochs due to computation limitaion.
## Trial output form the network
![Figure_1](https://user-images.githubusercontent.com/81060461/122214817-271e2480-cec8-11eb-9016-f9407ae20521.png)
