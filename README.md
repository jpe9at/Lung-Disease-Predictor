Train a CNN on the Chest X-Ray Images (Pneumonia) from Kaggle and apply the DeepSHAP explainer to it. 

The dataset can be obtained at: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The programm mainly serves the purpose to ascertain the viability of the DeepSHAP explainer. The main question was, whether its explanaitions are accessible for the doctor or for the layperson alike. 
From the layperson's perspective the results seem to be quite difficult to interpret. The only observation possible from that perspective is cases where the predictions of the CNN are based on obviously inappropriate parts of the images (like areas outside the ribcage). 
