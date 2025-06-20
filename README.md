# Doodles-For-Emotion
Emotional Analysis of children using doodles
The project sets out to build a binary classifier that categorizes doodles as either “good” (indicative of a mentally fine child) or “bad” (suggestive of emotional or psychological distress). This system could potentially assist mental health professionals by acting as a pre-screening tool or an auxiliary decision-support system. By applying state-of-the-art deep learning techniques and ensuring ethical and clinical responsibility, the proposed model is designed with the long-term goal of enhancing child psychology practices.

# I. METHODOLOGY
The methodology adopted in this project follows a systematic flow—from data collection and preprocessing to model development and evaluation.

i) DATASET PREPARATION 
Due to the absence of any pre-labeled dataset distinguishing mentally fine and distressed children's doodles, this project required manual curation. Publicly shared doodles were sourced and categorized based on visual indicators such as color tone, stroke intensity, and thematic elements. Drawings expressing happiness, stability, and creativity were classified as “good,” whereas those reflecting darkness, chaos, or fear were labeled as “bad.”
The images were resized to 224x224 pixels for compatibility with Vision Transformer input dimensions. Standard normalization techniques were applied to standardize the pixel values. Data augmentation strategies such as rotation, flipping, and noise injection were also employed to improve model generalizability and reduce overfitting.

ii) MODEL DEVELOPMENT 
A pre-trained Vision Transformer model (google/vit-base-patch16-224) was fine-tuned on the curated doodle dataset. The ViT model was chosen due to its superior ability to capture contextual relationships in image data. The classifier head was modified to perform binary classification. Cross-entropy loss was used as the objective function, and the AdamW optimizer was selected with a learning rate of 5e-5. The model was trained for five epochs with a batch size of 32.
The training process showed steady improvements in accuracy with each epoch. Initial validation accuracy started at 95% and stabilized at 100%, indicating strong learning patterns but also signaling possible overfitting due to dataset limitations.
Training Output:
 
# II. EVALUATION METRICS 
To assess model performance, several metrics were applied: accuracy, precision, and recall. In addition to numerical scores, three visualization tools were employed:
•	A confusion matrix to depict correct and incorrect classifications
•	A Precision-Recall curve to examine model confidence
•	A ROC-AUC curve to evaluate overall classification quality and separability

# III. RESULTS AND OBSERVATIONS
The evaluation of the model involved a combination of traditional metrics and visualization techniques to ensure both statistical accuracy and interpretability. The transformer-based model showed a consistent increase in classification performance across epochs, which was confirmed through training and validation metrics. During the final epoch, the model achieved a perfect validation accuracy of 100%, having started from an initial 95%. The training loss continually decreased, indicating convergence and learning stability. While these figures suggest robust model performance, caution is required due to the limited dataset size, which may have led to overfitting.
|Model	                | Accuracy	| Precision	| Recall | 
|YOLOv8	               | 65%	    | Moderate	 |Moderate |
|Attention Transformer	|100%	    | High    	 |High     |
The confusion matrix provided a clear picture of the model’s ability to correctly distinguish between the two classes. As shown below, all test images from the "Good" and "Bad" categories were accurately classified without any misclassifications:
 

To further analyze classifier confidence and performance across thresholds, precision-recall and ROC-AUC curves were plotted. The precision-recall curve showed near-perfect precision at all recall levels, indicating that the model rarely misclassifies a “Bad” doodle as “Good.” This is significant from a clinical perspective, as false negatives (failing to identify an emotionally distressed child) can have serious consequences.
Precision-Recall Curve:
The ROC-AUC curve demonstrated an area under the curve (AUC) of 1.00, suggesting that the model has exceptional discriminatory power. In practical terms, this means that the classifier can separate the two classes with extremely high accuracy across different thresholds.
  
ROC-AUC Curve:
 
Overall, the results validate the transformer model’s strength in learning emotional cues from abstract visual data. Nonetheless, these outcomes also highlight the necessity for real-world testing and the acquisition of a more diverse dataset to ensure the model generalizes beyond the initial experimental setup.
