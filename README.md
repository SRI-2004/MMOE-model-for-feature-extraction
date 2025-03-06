
# **Multi-Modal Mixture of Experts (MMoE) for Multi-Attribute Classification**  

## **Overview**  
This repository presents a **Mixture of Experts (MMoE) model** for **multi-attribute classification** using **DINOv2 as a feature extractor**. The model is designed for structured prediction tasks where multiple attributes are assigned to each image. Instead of using a **single shared representation for all tasks**, the **MMoE architecture dynamically routes features** through specialized expert networks, enhancing **task-specific learning and computational efficiency**.  

## **Key Contributions & Differentiators**  

- **Dynamic Feature Routing** – The model leverages **task-specific gating networks** that **dynamically select experts**, unlike traditional models that share a single feature extractor for all attributes.  
- **DINOv2-Based Feature Extraction** – Uses a **self-supervised Vision Transformer (ViT)** to **generate high-quality embeddings**, improving generalization and reducing training time.  
- **Efficient Multi-Output Learning** – Unlike standard CNN-based classifiers, this model **processes multiple attributes in parallel** while **minimizing computational redundancy**.  
- **Scalable for Real-World Applications** – Well-suited for **fashion attribute classification, medical imaging, and automated tagging in e-commerce**.  
- **Robust Data Augmentation** – Incorporates **Albumentations-based transformations** like **RandomResizedCrop, CoarseDropout, and ColorJitter** to improve generalization.  

## **Architecture**  

### **1. Feature Extractor – DINOv2 (Self-Supervised ViT)**  
- The model uses **DINOv2 (facebook/dinov2-base)** as a **pretrained vision transformer** to **extract high-dimensional feature embeddings** from images.  
- Unlike standard CNNs, **DINOv2 learns richer semantic representations** without requiring extensive labeled data.  

### **2. Shared Feature Processing**  
- Feature embeddings are **refined through a fully connected layer with BatchNorm, ReLU activations, and Dropout** to improve generalization.  

### **3. Mixture of Experts (MMoE) Layer**  
- **Multiple expert networks** process shared features, allowing task-specific learning.  
- **Task-specific gating networks** dynamically route features to **the most relevant experts** using **softmax-based selection**.  

### **4. Task-Specific Classification Heads**  
- Each output head predicts a **specific attribute category** (e.g., color, pattern, sleeve type).  
- Uses **fully connected layers with ReLU activations and Softmax** for final predictions.  

---

## **Dataset & Preprocessing**  
- The dataset consists of **fashion images** labeled with **multiple attributes** (e.g., **color, neckline, pattern, sleeve length**).  
- Categories include **Men’s T-Shirts, Sarees, Kurtis, and Women’s Tops**, each having **distinct attributes**.  
- **Data Augmentation**: Uses **Albumentations** for **random crops, horizontal flips, distortions, and color transformations**.  

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/mmoe-multi-attribute.git
cd mmoe-multi-attribute
```

### **2. Install Dependencies**  
```bash
pip install torch torchvision transformers albumentations numpy pandas matplotlib tqdm scikit-learn
```

---

## **Training & Evaluation**  

### **Train the Model**  
Run the training script with:  
```bash
python train.py
```
- **Optimizer**: AdamW with **different learning rates** for **DINOv2** and **classification heads**.  
- **Loss Function**: CrossEntropy Loss for multi-attribute classification.  
- **Learning Rate Scheduler**: ReduceLROnPlateau (adaptive learning rate adjustment).  
- **Best Model Checkpointing** based on **highest weighted F1-score**.  

### **Evaluate the Model**  
Run evaluation on the validation set:  
```bash
python evaluate.py
```

---





## **Why This Model Is an Improvement**  

### **1. Task-Specific Feature Learning**  
- Unlike **fully shared multi-task models**, this model dynamically **routes features through expert networks**.  
- **Task-specific gating** improves performance on **multi-label classification** problems.  

### **2. Efficient Expert Routing**  
- Standard deep learning models **process all features uniformly**, leading to **redundant computations**.  
- In contrast, the **MMoE model assigns tasks dynamically**, reducing unnecessary processing while improving accuracy.  

### **3. DINOv2 for High-Quality Feature Extraction**  
- Instead of training a CNN from scratch, **DINOv2 provides semantically rich embeddings**.  
- This improves **feature generalization** and **reduces training time**.  

### **4. Scalable to Real-World Multi-Attribute Classification**  
- Unlike standard single-class models, **MMoE efficiently handles structured outputs**.  
- Can be applied to domains like **fashion AI, medical imaging, and e-commerce product tagging**.  

---

## **Future Work**  
- **Test on additional datasets** for improved generalization.  
- **Experiment with advanced expert selection mechanisms** to refine routing strategies.  
- **Implement Knowledge Distillation** to **reduce model size** for efficient deployment.  

---




