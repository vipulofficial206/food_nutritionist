# 🍎 Food Nutritionist – AI Based Food Recognition & Nutrition Analyzer

## 📌 Project Overview

Food Nutritionist is an Artificial Intelligence powered application that detects food items from images and provides their nutritional information. The system uses a Convolutional Neural Network (CNN) trained on the **Food-100 dataset** to classify food images and estimate dietary values such as calories, protein, carbohydrates, and fat content.

This project aims to help users understand their daily food intake and make healthier dietary decisions.

---

## 🎯 Objectives

* Automatically recognize food from an uploaded image
* Provide nutrition details of the identified food
* Assist users in calorie tracking and diet awareness
* Offer dietary suggestions using a nutrition knowledge base

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Convolutional Neural Networks (CNN)
* FAISS (Facebook AI Similarity Search) for knowledge retrieval
* Jupyter Notebook
* REST API (foodapi.py)

---

## 📂 Dataset Used

**Food-100 Dataset**

The Food-100 dataset contains categorized food images belonging to 100 different food classes.
It is used to train the CNN model to accurately identify food items from real-world images.

Dataset Features:

* Multiple food categories
* Real-life food images
* Suitable for image classification tasks
* Used widely for food recognition research

---

## ⚙️ System Workflow

1. User uploads a food image
2. CNN model processes the image
3. Model predicts the food class
4. System retrieves nutrition data from the knowledge base
5. Nutrition details are displayed:

   * Calories
   * Protein
   * Carbohydrates
   * Fat

---

## 🏗️ Project Structure

```
food_nutritionist/
│── modeltrain2.ipynb        # CNN training notebook
│── foodapi.py               # API for prediction
│── build_faiss_index.py     # Nutrition knowledge base search
│── foodmodels/              # Saved trained model
│── nutrition/               # Nutrition information database
│── KnowledgeBasenew/        # FAISS vector index
│── output images            # Model prediction results
```

---

## 🚀 How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/vipulofficial206/food_nutritionist.git
cd food_nutritionist
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run API

```bash
python foodapi.py
```

### 4. Predict

Upload a food image and the system will return the predicted food item and its nutritional values.

---

## 📊 Output

The model predicts:

* Food name
* Calories
* Protein
* Carbohydrates
* Fat

---

## 🔮 Future Improvements

* Mobile app integration
* Real-time camera detection
* Calorie tracking dashboard
* Personalized diet recommendations

---

## 👨‍💻 Author

Vipul

---

## ⭐ Conclusion


This project demonstrates the practical use of Deep Learning and Computer Vision in healthcare and nutrition monitoring systems. It can be extended into a diet assistant or smart health tracking application.
