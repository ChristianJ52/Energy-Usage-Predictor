
## 📊 Prediction Results

This project includes a simple ML-style predictor for building energy usage.  
It takes inputs like outdoor/indoor temperature, building area, and insulation quality, then estimates kWh consumption.  

Below is an example of the model's performance after logging 5 sample predictions:

<img width="1002" height="674" alt="Prediction results graph" src="https://github.com/user-attachments/assets/e85b4fd6-49a1-4d6a-a603-95732a7d0bc0" />


- **Left (scatter plot):** Predicted vs. actual energy use.  
  The red line shows a “perfect” prediction — points on this line would mean the model guessed exactly right.  
- **Right (bar chart):** Absolute error for each prediction.  
  Lower bars = closer to the actual usage.  

This is just a learning demo, but it shows the **core AI/ML workflow**:  
feature engineering → prediction → model evaluation → error analysis.
