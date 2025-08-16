# ---------------------------
# Energy Usage Predictor - AI/ML Demo
# Building Engineering meets Data Science
# ---------------------------

import os
import csv
import datetime
import matplotlib.pyplot as plt

# ---------------------------
# Utility Functions (extracted from calculator)
# ---------------------------

def ask_float(prompt):
    """Get valid float input from user"""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number like 12.5")

def c_to_f(celsius):
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9/5) + 32

def save_report(text, filename="prediction_report.txt"):
    """Save timestamped log entry"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

# ---------------------------
# Data Logging for ML Training
# ---------------------------

PREDICTION_LOG = "prediction_training_data.csv"

def log_prediction_data(outdoor_temp, indoor_temp, area, insulation_rating, predicted_kwh, actual_kwh=None):
    """Log prediction inputs and results for future ML training"""
    new_file = not os.path.exists(PREDICTION_LOG)
    
    with open(PREDICTION_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Create header if new file
        if new_file:
            writer.writerow([
                "timestamp", "outdoor_temp", "indoor_temp", "temp_diff", 
                "area", "insulation_rating", "predicted_kwh", "actual_kwh"
            ])
        
        # Calculate features
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temp_diff = indoor_temp - outdoor_temp
        
        # Log the data
        writer.writerow([
            timestamp, outdoor_temp, indoor_temp, temp_diff, 
            area, insulation_rating, predicted_kwh, actual_kwh
        ])

def plot_prediction_accuracy():
    """Visualize prediction accuracy over time"""
    if not os.path.exists(PREDICTION_LOG):
        print("No prediction data yet. Run some predictions first!")
        return
    
    predictions, actuals, timestamps = [], [], []
    
    with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["actual_kwh"] and row["actual_kwh"] != "None":
                try:
                    predictions.append(float(row["predicted_kwh"]))
                    actuals.append(float(row["actual_kwh"]))
                    timestamps.append(row["timestamp"])
                except ValueError:
                    continue
    
    if not predictions:
        print("No actual vs predicted data yet. Enter actual usage in predictions!")
        return
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.7)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual kWh')
    plt.ylabel('Predicted kWh')
    plt.title('Prediction Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    plt.bar(range(len(errors)), errors)
    plt.xlabel('Prediction Number')
    plt.ylabel('Absolute Error (kWh)')
    plt.title('Prediction Errors Over Time')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate accuracy metrics
    avg_error = sum(errors) / len(errors)
    print(f"\nModel Performance:")
    print(f"Average error: {avg_error:.2f} kWh")
    print(f"Total predictions with actual data: {len(predictions)}")

# ---------------------------
# Building Physics Functions
# ---------------------------

def rating_to_uvalue(rating):
    """
    Convert 1-10 insulation rating to realistic U-value (W/m²·K)
    Based on typical building envelope performance:
    - Rating 10 (excellent): 0.15 W/m²·K (passive house standard)
    - Rating 7-8 (good): 0.3-0.5 W/m²·K (modern well-insulated)  
    - Rating 4-6 (average): 0.8-1.2 W/m²·K (standard construction)
    - Rating 1-3 (poor): 1.5-2.0 W/m²·K (old building, single glazing)
    """
    if rating < 1 or rating > 10:
        raise ValueError("Insulation rating must be between 1-10")
    
    # Linear interpolation from poor (2.0) to excellent (0.15)
    return 2.15 - (rating * 0.2)

def calculate_heating_load(area, u_value, temp_diff):
    """
    Calculate heating/cooling load using standard building physics
    Q = U × A × ΔT (watts)
    """
    return u_value * area * abs(temp_diff)

def estimate_system_efficiency(temp_diff, heating_mode):
    """
    Estimate HVAC system efficiency based on operating conditions
    More extreme conditions = lower efficiency
    """
    if heating_mode:
        # Heat pumps become less efficient in extreme cold
        if abs(temp_diff) > 20:
            return 0.65  # 65% efficiency in extreme conditions
        elif abs(temp_diff) > 10:
            return 0.75  # 75% efficiency in moderate conditions
        else:
            return 0.85  # 85% efficiency in mild conditions
    else:
        # Cooling systems (air conditioning)
        if abs(temp_diff) > 15:
            return 0.70  # 70% efficiency in extreme heat
        elif abs(temp_diff) > 8:
            return 0.80  # 80% efficiency in moderate heat
        else:
            return 0.90  # 90% efficiency in mild conditions

# ---------------------------
# Core Prediction Model  
# ---------------------------

def energy_usage_predictor():
    """
    Predict energy usage using proper building physics calculations.
    This demonstrates key AI/ML concepts with engineering accuracy:
    - Feature collection and validation
    - Physics-based modeling  
    - Feature engineering with domain knowledge
    - Model evaluation and accuracy metrics
    """
    
    print("\n=== Energy Usage Predictor ===")
    print("Building Physics + Data Science")
    print("-" * 40)
    
    # Feature Collection with validation
    outdoor_temp = ask_float("Outdoor temperature (°C): ")
    indoor_temp = ask_float("Desired indoor temperature (°C): ")
    area = ask_float("Building floor area (m²): ")
    
    # Get insulation rating with validation
    while True:
        insulation_rating = ask_float("Insulation quality rating (1-10, 10=excellent): ")
        if 1 <= insulation_rating <= 10:
            break
        print("Please enter a rating between 1 and 10")
    
    hours = ask_float("Time period (hours): ")
    
    # Feature Engineering - derive meaningful variables
    temp_diff = indoor_temp - outdoor_temp
    heating_mode = temp_diff > 0
    u_value = rating_to_uvalue(insulation_rating)
    
    print(f"\nBuilding Analysis:")
    print(f"   Temperature difference: {temp_diff:.1f}°C")
    print(f"   Mode: {'Heating' if heating_mode else 'Cooling'}")
    print(f"   Derived U-value: {u_value:.3f} W/m²·K")
    print(f"   Building envelope: {area:.0f}m²")
    
    # Physics-Based Prediction Model
    
    # 1. Calculate thermal load using building physics
    thermal_load_watts = calculate_heating_load(area, u_value, temp_diff)
    
    # 2. Account for system efficiency  
    system_efficiency = estimate_system_efficiency(temp_diff, heating_mode)
    
    # 3. Calculate electrical power requirement
    electrical_power_kw = (thermal_load_watts / 1000) / system_efficiency
    
    # 4. Add base building loads (lighting, equipment, ventilation)
    base_load_kw = area * 0.008  # 8W/m² for base building services
    
    # 5. Total power demand
    total_power_kw = electrical_power_kw + base_load_kw
    
    # 6. Energy consumption over time period
    predicted_kwh = total_power_kw * hours
    
    # Results with engineering context
    print(f"\nPrediction Results:")
    print(f"   Thermal load: {thermal_load_watts:.0f} W ({thermal_load_watts/1000:.2f} kW)")
    print(f"   System efficiency: {system_efficiency*100:.0f}%")
    print(f"   Electrical power needed: {electrical_power_kw:.2f} kW")
    print(f"   Base building load: {base_load_kw:.2f} kW")
    print(f"   Total power demand: {total_power_kw:.2f} kW")
    print(f"   Predicted energy usage: {predicted_kwh:.2f} kWh over {hours:.1f} hours")
    print(f"   Estimated cost (€0.25/kWh): €{predicted_kwh * 0.25:.2f}")
    
    # Model Evaluation
    print(f"\nModel Validation:")
    actual_input = input("   Enter actual usage if known (or press Enter): ").strip()
    
    actual_kwh = None
    if actual_input:
        try:
            actual_kwh = float(actual_input)
            error = abs(predicted_kwh - actual_kwh)
            percentage_error = (error / actual_kwh) * 100
            accuracy = max(0, 100 - percentage_error)
            print(f"   Actual usage: {actual_kwh:.2f} kWh")
            print(f"   Prediction error: {error:.2f} kWh ({percentage_error:.1f}%)")
            print(f"   Model accuracy: {accuracy:.1f}%")
        except ValueError:
            print("   Invalid input - skipping validation")
    
    # Enhanced data logging with physics parameters
    log_prediction_data(outdoor_temp, indoor_temp, area, insulation_rating, 
                       predicted_kwh, actual_kwh)
    
    save_report(f"Prediction: {predicted_kwh:.2f} kWh for {hours:.1f}h, "
               f"U-value={u_value:.3f}, thermal_load={thermal_load_watts:.0f}W, "
               f"efficiency={system_efficiency*100:.0f}%")
    
    print(f"   Data logged for ML model training!")
    
    return {
        'predicted_kwh': predicted_kwh,
        'thermal_load_watts': thermal_load_watts,
        'u_value': u_value,
        'system_efficiency': system_efficiency
    }

# ---------------------------
# Main Program
# ---------------------------

def main():
    """Main program loop"""
    while True:
        print("\nEnergy Usage Predictor - AI/ML with Building Physics")
        print("1. Make energy prediction")
        print("2. View prediction accuracy")  
        print("3. View prediction history")
        print("4. Exit")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == "1":
            energy_usage_predictor()
        elif choice == "2":
            plot_prediction_accuracy()
        elif choice == "3":
            if os.path.exists("prediction_report.txt"):
                print("\n--- Prediction History ---")
                with open("prediction_report.txt", "r") as f:
                    print(f.read())
                print("--- End ---")
            else:
                print("No prediction history yet!")
        elif choice == "4":
            print("Thanks for exploring AI/ML with building physics!")
            break
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()