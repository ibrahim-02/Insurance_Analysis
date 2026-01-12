# Actuarial Insigths System for Insurance Product Optimization

Comprehensive Python-based EDA tool for analyzing 105k+ insurance records, covering policy distribution, customer demographics, vehicle risks, claims behavior, and risk segmentation.

## Features

- **Data Processing:** Automated feature engineering (age, driving experience, vehicle age, loss ratios)
- **Policy Analysis:** Distribution channels, risk types, policies in force
- **Demographics:** Age groups, driving experience, customer tenure analysis
- **Vehicle Risks:** Fuel types, vehicle age, power/capacity, vehicle values
- **Claims Behavior:** Frequency, cost, loss ratios, payment methods
- **Risk Segmentation:** Geographic areas, second driver impact, combined risk factors
- **Visualizations:** 12 comprehensive charts (loss ratios, distributions, correlations)

## Technologies

- Python 3.x
- Pandas - Data manipulation
- NumPy - Numerical operations
- Matplotlib & Seaborn - Visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Basic usage
python insurance_eda.py

# In the script, update the file path
analyzer = InsuranceDataAnalyzer('your_data.csv')
analyzer.run_full_analysis()
```

## Data Requirements

CSV file with semicolon separator containing fields:
- Policy info: ID, Distribution_channel, Type_risk, Policies_in_force
- Customer data: Date_birth, Date_driving_licence, Seniority, Second_driver
- Vehicle data: Year_matriculation, Type_fuel, Power, Cylinder_capacity, Value_vehicle
- Claims data: Premium, Cost_claims_year, N_claims_year, N_claims_history
- Dates: Date_start_contract, Date_last_renewal, Date_next_renewal, Date_lapse
- Location: Area




## Key Metrics Analyzed

- Overall Loss Ratio: (Total Claims Cost / Total Premium) × 100
- Claims Frequency: % of policies with claims
- Claim Severity: Average cost per claim
- Risk segmentation by demographics, geography, and vehicle characteristics

## Author

Ibrahim - MS Data Analytics Engineering, Northeastern University
