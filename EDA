"""
Insurance Data Exploratory Data Analysis
Author: Mohammed
Description: Comprehensive analysis of 105k+ insurance records covering policy distribution,
             customer demographics, vehicle risks, claims behavior, and risk segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class InsuranceDataAnalyzer:
    """
    A comprehensive insurance data analyzer for EDA and risk assessment
    """
    
    def __init__(self, filepath):
        """Initialize with data loading"""
        print("Loading insurance data...")
        self.df = pd.read_csv(filepath, sep=';', encoding='utf-8')
        print(f"Data loaded successfully: {self.df.shape[0]:,} records, {self.df.shape[1]} columns\n")
        self.prepare_data()
    
    def prepare_data(self):
        """Data preprocessing and feature engineering"""
        print("Preparing and engineering features...")
        
        # Convert date columns
        date_cols = ['Date_start_contract', 'Date_last_renewal', 'Date_next_renewal', 
                     'Date_birth', 'Date_driving_licence', 'Date_lapse']
        
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], format='%d/%m/%Y', errors='coerce')
        
        # Calculate age from date of birth
        self.df['Age'] = (pd.to_datetime('2019-01-01') - self.df['Date_birth']).dt.days / 365.25
        self.df['Age'] = self.df['Age'].round(0)
        
        # Calculate driving experience
        self.df['Driving_experience'] = (pd.to_datetime('2019-01-01') - self.df['Date_driving_licence']).dt.days / 365.25
        self.df['Driving_experience'] = self.df['Driving_experience'].round(0)
        
        # Calculate vehicle age
        self.df['Vehicle_age'] = 2019 - self.df['Year_matriculation']
        
        # Calculate loss ratio per record
        self.df['Loss_ratio'] = np.where(self.df['Premium'] > 0, 
                                         (self.df['Cost_claims_year'] / self.df['Premium']) * 100, 
                                         0)
        
        # Create age groups
        self.df['Age_group'] = pd.cut(self.df['Age'], 
                                       bins=[0, 25, 35, 45, 55, 65, 100],
                                       labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
        
        # Create vehicle age groups
        self.df['Vehicle_age_group'] = pd.cut(self.df['Vehicle_age'],
                                               bins=[-1, 3, 7, 12, 100],
                                               labels=['0-3 years', '4-7 years', '8-12 years', '13+ years'])
        
        # Clean Distribution_channel (handle date values)
        self.df['Distribution_channel'] = self.df['Distribution_channel'].apply(
            lambda x: 0 if isinstance(x, str) else x
        )
        
        print("Feature engineering completed.\n")
    
    def basic_statistics(self):
        """Display basic dataset statistics"""
        print("="*70)
        print("BASIC DATASET STATISTICS")
        print("="*70)
        
        print(f"Total Records: {self.df.shape[0]:,}")
        print(f"Total Unique Customers: {self.df['ID'].nunique():,}")
        print(f"Date Range: {self.df['Date_last_renewal'].min()} to {self.df['Date_last_renewal'].max()}")
        print(f"\nMissing Values by Column:")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            for col, count in missing.items():
                print(f"  {col}: {count:,} ({count/len(self.df)*100:.2f}%)")
        else:
            print("  No missing values found")
        print()
    
    def policy_distribution_analysis(self):
        """Analyze policy distribution patterns"""
        print("="*70)
        print("POLICY DISTRIBUTION ANALYSIS")
        print("="*70)
        
        # Distribution channel analysis
        print("\n1. Distribution by Channel:")
        channel_dist = self.df.groupby('Distribution_channel').agg({
            'ID': 'count',
            'Premium': 'sum',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).rename(columns={'ID': 'Count'})
        channel_dist['Loss_Ratio_%'] = (channel_dist['Cost_claims_year'] / channel_dist['Premium'] * 100).round(2)
        channel_dist['Avg_Premium'] = (channel_dist['Premium'] / channel_dist['Count']).round(2)
        print(channel_dist)
        
        # Type of risk analysis
        print("\n2. Distribution by Risk Type:")
        risk_dist = self.df.groupby('Type_risk').agg({
            'ID': 'count',
            'Premium': 'sum',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).rename(columns={'ID': 'Count'})
        risk_dist['Loss_Ratio_%'] = (risk_dist['Cost_claims_year'] / risk_dist['Premium'] * 100).round(2)
        risk_dist['Claims_Frequency_%'] = (risk_dist['N_claims_year'] / risk_dist['Count'] * 100).round(2)
        print(risk_dist)
        
        # Policies in force distribution
        print("\n3. Policies in Force Distribution:")
        policies_dist = self.df['Policies_in_force'].value_counts().sort_index()
        print(policies_dist)
        print()
    
    def customer_demographics_analysis(self):
        """Analyze customer demographic patterns"""
        print("="*70)
        print("CUSTOMER DEMOGRAPHICS ANALYSIS")
        print("="*70)
        
        # Age analysis
        print("\n1. Customer Age Statistics:")
        print(self.df['Age'].describe().round(2))
        
        print("\n2. Analysis by Age Group:")
        age_analysis = self.df.groupby('Age_group').agg({
            'ID': 'count',
            'Premium': ['sum', 'mean'],
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).round(2)
        age_analysis.columns = ['Count', 'Total_Premium', 'Avg_Premium', 'Total_Claims_Cost', 'N_Claims']
        age_analysis['Loss_Ratio_%'] = (age_analysis['Total_Claims_Cost'] / age_analysis['Total_Premium'] * 100).round(2)
        print(age_analysis)
        
        # Driving experience
        print("\n3. Driving Experience Statistics:")
        print(self.df['Driving_experience'].describe().round(2))
        
        # Seniority (customer tenure)
        print("\n4. Customer Seniority Distribution:")
        seniority_stats = self.df.groupby('Seniority').agg({
            'ID': 'count',
            'Lapse': 'sum'
        }).rename(columns={'ID': 'Count', 'Lapse': 'Total_Lapsed'})
        seniority_stats['Lapse_Rate_%'] = (seniority_stats['Total_Lapsed'] / seniority_stats['Count'] * 100).round(2)
        print(seniority_stats.head(10))
        print()
    
    def vehicle_risk_analysis(self):
        """Analyze vehicle-related risk patterns"""
        print("="*70)
        print("VEHICLE-RELATED RISK ANALYSIS")
        print("="*70)
        
        # Fuel type analysis
        print("\n1. Analysis by Fuel Type:")
        fuel_analysis = self.df.groupby('Type_fuel').agg({
            'ID': 'count',
            'Premium': 'sum',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum',
            'Value_vehicle': 'mean'
        }).round(2)
        fuel_analysis.columns = ['Count', 'Total_Premium', 'Total_Claims', 'N_Claims', 'Avg_Vehicle_Value']
        fuel_analysis['Loss_Ratio_%'] = (fuel_analysis['Total_Claims'] / fuel_analysis['Total_Premium'] * 100).round(2)
        print(fuel_analysis)
        
        # Vehicle age analysis
        print("\n2. Analysis by Vehicle Age:")
        vehicle_age_analysis = self.df.groupby('Vehicle_age_group').agg({
            'ID': 'count',
            'Premium': 'mean',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).round(2)
        vehicle_age_analysis.columns = ['Count', 'Avg_Premium', 'Total_Claims_Cost', 'N_Claims']
        print(vehicle_age_analysis)
        
        # Power and cylinder capacity analysis
        print("\n3. Vehicle Power Statistics:")
        print(self.df['Power'].describe().round(2))
        
        print("\n4. Cylinder Capacity Statistics:")
        print(self.df['Cylinder_capacity'].describe().round(2))
        
        # Vehicle value by risk type
        print("\n5. Average Vehicle Value by Risk Type:")
        vehicle_value_risk = self.df.groupby('Type_risk')['Value_vehicle'].mean().round(2)
        print(vehicle_value_risk)
        print()
    
    def claims_behavior_analysis(self):
        """Analyze claims frequency, cost, and patterns"""
        print("="*70)
        print("CLAIMS BEHAVIOR ANALYSIS")
        print("="*70)
        
        # Overall claims statistics
        total_premium = self.df['Premium'].sum()
        total_claims_cost = self.df['Cost_claims_year'].sum()
        total_claims = self.df['N_claims_year'].sum()
        records_with_claims = (self.df['N_claims_year'] > 0).sum()
        
        print("\n1. Overall Claims Statistics:")
        print(f"Total Premium: €{total_premium:,.2f}")
        print(f"Total Claims Cost: €{total_claims_cost:,.2f}")
        print(f"Overall Loss Ratio: {(total_claims_cost/total_premium*100):.2f}%")
        print(f"Total Number of Claims: {total_claims:,}")
        print(f"Records with Claims: {records_with_claims:,} ({records_with_claims/len(self.df)*100:.2f}%)")
        print(f"Average Claims Cost per Claim: €{total_claims_cost/total_claims:,.2f}" if total_claims > 0 else "N/A")
        
        # Claims distribution
        print("\n2. Claims Distribution:")
        claims_dist = self.df['N_claims_year'].value_counts().sort_index()
        print(claims_dist.head(10))
        
        # Claims by payment method
        print("\n3. Claims Analysis by Payment Method:")
        payment_analysis = self.df.groupby('Payment').agg({
            'ID': 'count',
            'Premium': 'sum',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).round(2)
        payment_analysis.columns = ['Count', 'Total_Premium', 'Total_Claims_Cost', 'N_Claims']
        payment_analysis['Loss_Ratio_%'] = (payment_analysis['Total_Claims_Cost'] / payment_analysis['Total_Premium'] * 100).round(2)
        print(payment_analysis)
        
        # Historical claims analysis
        print("\n4. Historical Claims Pattern:")
        print(f"Average Historical Claims per Policy: {self.df['N_claims_history'].mean():.2f}")
        print(f"Max Historical Claims: {self.df['N_claims_history'].max()}")
        print(f"Records with Historical Claims: {(self.df['N_claims_history'] > 0).sum():,}")
        print()
    
    def risk_segmentation_analysis(self):
        """Comprehensive risk segmentation across multiple dimensions"""
        print("="*70)
        print("RISK SEGMENTATION ANALYSIS")
        print("="*70)
        
        # 1. Risk segmentation by Area
        print("\n1. Risk Segmentation by Geographic Area:")
        area_analysis = self.df.groupby('Area').agg({
            'ID': 'count',
            'Premium': 'sum',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).round(2)
        area_analysis.columns = ['Count', 'Total_Premium', 'Total_Claims_Cost', 'N_Claims']
        area_analysis['Loss_Ratio_%'] = (area_analysis['Total_Claims_Cost'] / area_analysis['Total_Premium'] * 100).round(2)
        area_analysis['Claims_Frequency_%'] = (area_analysis['N_Claims'] / area_analysis['Count'] * 100).round(2)
        area_analysis['Avg_Claim_Severity'] = (area_analysis['Total_Claims_Cost'] / area_analysis['N_Claims']).round(2)
        area_analysis = area_analysis.sort_values('Loss_Ratio_%', ascending=False)
        print(area_analysis)
        
        # 2. Risk segmentation by Second Driver
        print("\n2. Risk Segmentation by Second Driver Presence:")
        second_driver_analysis = self.df.groupby('Second_driver').agg({
            'ID': 'count',
            'Premium': 'sum',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).round(2)
        second_driver_analysis.columns = ['Count', 'Total_Premium', 'Total_Claims_Cost', 'N_Claims']
        second_driver_analysis['Loss_Ratio_%'] = (second_driver_analysis['Total_Claims_Cost'] / second_driver_analysis['Total_Premium'] * 100).round(2)
        second_driver_analysis['Claims_Frequency_%'] = (second_driver_analysis['N_Claims'] / second_driver_analysis['Count'] * 100).round(2)
        second_driver_analysis['Avg_Claim_Severity'] = (second_driver_analysis['Total_Claims_Cost'] / second_driver_analysis['N_Claims']).round(2)
        second_driver_analysis.index = ['No Second Driver', 'Has Second Driver']
        print(second_driver_analysis)
        
        # 3. Combined segmentation: Risk Type x Second Driver
        print("\n3. Combined Risk Segmentation (Risk Type x Second Driver):")
        combined_analysis = self.df.groupby(['Type_risk', 'Second_driver']).agg({
            'ID': 'count',
            'Premium': 'sum',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).round(2)
        combined_analysis.columns = ['Count', 'Total_Premium', 'Total_Claims_Cost', 'N_Claims']
        combined_analysis['Loss_Ratio_%'] = (combined_analysis['Total_Claims_Cost'] / combined_analysis['Total_Premium'] * 100).round(2)
        combined_analysis['Claims_Frequency_%'] = (combined_analysis['N_Claims'] / combined_analysis['Count'] * 100).round(2)
        print(combined_analysis)
        
        # 4. Combined segmentation: Area x Risk Type (top areas)
        print("\n4. Combined Risk Segmentation (Top Areas x Risk Type):")
        top_areas = self.df['Area'].value_counts().head(2).index
        area_risk_analysis = self.df[self.df['Area'].isin(top_areas)].groupby(['Area', 'Type_risk']).agg({
            'ID': 'count',
            'Premium': 'sum',
            'Cost_claims_year': 'sum',
            'N_claims_year': 'sum'
        }).round(2)
        area_risk_analysis.columns = ['Count', 'Total_Premium', 'Total_Claims_Cost', 'N_Claims']
        area_risk_analysis['Loss_Ratio_%'] = (area_risk_analysis['Total_Claims_Cost'] / area_risk_analysis['Total_Premium'] * 100).round(2)
        print(area_risk_analysis)
        print()
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Loss Ratio by Risk Type
        ax1 = plt.subplot(4, 3, 1)
        risk_lr = self.df.groupby('Type_risk').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        ).sort_values(ascending=False)
        risk_lr.plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Loss Ratio by Risk Type', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Risk Type')
        ax1.set_ylabel('Loss Ratio (%)')
        ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Threshold')
        ax1.legend()
        plt.xticks(rotation=0)
        
        # 2. Claims Distribution
        ax2 = plt.subplot(4, 3, 2)
        claims_dist = self.df['N_claims_year'].value_counts().sort_index()
        claims_dist.head(10).plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Claims Frequency Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Claims')
        ax2.set_ylabel('Frequency')
        plt.xticks(rotation=0)
        
        # 3. Loss Ratio by Second Driver
        ax3 = plt.subplot(4, 3, 3)
        second_driver_lr = self.df.groupby('Second_driver').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        )
        second_driver_lr.index = ['No Second Driver', 'Has Second Driver']
        second_driver_lr.plot(kind='bar', ax=ax3, color=['green', 'orange'])
        ax3.set_title('Loss Ratio by Second Driver', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Second Driver Status')
        ax3.set_ylabel('Loss Ratio (%)')
        plt.xticks(rotation=45)
        
        # 4. Age Distribution
        ax4 = plt.subplot(4, 3, 4)
        self.df['Age'].hist(bins=30, ax=ax4, color='skyblue', edgecolor='black')
        ax4.set_title('Customer Age Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Age')
        ax4.set_ylabel('Frequency')
        ax4.axvline(self.df['Age'].median(), color='r', linestyle='--', label=f'Median: {self.df["Age"].median():.0f}')
        ax4.legend()
        
        # 5. Loss Ratio by Age Group
        ax5 = plt.subplot(4, 3, 5)
        age_lr = self.df.groupby('Age_group').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        )
        age_lr.plot(kind='bar', ax=ax5, color='teal')
        ax5.set_title('Loss Ratio by Age Group', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Age Group')
        ax5.set_ylabel('Loss Ratio (%)')
        plt.xticks(rotation=45)
        
        # 6. Loss Ratio by Area
        ax6 = plt.subplot(4, 3, 6)
        area_lr = self.df.groupby('Area').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        ).sort_values(ascending=False)
        area_lr.plot(kind='bar', ax=ax6, color='purple')
        ax6.set_title('Loss Ratio by Geographic Area', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Area')
        ax6.set_ylabel('Loss Ratio (%)')
        plt.xticks(rotation=0)
        
        # 7. Vehicle Age Distribution
        ax7 = plt.subplot(4, 3, 7)
        self.df['Vehicle_age'].hist(bins=30, ax=ax7, color='lightcoral', edgecolor='black')
        ax7.set_title('Vehicle Age Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Vehicle Age (years)')
        ax7.set_ylabel('Frequency')
        
        # 8. Loss Ratio by Fuel Type
        ax8 = plt.subplot(4, 3, 8)
        fuel_lr = self.df.groupby('Type_fuel').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        ).sort_values(ascending=False)
        fuel_lr.plot(kind='bar', ax=ax8, color='salmon')
        ax8.set_title('Loss Ratio by Fuel Type', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Fuel Type')
        ax8.set_ylabel('Loss Ratio (%)')
        plt.xticks(rotation=0)
        
        # 9. Premium Distribution (log scale)
        ax9 = plt.subplot(4, 3, 9)
        self.df['Premium'].hist(bins=50, ax=ax9, color='gold', edgecolor='black')
        ax9.set_title('Premium Distribution', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Premium (€)')
        ax9.set_ylabel('Frequency')
        ax9.set_yscale('log')
        
        # 10. Claims Cost Distribution (log scale)
        ax10 = plt.subplot(4, 3, 10)
        claims_with_cost = self.df[self.df['Cost_claims_year'] > 0]['Cost_claims_year']
        claims_with_cost.hist(bins=50, ax=ax10, color='tomato', edgecolor='black')
        ax10.set_title('Claims Cost Distribution (Claims > 0)', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Claims Cost (€)')
        ax10.set_ylabel('Frequency')
        ax10.set_yscale('log')
        
        # 11. Distribution Channel Performance
        ax11 = plt.subplot(4, 3, 11)
        channel_analysis = self.df.groupby('Distribution_channel').agg({
            'ID': 'count',
            'Premium': 'sum'
        })
        channel_analysis['ID'].plot(kind='bar', ax=ax11, color='mediumpurple')
        ax11.set_title('Policy Count by Distribution Channel', fontsize=12, fontweight='bold')
        ax11.set_xlabel('Distribution Channel')
        ax11.set_ylabel('Number of Policies')
        plt.xticks(rotation=0)
        
        # 12. Correlation Heatmap (Numerical Features)
        ax12 = plt.subplot(4, 3, 12)
        numeric_cols = ['Age', 'Driving_experience', 'Premium', 'Cost_claims_year', 
                       'N_claims_year', 'Vehicle_age', 'Power', 'Value_vehicle', 'Loss_ratio']
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax12, cbar_kws={'shrink': 0.8})
        ax12.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('insurance_eda_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'insurance_eda_analysis.png'")
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("EXECUTIVE SUMMARY REPORT")
        print("="*70)
        
        total_records = len(self.df)
        unique_customers = self.df['ID'].nunique()
        total_premium = self.df['Premium'].sum()
        total_claims_cost = self.df['Cost_claims_year'].sum()
        overall_loss_ratio = (total_claims_cost / total_premium * 100)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   • Total Records: {total_records:,}")
        print(f"   • Unique Customers: {unique_customers:,}")
        print(f"   • Average Policies per Customer: {total_records/unique_customers:.2f}")
        
        print(f"\n💰 Financial Metrics:")
        print(f"   • Total Premium: €{total_premium:,.2f}")
        print(f"   • Total Claims Cost: €{total_claims_cost:,.2f}")
        print(f"   • Overall Loss Ratio: {overall_loss_ratio:.2f}%")
        print(f"   • Average Premium: €{self.df['Premium'].mean():.2f}")
        
        print(f"\n📈 Claims Statistics:")
        print(f"   • Total Claims: {self.df['N_claims_year'].sum():,}")
        print(f"   • Claims Frequency: {(self.df['N_claims_year'] > 0).sum()/len(self.df)*100:.2f}%")
        print(f"   • Average Claims Cost: €{self.df[self.df['Cost_claims_year']>0]['Cost_claims_year'].mean():.2f}")
        
        # Key Insights
        print(f"\n🔍 Key Risk Insights:")
        
        # Highest risk type
        risk_lr = self.df.groupby('Type_risk').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        )
        highest_risk = risk_lr.idxmax()
        print(f"   • Highest Risk Product: Type {highest_risk} ({risk_lr[highest_risk]:.2f}% loss ratio)")
        
        # Second driver impact
        sd_lr = self.df.groupby('Second_driver').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        )
        sd_impact = sd_lr[1] - sd_lr[0]
        print(f"   • Second Driver Impact: +{sd_impact:.2f}% higher loss ratio")
        
        # Highest risk area
        area_lr = self.df.groupby('Area').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        )
        highest_area = area_lr.idxmax()
        print(f"   • Highest Risk Area: Area {highest_area} ({area_lr[highest_area]:.2f}% loss ratio)")
        
        # Fuel type performance
        fuel_lr = self.df.groupby('Type_fuel').apply(
            lambda x: (x['Cost_claims_year'].sum() / x['Premium'].sum() * 100)
        )
        best_fuel = fuel_lr.idxmin()
        print(f"   • Best Performing Fuel: {best_fuel} ({fuel_lr[best_fuel]:.2f}% loss ratio)")
        
        print("\n" + "="*70)
        print("Analysis completed successfully!")
        print("="*70 + "\n")
    
    def run_full_analysis(self):
        """Execute complete analysis pipeline"""
        print("\n" + "🚀 "*20)
        print("INSURANCE DATA COMPREHENSIVE ANALYSIS")
        print("🚀 "*20 + "\n")
        
        self.basic_statistics()
        self.policy_distribution_analysis()
        self.customer_demographics_analysis()
        self.vehicle_risk_analysis()
        self.claims_behavior_analysis()
        self.risk_segmentation_analysis()
        self.generate_summary_report()
        
        # Generate visualizations
        generate_viz = input("\nGenerate visualizations? (y/n): ").lower()
        if generate_viz == 'y':
            self.generate_visualizations()


# Main execution
if __name__ == "__main__":
    # Initialize analyzer with your data file
    analyzer = InsuranceDataAnalyzer('insurancedata.csv')
    
    # Run complete analysis
    analyzer.run_full_analysis()
    
    # Optional: Access specific analyses
    # analyzer.policy_distribution_analysis()
    # analyzer.risk_segmentation_analysis()
