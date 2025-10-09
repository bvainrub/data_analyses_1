# CSV Data Analysis
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Step 1: Loading and Exploring Data
# ==================================

def load_and_explore_csv(file_path):
    """Load CSV file and perform initial exploration"""
    
    print("📊 STEP 1: LOADING DATA")
    print("=" * 50)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\nColumn names and data types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    print("\nDataset info:")
    print(df.info())
    
    return df

# Step 2: Data Quality Assessment
# ===============================

def assess_data_quality(df):
    """Check for missing values, duplicates, and data quality issues"""
    
    print("\n📋 STEP 2: DATA QUALITY ASSESSMENT")
    print("=" * 50)
    
    # Check for missing values
    print("Missing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    if missing_values.sum() == 0:
        print("✅ No missing values found!")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")
    
    if duplicate_count == 0:
        print("✅ No duplicate rows found!")
    
    # Check data types and potential issues
    print("\nData type analysis:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype} - Unique values: {df[col].nunique()}")
    
    return df

# Step 3: Basic Statistics and Summaries
# ======================================

def generate_basic_statistics(df):
    """Generate descriptive statistics for the dataset"""
    
    print("\n📈 STEP 3: BASIC STATISTICS")
    print("=" * 50)
    
    # Numerical columns statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print("Numerical columns summary:")
    print(df[numerical_cols].describe())
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    print(f"\nCategorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        print(f"\n{col} - Value counts:")
        print(df[col].value_counts().head())
    
    return numerical_cols, categorical_cols

# Step 4: Data Filtering and Selection
# ====================================

def demonstrate_filtering(df):
    """Show various ways to filter and select data"""
    
    print("\n🔍 STEP 4: DATA FILTERING AND SELECTION")
    print("=" * 50)
    
    # Select specific columns
    print("Selecting specific columns (name and age):")
    basic_info = df[['first_name', 'last_name', 'age']].head()
    print(basic_info)
    
    # Filter rows based on conditions
    print("\nFiltering: Customers older than 50:")
    older_customers = df[df['age'] > 50]
    print(f"Found {len(older_customers)} customers older than 50")
    print(older_customers[['first_name', 'last_name', 'age']].head())
    
    # Multiple conditions
    print("\nFiltering: Active customers with high spending (>$2000):")
    high_value_active = df[(df['is_active'] == True) & (df['total_spent'] > 2000)]
    print(f"Found {len(high_value_active)} high-value active customers")
    
    # Filter by text/string conditions
    print("\nFiltering: Customers from IT department:")
    it_customers = df[df['department'] == 'IT']
    print(f"Found {len(it_customers)} IT department customers")
    
    return older_customers, high_value_active, it_customers

# Step 5: Data Grouping and Aggregation
# =====================================

def demonstrate_grouping(df):
    """Show how to group data and perform aggregations"""
    
    print("\n📊 STEP 5: DATA GROUPING AND AGGREGATION")
    print("=" * 50)
    
    # Group by single column
    print("Average spending by customer segment:")
    segment_spending = df.groupby('customer_segment')['total_spent'].mean().round(2)
    print(segment_spending)
    
    # Group by multiple columns
    print("\nAverage salary by department and customer segment:")
    dept_segment_salary = df.groupby(['department', 'customer_segment'])['salary'].mean().round(2)
    print(dept_segment_salary.head(10))
    
    # Multiple aggregations
    print("\nMultiple statistics by department:")
    dept_stats = df.groupby('department').agg({
        'salary': ['mean', 'median', 'min', 'max'],
        'age': 'mean',
        'total_spent': 'sum',
        'customer_id': 'count'
    }).round(2)
    print(dept_stats)
    
    return segment_spending, dept_stats

# Step 6: Data Transformation and New Columns
# ===========================================

def demonstrate_transformation(df):
    """Show how to create new columns and transform data"""
    
    print("\n🔄 STEP 6: DATA TRANSFORMATION")
    print("=" * 50)
    
    # Create a copy to avoid modifying original data
    df_transformed = df.copy()
    
    # Create new columns based on existing data
    df_transformed['full_name'] = df_transformed['first_name'] + ' ' + df_transformed['last_name']
    df_transformed['spending_per_order'] = df_transformed['total_spent'] / (df_transformed['orders_count'] + 1)  # +1 to avoid division by zero
    
    # Create age categories
    def categorize_age(age):
        if age < 30:
            return 'Young'
        elif age < 50:
            return 'Middle-aged'
        else:
            return 'Senior'
    
    df_transformed['age_category'] = df_transformed['age'].apply(categorize_age)
    
    # Create binary columns
    df_transformed['high_spender'] = df_transformed['total_spent'] > df_transformed['total_spent'].median()
    
    print("New columns created:")
    new_columns = ['full_name', 'spending_per_order', 'age_category', 'high_spender']
    print(df_transformed[new_columns].head())
    
    return df_transformed

# Step 7: Sorting and Ranking
# ===========================

def demonstrate_sorting(df):
    """Show different ways to sort data"""
    
    print("\n📋 STEP 7: SORTING AND RANKING")
    print("=" * 50)
    
    # Sort by single column
    print("Top 10 highest spenders:")
    top_spenders = df.nlargest(10, 'total_spent')[['first_name', 'last_name', 'total_spent']]
    print(top_spenders)
    
    # Sort by multiple columns
    print("\nSorted by department, then by salary (descending):")
    sorted_data = df.sort_values(['department', 'salary'], ascending=[True, False])
    print(sorted_data[['first_name', 'last_name', 'department', 'salary']].head(10))
    
    return top_spenders

# Step 8: Data Export and Saving
# ==============================

def save_processed_data(df, filename='processed_data.csv'):
    """Save processed data to new CSV file"""
    
    print("\n💾 STEP 8: SAVING PROCESSED DATA")
    print("=" * 50)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Data saved to {filename}")
    
    # Save specific columns only
    summary_filename = 'customer_summary.csv'
    summary_columns = ['customer_id', 'first_name', 'last_name', 'total_spent', 'customer_segment']
    df[summary_columns].to_csv(summary_filename, index=False)
    print(f"✅ Summary data saved to {summary_filename}")

# Main Analysis Pipeline
# =====================

def run_complete_analysis(file_path='sample_customer_data.csv'):
    """Run the complete analysis pipeline"""
    
    print("🚀 STARTING COMPLETE CSV ANALYSIS")
    print("=" * 60)
    
    try:
        # Step 1: Load and explore
        df = load_and_explore_csv(file_path)
        
        # Step 2: Assess data quality
        df = assess_data_quality(df)
        
        # Step 3: Generate statistics
        numerical_cols, categorical_cols = generate_basic_statistics(df)
        
        # Step 4: Demonstrate filtering
        older_customers, high_value_active, it_customers = demonstrate_filtering(df)
        
        # Step 5: Show grouping and aggregation
        segment_spending, dept_stats = demonstrate_grouping(df)
        
        # Step 6: Transform data
        df_transformed = demonstrate_transformation(df)
        
        # Step 7: Sort data
        top_spenders = demonstrate_sorting(df)
        
        # Step 8: Save results
        save_processed_data(df_transformed)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return df, df_transformed
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        print("Please run the CSV generation script first.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Run the complete analysis
    original_df, processed_df = run_complete_analysis()
    
    # Additional examples 
    print("\n🎓 ADDITIONAL  EXAMPLES")
    print("=" * 50)
    
    # Example 1: Working with dates
    if 'registration_date' in original_df.columns:
        print("\nWorking with dates:")
        original_df['registration_date'] = pd.to_datetime(original_df['registration_date'])
        original_df['days_since_registration'] = (pd.Timestamp.now() - original_df['registration_date']).dt.days
        print(original_df[['first_name', 'registration_date', 'days_since_registration']].head())
    
    # Example 2: Finding correlations
    print("\nCorrelation analysis between numerical variables:")
    numerical_data = original_df.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr().round(2)
    print(correlation_matrix['total_spent'].sort_values(ascending=False))
    
