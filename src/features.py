import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

@st.cache_data # type: ignore
def engineer_features(df):
    """Engineer features for anomaly detection."""
    from sklearn.preprocessing import StandardScaler  # Import inside function
    
    df_copy = df.copy()
    
    # 1. Date features
    if 'date_of_service' in df_copy.columns:
        df_copy['date_of_service'] = pd.to_datetime(df_copy['date_of_service'])
        df_copy['year'] = df_copy['date_of_service'].dt.year
        df_copy['month'] = df_copy['date_of_service'].dt.month
        df_copy['day_of_week'] = df_copy['date_of_service'].dt.dayofweek
        df_copy['day_of_month'] = df_copy['date_of_service'].dt.day
        df_copy['quarter'] = df_copy['date_of_service'].dt.quarter
    
    # 2. Frequency encoding
    if 'provider_npi' in df_copy.columns:
        provider_freq = df_copy['provider_npi'].value_counts(normalize=True)
        df_copy['provider_freq'] = df_copy['provider_npi'].map(provider_freq)
    
    if 'cpt_procedure_code' in df_copy.columns:
        procedure_freq = df_copy['cpt_procedure_code'].value_counts(normalize=True)
        df_copy['procedure_freq'] = df_copy['cpt_procedure_code'].map(procedure_freq)
    
    # 3. Provider aggregates
    provider_agg = df_copy.groupby('provider_npi').agg({
        'billed_amount': ['mean', 'std', 'count']
    }).reset_index()
    provider_agg.columns = ['provider_npi', 'provider_mean', 'provider_std', 'provider_volume']
    df_copy = df_copy.merge(provider_agg, on='provider_npi', how='left')
    
    # 4. Rolling averages (30 days)
    if 'date_of_service' in df_copy.columns:
        df_copy = df_copy.sort_values(['provider_npi', 'date_of_service'])
        df_copy['rolling_mean_30d'] = df_copy.groupby('provider_npi')['billed_amount'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        df_copy['rolling_std_30d'] = df_copy.groupby('provider_npi')['billed_amount'].transform(
            lambda x: x.rolling(window=30, min_periods=1).std()
        )
    
    # 5. Patient features
    if 'patient_id' in df_copy.columns and 'date_of_service' in df_copy.columns:
        df_copy = df_copy.sort_values(['patient_id', 'date_of_service'])
        df_copy['days_since_last_claim'] = df_copy.groupby('patient_id')['date_of_service'].transform(
            lambda x: x.diff().dt.days
        )
    
    # 6. Select features
    feature_cols = [
        'billed_amount',
        'provider_freq',
        'procedure_freq',
        'provider_mean',
        'provider_std',
        'provider_volume',
        'rolling_mean_30d',
        'rolling_std_30d'
    ]
    
    # Add date features if available
    if 'year' in df_copy.columns:
        feature_cols.extend(['year', 'month', 'day_of_week', 'day_of_month', 'quarter'])
    
    if 'days_since_last_claim' in df_copy.columns:
        feature_cols.append('days_since_last_claim')
    
    # 7. Create feature matrix
    df_features = df_copy[feature_cols].copy()
    
    # 8. Handle missing values
    df_features = df_features.fillna(df_features.mean())
    
    # 9. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    return X_scaled, df_copy, scaler, feature_cols
