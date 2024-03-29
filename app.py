import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import beta

def plot_beta_distributions(alpha1, beta1, alpha2, beta2):
    x = np.linspace(0, 1, 1000)
    y1 = beta.pdf(x, alpha1, beta1)
    y2 = beta.pdf(x, alpha2, beta2)

    fig, ax = plt.subplots(figsize=(10, 6))  # Create a Matplotlib figure and axis
    ax.plot(x, y1, label=f'Beta({alpha1}, {beta1})', color='blue')
    ax.plot(x, y2, label=f'Beta({alpha2}, {beta2})', color='red')

    ax.set_title('Beta Distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True)

    return fig

def main():
    st.title("Bayesian A/B Test")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        st.sidebar.header("Configurations")
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Sidebar inputs
        sample_columns = st.sidebar.selectbox("Select sample column for analysis", df.columns, index=None)
        population_columns = st.sidebar.selectbox("Select population column for analysis", df.columns, index=None)
        filter_column = st.sidebar.selectbox("Select a column that discriminate the control and treatment variant", df.columns)
        control_value = st.sidebar.selectbox(f"Select a value that indicates control variant {filter_column}", df[filter_column].unique(), index=None)
        treatment_value = st.sidebar.selectbox(f"Select a value that indicates treatment variant {filter_column}", df[filter_column].unique(), index=None)
        additional_filter_column = st.sidebar.selectbox(f"Select a column to filter (if any)", df.columns)
        additional_filter_value = st.sidebar.multiselect(f"Select a value to filter in {additional_filter_column}", df[additional_filter_column].unique())

        # Bayesian A/B Test
        trials = st.sidebar.slider("Select number of trials (for monte carlo simulation):", min_value=0, max_value=int(1e6), value=int(1e6))
        prior_alpha = st.sidebar.number_input("Select value for prior alpha (if unknown, please select 1):", min_value=0, value=1)
        prior_beta = st.sidebar.number_input("Select value for prior beta (if unknown, please select 1):", min_value=0, value=1)
        
        if all(items is not None for items in [sample_columns, population_columns, filter_column, control_value, treatment_value]):
            if additional_filter_value:
                print(additional_filter_value)
                df = df[df[additional_filter_column].isin(additional_filter_value)]
            else:
                df = df
            df_control = df[(df[filter_column]==control_value)]
            df_treatment = df[(df[filter_column]==treatment_value)]
            control_alpha = int(df_control[sample_columns].sum() + prior_alpha)
            control_beta = int((df_control[population_columns].sum() - df_control[sample_columns].sum()) + prior_beta)
            treatment_alpha = int(df_treatment[sample_columns].sum() + prior_alpha)
            treatment_beta = int((df_treatment[population_columns].sum() - df_treatment[sample_columns].sum()) + prior_beta)
            control_samples = np.random.beta(control_alpha, 
                                            control_beta, 
                                            size=int(trials))
            treatment_samples = np.random.beta(treatment_alpha, 
                                            treatment_beta, 
                                            size=int(trials))
            confidence_level = sum(treatment_samples>control_samples)/trials
            if confidence_level<0.4:
                confidence_level = 1 - confidence_level

            # Write Metrics
            col1, col2 = st.columns(2)
            # Display metrics in the first column
            with col1:
                st.header("Control")
                st.metric(f"{sample_columns}", df_control[sample_columns].sum())
                st.metric(f"{population_columns}", df_control[population_columns].sum())
                st.metric(f"Percentage {sample_columns} on Control", str(round((df_control[sample_columns].sum())/(df_control[population_columns].sum())*100,2)) + '%')

            # Display metrics in the second column
            with col2:
                st.header("Treatment")
                st.metric(f"{sample_columns}", df_treatment[sample_columns].sum())
                st.metric(f"{population_columns}", df_treatment[population_columns].sum())
                st.metric(f"Percentage {sample_columns} on Treatment", str(round((df_treatment[sample_columns].sum())/(df_treatment[population_columns].sum())*100,2)) + '%')

            # Plot PDF distribution
            st.write("Probability Density Function on Control and Treatment")
            st.pyplot(plot_beta_distributions(control_alpha, control_beta, treatment_alpha, treatment_beta))

            # Show filtered dataframe
            st.metric(label="Confidence level that the treatment and control is statistically significant:", 
                    value=str(round(confidence_level*100, 2))+'%')

if __name__ == "__main__":
    main()