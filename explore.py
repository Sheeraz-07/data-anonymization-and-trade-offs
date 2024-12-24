
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(new_dataset, cleaned_dataset):
    # # Display first 10 rows of both datasets
    # st.write("### New Dataset (First 10 Rows)")
    # st.dataframe(new_dataset.head(10))

    # st.write("### Cleaned Dataset (First 10 Rows)")
    # st.dataframe(cleaned_dataset.head(10))

    # Plot mean salary by Job Title
    st.write("### Mean Salary by Job Title")
    job_salary = new_dataset.groupby('Job_Title')['Salary'].mean().reset_index()
    job_salary = job_salary.sort_values(by='Salary', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Salary', y='Job_Title', data=job_salary, ax=ax)
    st.pyplot(fig)

    # Plot number of people by Job Title
    st.write("### Number of People by Job Title")
    job_count = new_dataset['Job_Title'].value_counts().reset_index()
    job_count.columns = ['Job_Title', 'Count']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Job_Title', data=job_count, ax=ax)
    st.pyplot(fig)

    # Plot Experience vs Salary
    st.write("### Experience vs Salary")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Experience', y='Salary', data=new_dataset, ax=ax)
    st.pyplot(fig)
    # #plot Distribution by Location
    # st.write("### Distribution by Location")
    # fig, ax = plt.subplots(figsize=(10,6))
    # sns.
    # Plot Salary comparison by Location
    st.write("### Salary Comparison by Location")
    location_salary = new_dataset.groupby('Location')['Salary'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Salary', y='Location', data=location_salary, ax=ax)
    st.pyplot(fig)
