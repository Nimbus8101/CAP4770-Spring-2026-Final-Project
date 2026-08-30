import matplotlib.pyplot as plt
import seaborn as sns

# Define the data visualization function for reuse
def data_visualization(df):
    # ========== Age Distribution ========== #
    df["Age"].plot(kind="hist", bins=20)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Age Distribution")
    plt.savefig("age_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    df["Age"].plot(kind="box")
    plt.title("Age Boxplot")
    plt.savefig("age_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Gender Distribution ========== #
    df["Gender"].value_counts().plot(kind="bar")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Gender Distribution")
    plt.savefig("gender_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Ratio of men to women
    print("Gender Ratio:")
    print(df["Gender"].value_counts(normalize=True))

    # ========== Years at Company Distribution ========== #
    df["Years at Company"].plot(kind="hist", bins=20)
    plt.xlabel("Years at Company")
    plt.ylabel("Frequency")
    plt.title("Years at Company Distribution")
    plt.savefig("years_at_company_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    df["Years at Company"].plot(kind="box")
    plt.title("Years at Company Boxplot")
    plt.savefig("years_at_company_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Job Role Distribution ========== #
    df["Job Role"].value_counts().plot(kind="bar")
    plt.xlabel("Job Role")
    plt.ylabel("Count")
    plt.title("Job Role Distribution")
    plt.savefig("job_role_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Monthly Income Distribution ========== #
    df["Monthly Income"].plot(kind="hist", bins=20)
    plt.xlabel("Monthly Income")
    plt.ylabel("Frequency")
    plt.title("Monthly Income Distribution")
    plt.savefig("monthly_income_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    df["Monthly Income"].plot(kind="box")
    plt.title("Monthly Income Boxplot")
    plt.savefig("monthly_income_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Work Life Balance Distribution ========== #
    df["Work-Life Balance"].value_counts().plot(kind="bar")
    plt.xlabel("Work-Life Balance")
    plt.ylabel("Count")
    plt.title("Work-Life Balance Distribution")
    plt.savefig("work_life_balance_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Job Satisfaction Distribution ========== #
    df["Job Satisfaction"].value_counts().plot(kind="bar")
    plt.xlabel("Job Satisfaction")
    plt.ylabel("Count")
    plt.title("Job Satisfaction Distribution")
    plt.savefig("job_satisfaction_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Performance Rating Distribution ========== #
    df["Performance Rating"].value_counts().plot(kind="bar")
    plt.xlabel("Performance Rating")
    plt.ylabel("Count")
    plt.title("Performance Rating Distribution")
    plt.savefig("performance_rating_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Number of Promotions Distribution ========== #
    df["Number of Promotions"].value_counts().plot(kind="bar")
    plt.xlabel("Number of Promotions")
    plt.ylabel("Count")
    plt.title("Number of Promotions Distribution")
    plt.savefig("number_of_promotions_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    df["Number of Promotions"].plot(kind="box")
    plt.title("Number of Promotions Boxplot")
    plt.savefig("number_of_promotions_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Overtime Distribution ========== #
    df["Overtime"].value_counts().plot(kind="bar")
    plt.xlabel("Overtime")
    plt.ylabel("Count")
    plt.title("Overtime Distribution")
    plt.savefig("overtime_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Distance from Home Distribution ========== #
    df["Distance from Home"].plot(kind="hist", bins=20)
    plt.xlabel("Distance from Home")
    plt.ylabel("Frequency")
    plt.title("Distance from Home Distribution")
    plt.savefig("distance_from_home_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    df["Distance from Home"].plot(kind="box")
    plt.title("Distance from Home Boxplot")
    plt.savefig("distance_from_home_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Education Distribution ========== #
    df["Education Level"].value_counts().plot(kind="bar")
    plt.xlabel("Education Level")
    plt.ylabel("Count")
    plt.title("Education Level Distribution")
    plt.savefig("education_level_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Marital Status Distribution ========== #
    df["Marital Status"].value_counts().plot(kind="bar")
    plt.xlabel("Marital Status")
    plt.ylabel("Count")
    plt.title("Marital Status Distribution")
    plt.savefig("marital_status_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Number of Dependents Distribution ========== #
    df["Number of Dependents"].value_counts().plot(kind="bar")
    plt.xlabel("Number of Dependents")
    plt.ylabel("Count")
    plt.title("Number of Dependents Distribution")
    plt.savefig("number_of_dependents_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    df["Number of Dependents"].plot(kind="box")
    plt.title("Number of Dependents Boxplot")
    plt.savefig("number_of_dependents_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Job Level Distribution ========== #
    df["Job Level"].value_counts().plot(kind="bar")
    plt.xlabel("Job Level")
    plt.ylabel("Count")
    plt.title("Job Level Distribution")
    plt.savefig("job_level_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Company Size Distribution ========== #
    df["Company Size"].value_counts().plot(kind="bar")
    plt.xlabel("Company Size")
    plt.ylabel("Count")
    plt.title("Company Size Distribution")
    plt.savefig("company_size_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Company Tenure Distribution ========== #
    df["Company Tenure"].plot(kind="hist", bins=20)
    plt.xlabel("Company Tenure")
    plt.ylabel("Frequency")
    plt.title("Company Tenure Distribution")
    plt.savefig("company_tenure_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    df["Company Tenure"].plot(kind="box")
    plt.title("Company Tenure Boxplot")
    plt.savefig("company_tenure_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Remote Work Distribution ========== #
    df["Remote Work"].value_counts().plot(kind="bar")
    plt.xlabel("Remote Work")
    plt.ylabel("Count")
    plt.title("Remote Work Distribution")
    plt.savefig("remote_work_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Leadership Opportunities Distribution ========== #
    df["Leadership Opportunities"].value_counts().plot(kind="bar")
    plt.xlabel("Leadership Opportunities")
    plt.ylabel("Count")
    plt.title("Leadership Opportunities Distribution")
    plt.savefig("leadership_opportunities_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Innovation Opportunities Distribution ========== #
    df["Innovation Opportunities"].value_counts().plot(kind="bar")
    plt.xlabel("Innovation Opportunities")
    plt.ylabel("Count")
    plt.title("Innovation Opportunities Distribution")
    plt.savefig("innovation_opportunities_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Company Reputation Distribution ========== #
    df["Company Reputation"].value_counts().plot(kind="bar")
    plt.xlabel("Company Reputation")
    plt.ylabel("Count")
    plt.title("Company Reputation Distribution")
    plt.savefig("company_reputation_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Employee Recognition Distribution ========== #
    df["Employee Recognition"].value_counts().plot(kind="bar")
    plt.xlabel("Employee Recognition")
    plt.ylabel("Count")
    plt.title("Employee Recognition Distribution")
    plt.savefig("employee_recognition_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Attrition Distribution ========== #
    df["Attrition"].value_counts().plot(kind="bar")
    plt.xlabel("Attrition")
    plt.ylabel("Count")
    plt.title("Attrition Distribution")
    plt.savefig("attrition_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Pairplot of All Features ========== #
    import seaborn as sns
    sns.pairplot(df)
    plt.suptitle("Pairplot of All Features", y=1.02)
    plt.savefig("pairplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ========== Correlation Heatmap ========== #
    numeric_df = df.select_dtypes(include=["number"])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()



def data_visualization_preprocessed(df):
    """
    Visualization pipeline for preprocessed data:
    - numeric-only (standardized + encoded features)
    """

    numeric_df = df.select_dtypes(include=["number"])

    # ========== 1. Summary Boxplots (Outliers) ========== #
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=numeric_df)
    plt.xticks(rotation=90)
    plt.title("Boxplot of All Features (Outlier Detection)")
    plt.show()

    # ========== 2. Histograms for distributions ========== #
    numeric_df.hist(bins=30, figsize=(16, 10))
    plt.suptitle("Feature Distributions", y=1.02)
    plt.show()

    # ========== 3. Correlation Heatmap ========== #
    plt.figure(figsize=(12, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.show()

    # ========== 4. Top correlated feature pairs ========== #
    corr_unstacked = corr.unstack()
    corr_unstacked = corr_unstacked[corr_unstacked != 1.0]
    top_corr = corr_unstacked.abs().sort_values(ascending=False).head(10)

    print("\nTop Correlated Feature Pairs:")
    print(top_corr)

    # ========== 5. Missing values check ========== #
    print("\nMissing values per column:")
    print(df.isnull().sum().sort_values(ascending=False))