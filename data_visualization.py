import matplotlib.pyplot as plt
import seaborn as sns

# Define the data visualization function for reuse
def data_visualization(df):
    # ========== Age Distribution ========== #
    df["Age"].plot(kind="hist", bins=20)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Age Distribution")
    plt.show()

    df["Age"].plot(kind="box")
    plt.title("Age Boxplot")
    plt.show()

    # ========== Gender Distribution ========== #
    df["Gender"].value_counts().plot(kind="bar")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Gender Distribution")
    plt.show()

    # Ratio of men to women
    print("Gender Ratio:")
    print(df["Gender"].value_counts(normalize=True))

    # ========== Years at Company Distribution ========== #
    df["Years at Company"].plot(kind="hist", bins=20)
    plt.xlabel("Years at Company")
    plt.ylabel("Frequency")
    plt.title("Years at Company Distribution")
    plt.show()

    df["Years at Company"].plot(kind="box")
    plt.title("Years at Company Boxplot")
    plt.show()

    # ========== Job Role Distribution ========== #
    df["Job Role"].value_counts().plot(kind="bar")
    plt.xlabel("Job Role")
    plt.ylabel("Count")
    plt.title("Job Role Distribution")
    plt.show()

    # ========== Monthly Income Distribution ========== #
    df["Monthly Income"].plot(kind="hist", bins=20)
    plt.xlabel("Monthly Income")
    plt.ylabel("Frequency")
    plt.title("Monthly Income Distribution")
    plt.show()

    df["Monthly Income"].plot(kind="box")
    plt.title("Monthly Income Boxplot")
    plt.show()

    # ========== Work Life Balance Distribution ========== #
    df["Work-Life Balance"].value_counts().plot(kind="bar")
    plt.xlabel("Work-Life Balance")
    plt.ylabel("Count")
    plt.title("Work-Life Balance Distribution")
    plt.show()

    # ========== Job Satisfaction Distribution ========== #
    df["Job Satisfaction"].value_counts().plot(kind="bar")
    plt.xlabel("Job Satisfaction")
    plt.ylabel("Count")
    plt.title("Job Satisfaction Distribution")
    plt.show()

    # ========== Performance Rating Distribution ========== #
    df["Performance Rating"].value_counts().plot(kind="bar")
    plt.xlabel("Performance Rating")
    plt.ylabel("Count")
    plt.title("Performance Rating Distribution")
    plt.show()

    # ========== Number of Promotions Distribution ========== #
    df["Number of Promotions"].value_counts().plot(kind="bar")
    plt.xlabel("Number of Promotions")
    plt.ylabel("Count")
    plt.title("Number of Promotions Distribution")
    plt.show()

    df["Number of Promotions"].plot(kind="box")
    plt.title("Number of Promotions Boxplot")
    plt.show()

    # ========== Overtime Distribution ========== #
    df["Overtime"].value_counts().plot(kind="bar")
    plt.xlabel("Overtime")
    plt.ylabel("Count")
    plt.title("Overtime Distribution")
    plt.show()

    # ========== Distance from Home Distribution ========== #
    df["Distance from Home"].plot(kind="hist", bins=20)
    plt.xlabel("Distance from Home")
    plt.ylabel("Frequency")
    plt.title("Distance from Home Distribution")
    plt.show()

    df["Distance from Home"].plot(kind="box")
    plt.title("Distance from Home Boxplot")
    plt.show()

    # ========== Education Distribution ========== #
    df["Education Level"].value_counts().plot(kind="bar")
    plt.xlabel("Education Level")
    plt.ylabel("Count")
    plt.title("Education Level Distribution")
    plt.show()

    # ========== Marital Status Distribution ========== #
    df["Marital Status"].value_counts().plot(kind="bar")
    plt.xlabel("Marital Status")
    plt.ylabel("Count")
    plt.title("Marital Status Distribution")
    plt.show()

    # ========== Number of Dependents Distribution ========== #
    df["Number of Dependents"].value_counts().plot(kind="bar")
    plt.xlabel("Number of Dependents")
    plt.ylabel("Count")
    plt.title("Number of Dependents Distribution")
    plt.show()

    df["Number of Dependents"].plot(kind="box")
    plt.title("Number of Dependents Boxplot")
    plt.show()

    # ========== Job Level Distribution ========== #
    df["Job Level"].value_counts().plot(kind="bar")
    plt.xlabel("Job Level")
    plt.ylabel("Count")
    plt.title("Job Level Distribution")
    plt.show()

    # ========== Company Size Distribution ========== #
    df["Company Size"].value_counts().plot(kind="bar")
    plt.xlabel("Company Size")
    plt.ylabel("Count")
    plt.title("Company Size Distribution")
    plt.show()

    # ========== Company Tenure Distribution ========== #
    df["Company Tenure"].plot(kind="hist", bins=20)
    plt.xlabel("Company Tenure")
    plt.ylabel("Frequency")
    plt.title("Company Tenure Distribution")
    plt.show()

    df["Company Tenure"].plot(kind="box")
    plt.title("Company Tenure Boxplot")
    plt.show()

    # ========== Remote Work Distribution ========== #
    df["Remote Work"].value_counts().plot(kind="bar")
    plt.xlabel("Remote Work")
    plt.ylabel("Count")
    plt.title("Remote Work Distribution")
    plt.show()

    # ========== Leadership Opportunities Distribution ========== #
    df["Leadership Opportunities"].value_counts().plot(kind="bar")
    plt.xlabel("Leadership Opportunities")
    plt.ylabel("Count")
    plt.title("Leadership Opportunities Distribution")
    plt.show()

    # ========== Innovation Opportunities Distribution ========== #
    df["Innovation Opportunities"].value_counts().plot(kind="bar")
    plt.xlabel("Innovation Opportunities")
    plt.ylabel("Count")
    plt.title("Innovation Opportunities Distribution")
    plt.show()

    # ========== Company Reputation Distribution ========== #
    df["Company Reputation"].value_counts().plot(kind="bar")
    plt.xlabel("Company Reputation")
    plt.ylabel("Count")
    plt.title("Company Reputation Distribution")
    plt.show()

    # ========== Employee Recognition Distribution ========== #
    df["Employee Recognition"].value_counts().plot(kind="bar")
    plt.xlabel("Employee Recognition")
    plt.ylabel("Count")
    plt.title("Employee Recognition Distribution")
    plt.show()

    # ========== Attrition Distribution ========== #
    df["Attrition"].value_counts().plot(kind="bar")
    plt.xlabel("Attrition")
    plt.ylabel("Count")
    plt.title("Attrition Distribution")
    plt.show()

    # ========== Pairplot of All Features ========== #
    import seaborn as sns
    sns.pairplot(df)
    plt.suptitle("Pairplot of All Features", y=1.02)
    plt.show()

    # ========== Correlation Heatmap ========== #
    numeric_df = df.select_dtypes(include=["number"])

    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
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

    # ========== 4. Top correlated feature pairs (optional insight) ========== #
    corr_unstacked = corr.unstack()
    corr_unstacked = corr_unstacked[corr_unstacked != 1.0]
    top_corr = corr_unstacked.abs().sort_values(ascending=False).head(10)

    print("\nTop Correlated Feature Pairs:")
    print(top_corr)

    # ========== 5. Missing values check (useful post-preprocessing sanity check) ========== #
    print("\nMissing values per column:")
    print(df.isnull().sum().sort_values(ascending=False))