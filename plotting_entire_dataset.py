import pandas as pd
import matplotlib.pyplot as plt

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def plot_data(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    num_numerical_columns = len(numerical_columns)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, column1 in enumerate(numerical_columns):
        for j, column2 in enumerate(numerical_columns):
            if i != j:
                ax.scatter(df[column1], df[column2], alpha=0.5, label=f"{column1} vs {column2}")

    ax.set_xlabel('Columns')
    ax.set_ylabel('Values')
    ax.set_title('Scatter Plot of All Columns')
    ax.legend()
    plt.show()

def main():
    file_path = "facebook.csv" 
    df = load_dataset(file_path)
    plot_data(df)

if __name__ == "__main__":
    main()
