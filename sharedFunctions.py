import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve
from matplotlib_venn import venn2, venn3
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn . over_sampling import SMOTE

# Print unique values for every object column
def print_unique_values_for_object_columns(df):
    columns = df.select_dtypes(include=['object']).columns
    # Display unique values of each categorical column
    for column in columns:
        values = df[column].unique()
        print(f"{column} : {values}")

def encode_objectdtypes_columns(df, label_encode_columns):
    for column in df.columns:
        # converting all to lowercase when it is string data type
        unique_values_lower = set(x.lower() if isinstance(x, str) else x for x in df[column].unique())
        # print(unique_values_lower)
        # Apply mapping for the specific column
        if unique_values_lower == {"yes", "no"}:
            df[column] = df[column].str.lower().map({"yes": 1, "no": 0})
        elif unique_values_lower == {"year 1", "year 2", "year 3", "year 4"}:
            df[column] = df[column].str.lower().map({"year 1": 1, "year 2": 2, "year 3": 3, "year 4": 4})
        elif unique_values_lower == {"long", "short", "medium"}:
            df[column] = df[column].str.lower().map({"short": 0, "medium": 1, "long": 2})
        elif unique_values_lower == {"average", "high", "low"}:
            df[column] = df[column].str.lower().map({"low": 0, "average": 1, "high": 2})
        elif unique_values_lower == {"min", "mild", "mod", "modsev", "sev"}:
            df[column] = df[column].str.lower().map({"min": 0, "mild": 1, "mod": 2, "modsev": 3, "sev": 4})

    # Label encoding columns using LabelEncoder()
    label_encoder = LabelEncoder()
    for column in label_encode_columns:
        df[column] = label_encoder.fit_transform(df[column])

    return df

def plot_comparison_student(df, x, hue, title, xlabel = None, ylabel = None):
    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = hue
    
    plt.figure(figsize=(10,10))
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x=df[x], hue=df[hue], data=df)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    plt.show()
    return

def plot_categorical_columns(df, columns):
    categorical_columns = columns

    # Check if specified columns exist in the DataFrame
    missing_columns = set(categorical_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
        
    # Check if the list of categorical columns is empty
    if not categorical_columns:
        raise ValueError("List of categorical columns is empty.")
        
    # Calculate the number of rows and columns dynamically
    num_cols = 2
    num_rows = math.ceil(len(categorical_columns) / num_cols)

    # Set up subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 2 * num_rows))
    fig.subplots_adjust(hspace=1, bottom=0.1)  # Adjust bottom space

    # Loop through each categorical column and create bar plots
    for i, column in enumerate(categorical_columns):
        row, col = divmod(i, num_cols)
        sns.countplot(x=column, data=df, ax=axes[row, col])
        axes[row, col].set_title(f'Distribution of {column}')

    # If the last subplot is empty, remove it
    if len(categorical_columns) % num_cols != 0:
        empty_subplots = num_cols - (len(categorical_columns) % num_cols)
        for i in range(1, empty_subplots + 1):
            fig.delaxes(axes[-1, -i])

    # Display the plots
    plt.show()

def plot_venn(subsets, set_labels, set_colors, title="Venn Diagram", alpha=0.9):
    num_subsets = len(subsets)

    if num_subsets == 2:
        venn_function = venn2
    elif num_subsets == 3:
        venn_function = venn3
    else:
        raise ValueError("The function supports only 2 or 3 subsets.")

    venn_function(subsets=subsets, set_labels=set_labels, set_colors=set_colors, alpha=alpha)
    plt.title(title, fontsize=16)
    plt.show()

def plot_stacking_bars(df, x, y, title = 'Title leh Haloooo'):
    # Pivot the data to get counts for each combination A and B
    course_depression_counts = df.pivot_table(index=x, columns=y, aggfunc='size', fill_value=0)

    # Plot the stacked bar chart
    course_depression_counts.plot(kind='bar', stacked=True, figsize=(15, 15))
    plt.title(title)
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    plt.legend(title=y, bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.show()

# Plot to check is data is balanced
def class_distribution(df, target_column):
    class_count = pd.Series(df[target_column]).value_counts()

    plt.figure(figsize=(8, 6))
    ax_after = class_count.plot(kind='bar', color='lightgreen')
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')

    ax_after.set_xticks([])
    ax_after.set_xticklabels([])

    ax_after.set_yticks([])
    ax_after.set_yticklabels([])

    # Annotate each bar with its count
    for i, count in enumerate(class_count):
        ax_after.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Split the data, return X_train, X_test, X_valid, y_train, y_test, y_valid
def split_data(df, target_column):
    # Split the data into train, test, and validation sets
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid=train_test_split (X_train, y_train, test_size=0.25, random_state=42)

    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

# Handle imbalanced data, return X_res_df as resampled data
def SMOTE_resample(df, target_column):
    # Separate the features and target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=101)
    X_res, y_res = sm.fit_resample(X, y)

    # Combine the resampled data into a DataFrame
    X_res_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_column)], axis=1)
    
    # Shuffle the data
    X_res_df = X_res_df.sample(frac=1, random_state=101).reset_index(drop=True)
    return X_res_df

# Function to plot the ROC curve for classifiers
def plot_all_roc_curves(predictions, dataset_key, prediction_type):
    plt.figure(figsize=(10, 8))

    # Unpack only the needed elements from each tuple in the dictionary
    for model_info in predictions[dataset_key][prediction_type]:
        model_name = model_info[0]
        y_pred = model_info[2]
        y_true = model_info[3]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {prediction_type}')
    plt.legend(loc="lower right")
    plt.show()

# Function to plot the comparison for a specific dataset and mental health condition
def plot_comparison(predictions, dataset, condition):
    models = [entry[0] for entry in predictions[dataset][condition]]
    accuracies = [entry[4] for entry in predictions[dataset][condition]]

    # Define different colors for each bar
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=colors, alpha=0.6)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlabel('Models', fontsize=14)
    ax.set_title(f'Model Comparison for {condition} in Dataset {dataset}', fontsize=16)
    plt.ylim([0, 1.2])
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)

    # Display the accuracy values on top of the bars
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=12)

    # Display model names outside the bars
    for bar, model in zip(bars, models):
        height = bar.get_height()

    # Add a legend for the colors
    ax.legend(bars, models, title='Models', title_fontsize='12', fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()

# Function to plot confusion matrices for a specific dataset and mental health condition
def plot_confusion_matrices(predictions, dataset, condition):
    models = [entry[0] for entry in predictions[dataset][condition]]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

    for i, model in enumerate(models):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        if i < len(predictions[dataset][condition]):  # Check if there is a corresponding model
            y_pred = predictions[dataset][condition][i][2]# Extracting y_pred from predictions
            y_true = predictions[dataset][condition][i][3]  # Extracting y_true from predictions

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Plot confusion matrix using seaborn heatmap
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
            ax.set_title(f'Confusion Matrix for {model} - {condition}')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_feature_importance(predictions, dataset, condition, df):
    # Get the list of models
    selected_classifiers = ['RF', 'AdaBoost', 'XGBoost', 'CatBoost']
    
    # Get the list of models
    models = [entry[0] for entry in predictions[dataset][condition]]

    # Filter out only the selected classifiers
    selected_models = [model for model in models if model in selected_classifiers]

    if dataset == 2:
        num_rows = len(selected_models)
        num_cols = 1
        figsize = (12, num_rows * 5)  # Adjust the figure size accordingly
    else:
        num_rows = len(selected_models) // 2 + len(selected_models) % 2
        num_cols = 2
        figsize = (12, num_rows * 5)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

    for i, model in enumerate(selected_models):
        if dataset == 2:
            ax = axes[i]
        else:
            row = i // 2
            col = i % 2
            ax = axes[row, col]

        # Find the index of the model in the original list
        index = models.index(model)

        # Get the classifier from the predictions
        classifier = predictions[dataset][condition][index][1]

        # Check if the classifier has feature_importances_ attribute
        if hasattr(classifier, 'feature_importances_'):
            # Extract feature importances
            feature_importances = classifier.feature_importances_

            # Get the feature names from DataFrame
            feature_names = df.columns

            # Plot feature importance
            ax.bar(range(len(feature_importances)), feature_importances, alpha=0.5, color='orange')
            ax.set_title(f'Feature Importance for {model} - {condition}')
            ax.set_ylabel('Feature Importance', color='orange')
            ax.tick_params('y', colors='orange')
            ax.set_xticks(range(len(feature_importances)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right')  # Use feature names for x-axis labels
        else:
            ax.axis('off')  # Turn off axes for classifiers without feature_importances_

    plt.tight_layout()
    plt.show()

# Plot deep learning model history
def plot_history(history):
    """
    Plots the training and validation loss from a Keras model's history.
    
    Parameters:
        history: A Keras History object returned from the fit method.
        
    Returns:
        None
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()