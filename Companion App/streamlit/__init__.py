import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

def glob_concat(path, file_str):

    # Find the files in the folders
    files = glob.glob(os.path.join(path, file_str))

    # Print the files for verification when running the function
    display(files)

    # Combining all the files into a DataFrame
    df_files = [pd.read_sas(file) for file in files]
    combined_df = pd.concat(df_files)

    # Setting the index of the new DataFrame
    combined_df.SEQN = combined_df.SEQN.astype('int64')
    combined_df.set_index('SEQN', verify_integrity=True, inplace=True)
    return combined_df


def val_counts(df):

    for col in df.columns:
        print(f'{col} value counts', '\n')
        display(df[col].value_counts(dropna=False))
        print('--------------------------------------')


def cols_tokeep(df, col_list):

    df_copy = df.copy()
    for col in df_copy.columns:
        if col not in col_list:
            df_copy.drop(columns=[col], inplace=True)
        else:
            pass
    return df_copy


def first_cancer_count(x):

    if x['first_cancer_type'] != 'None':
        return 1
    else:
        return 0


def second_cancer_count(x):

    if x['second_cancer_type'] != 'None':
        return 1
    else:
        return 0


def third_cancer_count(x):

    if x['third_cancer_type'] != 'None':
        return 1
    else:
        return 0


def plotting_counts(df, col, target='depression'):

    # Sort the column values for plotting
    order_list = list(df[col].unique())
    order_list.sort()

    # Plot the figure
    fig, ax = plt.subplots(figsize=(16,8))
    x, y = col, target
    ax = sns.countplot(x=x, hue=y, data=df, order=order_list)

    # Set labels and title
    plt.title(f'{col.title()} By Count {target.title()}',
              fontdict={'fontsize': 30})
    plt.xlabel(f'{col.title()}', fontdict={'fontsize': 20})
    plt.ylabel(f'{target.title()} Count', fontdict={'fontsize': 20})
    plt.xticks(rotation=75)
    return fig, ax


def plotting_percentages(df, col, target='depression'):

    x, y = col, target

    # Temporary dataframe with percentage values
    temp_df = df.groupby(x)[y].value_counts(normalize=True)
    temp_df = temp_df.mul(100).rename('percent').reset_index()

    # Sort the column values for plotting
    order_list = list(df[col].unique())
    order_list.sort()

    # Plot the figure
    sns.set(font_scale=1.5)
    g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=temp_df,
                    height=8, aspect=2, order=order_list, legend_out=False)
    g.ax.set_ylim(0,100)

    # Loop through each bar in the graph and add the percentage value
    for p in g.ax.patches:
        txt = str(p.get_height().round(1)) + '%'
        txt_x = p.get_x()
        txt_y = p.get_height()
        g.ax.text(txt_x,txt_y,txt)

    # Set labels and title
    plt.title(f'{col.title()} By Percent {target.title()}',
              fontdict={'fontsize': 30})
    plt.xlabel(f'{col.title()}', fontdict={'fontsize': 20})
    plt.ylabel(f'{target.title()} Percentage', fontdict={'fontsize': 20})
    plt.xticks(rotation=75)
    return g


def plot_num_cols(df, col, target='depression'):

    # Generating the figure
    g = sns.catplot(x=target, y=col, data=df, kind='boxen',
                    height=7, aspect=2)

    # Setting the title
    plt.suptitle(f'{col.title()} and {target.title()}', fontsize=30, y=1.05)


def make_classification_report(model, y_true, x_test, title=''):

    # Generate predictions
    y_preds = model.predict(x_test)
    print('__________________________________________________________________')
    print(f'CLASSIFICATION REPORT FOR: \n\t{title}')
    print('__________________________________________________________________')
    print('\n')

    # Generate report
    report = metrics.classification_report(y_true, y_preds,
                                           target_names=['not depressed', 'depressed'])
    report_dict = metrics.classification_report(y_true, y_preds,
                                                output_dict=True,
                                                target_names=['not depressed', 'depressed'])

    # Add the title to the report dictionary
    report_dict['title'] = title
    print(report)
    print('__________________________________________________________________')

    return report_dict


def plot_roc_curve(model, xtest, ytest, title=''):

    # Creating the plot
    fig, ax = plt.subplots(figsize=(8,6), ncols=1)
    roc_plot = metrics.plot_roc_curve(model, xtest, ytest, ax=ax)

    # Setting the title of the plot
    ax.set_title(f'ROC Curve For {title}',
                 fontdict={'fontsize':17})

    # Setting a legend for the plot
    ax.legend()
    plt.show();

    return fig


def plot_top_features(model, xtrain, title=''):

    # Turn the feature importances into a series
    importances = pd.Series(model.feature_importances_, index=xtrain.columns)

    # Plot the top most important features
    importances.nlargest(20).sort_values().plot(kind='barh')
    plt.title(f'Most Important Features For {title}', fontdict={'fontsize':17})
    plt.xlabel('Importance')
    return importances.sort_values(ascending=False)


def evaluate_model(model, xtrain, xtest, ytest, tree=False, title=''):

    make_classification_report(model, ytest, xtest, title=title)
    plot_confusion_matrix(model, xtest, ytest, title=title)
    plot_roc_curve(model, xtest, ytest, title=title)