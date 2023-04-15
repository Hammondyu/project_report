import csv
import json
import json
import statistics
from scipy.stats import ttest_ind_from_stats
from statistics import mean, stdev
import re
from nltk.tokenize import word_tokenize
from matplotlib import rcParams
from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import os
import nltk
import django
import chardet
import jieba
import jieba.posseg as pseg
import opencc
import zhconv
from scipy.stats import shapiro, skew, kurtosis
import numpy as np
from tabulate import tabulate
from scipy.stats import describe
import openpyxl
from scipy.stats import mannwhitneyu,rankdata, friedmanchisquare, wilcoxon, kruskal
from collections import defaultdict
import scipy.stats as stats
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.anova import AnovaRM
from pandas import DataFrame
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# nltk.download('averaged_perceptron_tagger')
from nltk.probability import FreqDist
from textblob import TextBlob
from snownlp import SnowNLP
import nltk.sentiment.vader as vader

# nltk.download('vader_lexicon')
############## Read files into json ####################

# # Open the CSV file and read its contents
with open('corpus_named.csv', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    data = [row for row in csv_reader]

# Remove the 'Name_CN' column
for row in data:
    del row['Name_CN']

# Convert the data to JSON format
json_data = json.dumps(data, ensure_ascii=False, indent=4)

# Write the JSON data to a file
with open('corpus_unidentified.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

############## Read files into json ####################


############## Plot the investment compare #############
#
with open('corpus_prepared.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Convert the 'Identity' column to numeric
df['Identity'] = pd.to_numeric(df['Identity'], errors='coerce')

# Group the data by the 'Site' column
groups = df.groupby('Site')


def welch_ttest(group1, group2):
    for col in ['Ideology', 'Capital_Economic_Capital', 'Capital_Cultural_Capital', 'Capital_Social_Capital',
                'Capital_Symbolic_Capital', 'Identity']:
        # Convert the column to numeric
        group1_col = pd.to_numeric(group1[col], errors='coerce')
        group2_col = pd.to_numeric(group2[col], errors='coerce')

        # Calculate the mean and standard deviation of each group
        group1_mean = group1_col.mean()
        group2_mean = group2_col.mean()
        group1_std = group1_col.std()
        group2_std = group2_col.std()

        # Calculate the Welch's t-test statistic and p-value
        t, p = ttest_ind(group1_col, group2_col, equal_var=False)

        # Calculate the effect size (Cohen's d)
        pooled_std = ((len(group1_col) - 1) * group1_std ** 2 + (len(group2_col) - 1) * group2_std ** 2) / (
                len(group1_col) + len(group2_col) - 2)
        cohen_d = abs(group1_mean - group2_mean) / pooled_std

        # Print the results
        print(f"{col}: t = {t}, p = {p}, Cohen's d = {cohen_d}")


epst_group = groups.get_group('EPST')
emis_group = groups.get_group('EMIS')

welch_ttest(epst_group, emis_group)

# Set plot style
sns.set_style("whitegrid")
sns.set_palette("Set2")
rcParams['figure.figsize'] = 15, 10
plt.rcParams['font.family'] = 'Times New Roman'


def create_comparison_plot(df, x_col, y_col, title, ax):
    sns.boxplot(x=x_col, y=y_col, data=df, width=0.5, ax=ax, showfliers=True)
    add_stat_annotation(ax=ax, x=x_col, y=y_col, data=df, box_pairs=[('EPST', 'EMIS')],
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2, )
    ax.set_title(title)
    plt.grid(False)
    ax.set_xlabel('Site')
    ax.set_ylabel(y_col)
    ax.grid(False)
    sns.despine()


# Create figure and axes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)

handles = []
labels = []

# Create plots for each column
for i, col in enumerate(['Ideology', 'Capital_Economic_Capital', 'Capital_Cultural_Capital', 'Capital_Social_Capital',
                         'Capital_Symbolic_Capital', 'Identity']):
    # Convert the column to numeric
    epst_group_col = pd.to_numeric(epst_group[col], errors='coerce')
    emis_group_col = pd.to_numeric(emis_group[col], errors='coerce')

    # Drop NaN values
    epst_group_col = epst_group_col.dropna()
    emis_group_col = emis_group_col.dropna()

    # Combine the data into a single DataFrame
    df = pd.DataFrame({'Site': ['EPST'] * len(epst_group_col) + ['EMIS'] * len(emis_group_col),
                       col: list(epst_group_col) + list(emis_group_col)})

    # Create the subplot and plot
    ax = axes[int(i / 3)][i % 3]
    create_comparison_plot(df, 'Site', col, col, ax)

    # Get handles and labels for the legend
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Add legend outside of the subplots
# fig.legend(handles, labels, loc='lower center', ncol=2)

# Set font to Times New Roman for all text elements in the figure
plt.rc('font', family='Times New Roman')

# Save the figure
plt.savefig('investment_panel.png', dpi=600, bbox_inches='tight')

############## Plot the investment compare #############
# # Define the variables to be plotted
variables = ['Capital_Economic_Capital', 'Capital_Cultural_Capital', 'Capital_Social_Capital', 'Capital_Symbolic_Capital', 'Identity', 'Ideology']

# Define empty lists to store the values for each group
epst_values = [[] for _ in range(len(variables))]
emis_values = [[] for _ in range(len(variables))]


# Read data from file
with open('corpus_prepared.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Loop through the data and separate the values by group
for item in data:
    if item['Site'] == 'EPST':
        for i, var in enumerate(variables):
            epst_values[i].append(item[var])
    elif item['Site'] == 'EMIS':
        for i, var in enumerate(variables):
            emis_values[i].append(item[var])

epst_means = [np.mean(v) for v in epst_values]
emis_means = [np.mean(v) for v in emis_values]

# Plot the values in a 1x2 panel
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the values for the EPST group
axes[0].bar(range(len(variables)), [sorted(v, reverse=True)[0] for v in epst_values], color='blue', alpha=0.5, label='Highest')
axes[0].bar(range(len(variables)), [sorted(v, reverse=True)[1] for v in epst_values], color='blue', alpha=0.25, label='Second highest')
axes[0].bar(range(len(variables)), [sorted(v, reverse=True)[2] for v in epst_values], color='blue', alpha=0.1, label='Third highest')
axes[0].bar(range(len(variables)), epst_means, color='black', alpha=1, label='Mean')
axes[0].set_xticks(range(len(variables)))
axes[0].set_xticklabels(variables, rotation=90)
axes[0].set_title('EPST Group')

# Plot the values for the EMIS group
axes[1].bar(range(len(variables)), [sorted(v, reverse=True)[0] for v in emis_values], color='red', alpha=0.5, label='Highest')
axes[1].bar(range(len(variables)), [sorted(v, reverse=True)[1] for v in emis_values], color='red', alpha=0.25, label='Second highest')
axes[1].bar(range(len(variables)), [sorted(v, reverse=True)[2] for v in emis_values], color='red', alpha=0.1, label='Third highest')
axes[1].bar(range(len(variables)), emis_means, color='black', alpha=1, label='Mean')
axes[1].set_xticks(range(len(variables)))
axes[1].set_xticklabels(variables, rotation=90)
axes[1].set_title('EMIS Group')

# Set the legend and show the plot
# axes[0].legend()
# axes[1].legend()
plt.show()

################### Descriptive Data ###################

# # Modify the values of the specified keys from string to integer
for obj in data:
    obj["Statement_Language(s)"] = obj.pop("Stament_Language(s)")
    obj["NO."] = int(obj["NO."])
    obj["Statement_Language(s)"] = int(obj["Statement_Language(s)"])
    obj["if_codemix"] = int(obj["if_codemix"])
    obj["Ideology"] = int(obj["Ideology"])
    obj["Capital_Economic_Capital"] = int(obj["Capital_Economic_Capital"])
    obj["Capital_Cultural_Capital"] = int(obj["Capital_Cultural_Capital"])
    obj["Capital_Social_Capital"] = int(obj["Capital_Social_Capital"])
    obj["Capital_Symbolic_Capital"] = int(obj["Capital_Symbolic_Capital"])
    obj["Identity"] = int(obj["Identity"])
    obj["All_Investments"] = int(obj["All_Investments"])
    del obj["Source"]

# Write the modified data to a new file
with open('corpus_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

############## change to integer ###########

############## delete emoji ################
# Open the input JSON file for reading
with open('corpus_cleaned.json', 'r', encoding='utf-8') as infile:

    # Load the JSON data into a Python object
    data = json.load(infile)

# Loop over each dictionary in the JSON data
for item in data:

    # Extract the value of the "Statement_CN" key from the dictionary
    text = item.get('Statement_CN')

    # Remove all emoji from the text using regular expressions
    text = re.sub('[^\u0000-\uFFFF]+', '', text)

    # Store the modified text back into the dictionary
    item['Statement_CN'] = text

# Open the output JSON file for writing
with open('corpus_cleaned_no_emoji.json', 'w', encoding='utf-8') as outfile:

    # Write the modified JSON data to the output file
    json.dump(data, outfile, ensure_ascii=False, indent=4)


## CN_descriptive ###
Load the JSON file
Load the corpus data from the JSON file
with open('corpus_simplified.json', encoding='utf-8') as f:
    data1 = json.load(f)
    print('loaded')

Create a new list to store the modified data
new_data = []

# Process each item in the corpus data
for item in data1:
    if item['Statement_Language(s)'] == 1 or item['Statement_Language(s)'] == 4:
        # Get the word count of the content in Statement_CN_simplified
        content = item['Statement_CN']
        words = pseg.lcut(content)
        content_tokenized = []
        for word, pos_tag in words:
            content_tokenized.append(f"{word}/{pos_tag}")
        word_count = len(content_tokenized)
        # Get the type-token ratio of the content in Statement_CN
        tokens = set([word.split('/')[0] for word in content_tokenized])
        ttr = len(tokens) / word_count
        # Get the standardized type-token ratio of the content in Statement_CN
        sttr = ttr / ((2 * (word_count - 1)) / word_count)
        # Add the results to the item dictionary
        item['Word_Count_CN'] = word_count
        item['TTR_CN'] = ttr
        item['STTR_CN'] = sttr
        item['Content_Tokenized_CN'] = content_tokenized
    # Add the modified item dictionary to the new data list
    new_data.append(item)
    print(f'item {item["NO."]} processed')

# Save the modified data to a new JSON file
with open('corpus_CN_tokenized.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
    print('saved')

### CN_descriptive ###

# ## ENG_descriptive ###
# Open the JSON file for reading
# Load the file
with open('corpus_CN_tokenized.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# loop through all the items
for item in corpus:
    # check if Statement_Language(s) is 1 or 4
    if item['Statement_Language(s)'] == 2 or item['Statement_Language(s)'] == 4:
        # tokenize the content using NLTK's word_tokenize
        content_tokens = word_tokenize(item['Statement_ENG'])
        # POS tag the tokens using NLTK's pos_tag
        pos_tags = nltk.pos_tag(content_tokens)
        # join the word and POS tag and store in a new variable
        item['Content_Tokenized_ENG'] = ' '.join([f"{word}/{tag}" for word, tag in pos_tags])
        # calculate word count
        word_count = len(content_tokens)
        # calculate type-token ratio
        fdist = FreqDist(content_tokens)
        TTR = len(fdist) / word_count
        # calculate standardized type-token ratio
        STTR = len(fdist) / ((2 * word_count) - 1)
        # store the values in the item
        item['word_count_ENG'] = word_count
        item['TTR_ENG'] = TTR
        item['STTR_ENG'] = STTR
        print(f'item {item["NO."]} processed')

# save the modified corpus to a new file
with open('corpus_CN_ENG_tokenized.json', 'w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=4)

### modified_signgle line ###
with open('corpus_CN_ENG_tokenized.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# modify the value under the key 'Content_Tokenized_CN'

    for item in data:
        if item['Statement_Language(s)'] == 1 or item['Statement_Language(s)'] == 4:

            print(item["Content_Tokenized_CN"])
            item["Content_Tokenized_CN"] = ' '.join(item["Content_Tokenized_CN"])
            del item["Statement_CN_simplified"]

# save the modified JSON data to a new file
with open('corpus_CN_ENG_tokenized_modified.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


## sentiment analysis_CN ###
with open('corpus_CN_ENG_tokenized_modified.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# Loop through each item in the input data and analyze sentiment
for item in data:
    if item['Statement_Language(s)'] == 1 or item['Statement_Language(s)'] == 4:
        # Tokenize the Chinese text using jieba
        tokens = jieba.cut(item['Statement_CN'])

        # Join the tokens back into a string
        text = ' '.join(tokens)

        # Use SnowNLP to analyze the sentiment of the text
        s = SnowNLP(text)
        sentiment_CN = s.sentiments

        # Add the sentiment to the item in the data
        item['sentiment_CN'] = sentiment_CN
        print(f'item {item["NO."]} processed')

# Save the output data to a new file
with open('corpus_CN_senti_analysed.json', 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)

## sentiment analysis_ENG ###
Load the data from the input file
with open('corpus_CN_senti_analysed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize the sentiment analyzer
analyzer = vader.SentimentIntensityAnalyzer()

# Loop through the data and analyze the sentiment of English statements
for item in data:
    if item['Statement_Language(s)'] == 2 or item['Statement_Language(s)'] == 4:
        sentiment_ENG = analyzer.polarity_scores(item['Statement_ENG'])['compound']
        item['sentiment_ENG'] = sentiment_ENG
        print(f'item {item["NO."]} processed')

# Save the analyzed data to the output file
with open('corpus_ENG_senti_analysed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

## prepare for compare###
with open('corpus_ENG_senti_analysed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        if item['Statement_Language(s)'] == 4:
            print(f'item {item["NO."]} processing')
            word_count = (item['word_count_ENG'] + item['Word_Count_CN']) / 2
            STTR = (item['STTR_ENG'] + item['STTR_CN']) / 2
            Sentiment = (item['sentiment_CN'] + item['sentiment_ENG']) / 2
            item['word_count'] = word_count
            item['STTR'] = STTR
            item['Sentiment'] = Sentiment
            print(f'item {item["NO."]} processed')
        elif item['Statement_Language(s)'] == 1:
            print(f'item {item["NO."]} processing')
            item['word_count'] = item['Word_Count_CN']
            item['STTR'] = item['STTR_CN']
            item['Sentiment'] = item['sentiment_CN']
            print(f'item {item["NO."]} processed')
        elif item['Statement_Language(s)'] == 2:
            print(f'item {item["NO."]} processing')
            item['word_count'] = item['word_count_ENG']
            item['STTR'] = item['STTR_ENG']
            item['Sentiment'] = item['sentiment_ENG']
            print(f'item {item["NO."]} processed')

with open('corpus_prepared.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

########################## compute and test descriptives #####################
# read the json file
with open('corpus_prepared.json', 'r', encoding='utf-8') as f:
    corpus_data = json.load(f)

# divide the items into two groups according to "Site"
EPST_data = []
EMIS_data = []
for item in corpus_data:
    if item['Site'] == 'EPST':
        EPST_data.append(item)
    elif item['Site'] == 'EMIS':
        EMIS_data.append(item)

# read the value in the "Sentiment", "STTR", "word_count" keys of each item
EPST_sentiment = [item['Sentiment'] for item in EPST_data]
EPST_STTR = [item['STTR'] for item in EPST_data]
EPST_word_count = [item['word_count'] for item in EPST_data]

EMIS_sentiment = [item['Sentiment'] for item in EMIS_data]
EMIS_STTR = [item['STTR'] for item in EMIS_data]
EMIS_word_count = [item['word_count'] for item in EMIS_data]

# compute the average, standard deviation, medium value of the three values of the two groups
EPST_sentiment_avg = statistics.mean(EPST_sentiment)
EPST_sentiment_std = statistics.stdev(EPST_sentiment)
EPST_sentiment_med = statistics.median(EPST_sentiment)

EPST_STTR_avg = statistics.mean(EPST_STTR)
EPST_STTR_std = statistics.stdev(EPST_STTR)
EPST_STTR_med = statistics.median(EPST_STTR)

EPST_word_count_avg = statistics.mean(EPST_word_count)
EPST_word_count_std = statistics.stdev(EPST_word_count)
EPST_word_count_med = statistics.median(EPST_word_count)

EMIS_sentiment_avg = statistics.mean(EMIS_sentiment)
EMIS_sentiment_std = statistics.stdev(EMIS_sentiment)
EMIS_sentiment_med = statistics.median(EMIS_sentiment)

EMIS_STTR_avg = statistics.mean(EMIS_STTR)
EMIS_STTR_std = statistics.stdev(EMIS_STTR)
EMIS_STTR_med = statistics.median(EMIS_STTR)

EMIS_word_count_avg = statistics.mean(EMIS_word_count)
EMIS_word_count_std = statistics.stdev(EMIS_word_count)
EMIS_word_count_med = statistics.median(EMIS_word_count)

# test if they are significantly different between the two group using welch_ttest sentiment_ttest = ttest_ind(
EPST_sentiment, EMIS_sentiment, equal_var=False) STTR_ttest = ttest_ind(EPST_STTR, EMIS_STTR, equal_var=False)
word_count_ttest = ttest_ind(EPST_word_count, EMIS_word_count, equal_var=False) print(f"EPST_sentiment_avg = {
EPST_sentiment_avg:.2f}, SD = {EPST_sentiment_std:.2f}, median = {EPST_sentiment_med:.2f}") print(f"EPST_STTR_avg =
{EPST_STTR_avg:.2f}, SD = {EPST_STTR_std:.2f}, median = {EPST_STTR_med:.2f}") print(f"EPST_word_count_avg = {
EPST_word_count_avg:.2f}, SD = {EPST_word_count_std:.2f}, median = {EPST_word_count_med:.2f}") print(
f"EMIS_sentiment_avg = {EMIS_sentiment_avg:.2f}, SD = {EMIS_sentiment_std:.2f}, median = {EMIS_sentiment_med:.2f}")
print(f"EMIS_STTR_avg = {EMIS_STTR_avg:.2f}, SD = {EMIS_STTR_std:.2f}, median = {EMIS_STTR_med:.2f}") print(
f"EMIS_word_count_avg = {EMIS_word_count_avg:.2f}, SD = {EMIS_word_count_std:.2f}, median = {
EMIS_word_count_med:.2f}") print(sentiment_ttest) print(STTR_ttest) print(word_count_ttest)

# create a dictionary to store the variables
result_dict = {
    "EPST_sentiment_avg": f"{EPST_sentiment_avg:.2f}",
    "EPST_sentiment_SD": f"{EPST_sentiment_std:.2f}",
    "EPST_sentiment_median": f"{EPST_sentiment_med:.2f}",
    "EPST_STTR_avg": f"{EPST_STTR_avg:.2f}",
    "EPST_STTR_SD": f"{EPST_STTR_std:.2f}",
    "EPST_STTR_median": f"{EPST_STTR_med:.2f}",
    "EPST_word_count_avg": f"{EPST_word_count_avg:.2f}",
    "EPST_word_count_SD": f"{EPST_word_count_std:.2f}",
    "EPST_word_count_median": f"{EPST_word_count_med:.2f}",
    "EMIS_sentiment_avg": f"{EMIS_sentiment_avg:.2f}",
    "EMIS_sentiment_SD": f"{EMIS_sentiment_std:.2f}",
    "EMIS_sentiment_median": f"{EMIS_sentiment_med:.2f}",
    "EMIS_STTR_avg": f"{EMIS_STTR_avg:.2f}",
    "EMIS_STTR_SD": f"{EMIS_STTR_std:.2f}",
    "EMIS_STTR_median": f"{EMIS_STTR_med:.2f}",
    "EMIS_word_count_avg": f"{EMIS_word_count_avg:.2f}",
    "EMIS_word_count_SD": f"{EMIS_word_count_std:.2f}",
    "EMIS_word_count_median": f"{EMIS_word_count_med:.2f}",
    "sentiment_ttest": str(sentiment_ttest),
    "STTR_ttest": str(STTR_ttest),
    "word_count_ttest": str(word_count_ttest)
}

# write the dictionary to a JSON file
with open('descriptive_result.json', 'w') as f:
    json.dump(result_dict, f, indent=4)

########################## plot descriptives #####################
df = pd.read_json('corpus_prepared.json', encoding='utf-8')

# Divide the items into two groups based on the 'Site' column
grouped = df.groupby('Site')

# Extract the values of interest from each group
epst_vals = grouped.get_group('EPST')[['Sentiment', 'STTR', 'word_count']]
emis_vals = grouped.get_group('EMIS')[['Sentiment', 'STTR', 'word_count']]

# Create a 1x3 panel of subplots
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))


# Plot the first subplot: Sentiment
sns.set_palette("Set2")
sns.boxplot(y=epst_vals['Sentiment'], x='Site', data=df, ax=ax[0])
sns.boxplot(y=emis_vals['Sentiment'], x='Site', data=df, ax=ax[0])
sns.despine()
ax[0].set_xlabel('Sentiment')
add_stat_annotation(ax[0], data=df, x='Sentiment', y='Site', box_pairs=[("EPST", "EMIS")], test='Mann-Whitney', text_format='star', loc='inside')

# Plot the second subplot: STTR
sns.boxplot(y=epst_vals['STTR'], x='Site', data=df, ax=ax[1])
sns.boxplot(y=emis_vals['STTR'], x='Site', data=df, ax=ax[1])
sns.despine()
sns.set_palette("Set2")
ax[1].set_xlabel('STTR')
add_stat_annotation(ax[1], data=df, x='STTR', y='Site', box_pairs=[("EPST", "EMIS")], test='t-test_welch', text_format='star', loc='inside')

# Plot the third subplot: word_count
sns.boxplot(y=epst_vals['word_count'], x='Site', data=df, ax=ax[2])
sns.boxplot(y=emis_vals['word_count'], x='Site', data=df, ax=ax[2])
sns.despine()
sns.set_palette("Set2")
ax[2].set_xlabel('Word Count')
add_stat_annotation(ax[2], data=df, x='word_count', y='Site', box_pairs=[("EPST", "EMIS")], test='Mann-Whitney', text_format='star', loc='inside')

# Save the plot as 'descriptive_results.png'
plt.savefig('descriptive_results_new.png', dpi=600, bbox_inches='tight')

########################## test normality #####################
# # Read the data from the JSON file
with open('corpus_prepared.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Divide the data into two groups based on the 'Site' column
epst_data = df[df['Site'] == 'EPST']
emis_data = df[df['Site'] == 'EMIS']

# Define the columns to test for normality
columns = ['Sentiment', 'STTR', 'word_count', 'Capital_Economic_Capital', 'Capital_Cultural_Capital',
           'Capital_Social_Capital', 'Capital_Symbolic_Capital', 'Identity', 'Ideology']

# Test for normality and compute skewness, kurtosis, mean, and standard deviation within each group
results = {}
for group_name, group_data in [('EPST', epst_data), ('EMIS', emis_data)]:
    group_results = {}
    for col in columns:
        # Test for normality using Shapiro-Wilk test
        pvalue = shapiro(group_data[col])[1]
        if pvalue < 0.05:
            group_results[col] = {'normality': False, 'p-value': pvalue, 'skewness': skew(group_data[col]), 'kurtosis': kurtosis(group_data[col]), 'mean': group_data[col].mean(), 'std': group_data[col].std()}
            print(f'{col}: not normally distributed (p-value={pvalue:.4f})')
        else:
            group_results[col] = {'normality': True, 'p-value': pvalue, 'skewness': skew(group_data[col]), 'kurtosis': kurtosis(group_data[col]), 'mean': group_data[col].mean(), 'std': group_data[col].std()}
            print(f'{col}: normally distributed (p-value={pvalue:.4f})')
    results[group_name] = group_results

# Create the APA 7th format description of the data results
description = 'Descriptive statistics for variables tested for normality across two sites (EPST and EMIS). Mean and standard deviation are provided for normally distributed variables, while skewness and kurtosis are provided for variables that were not normally distributed. Shapiro-Wilk test was used to test for normality with a significance level of p < .05.'

# Add the description to the beginning of the results JSON file
# results.insert(0, ('description', description))

# Save the results to a JSON file
with open('corpus_value_normality_test.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

########################## visualized descriptives #####################
# Read the JSON file
with open('corpus_value_normality_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create empty DataFrame
df = pd.DataFrame(columns=['Measure', 'EPST', 'EPST_Norm', 'EMIS', 'EMIS_Norm'])

# Loop through measures
for measure in data['EPST']:
    epst_desc = describe(list(data['EPST'][measure].values()))
    emis_desc = describe(list(data['EMIS'][measure].values()))
    epst_mean, epst_sd = round(epst_desc.mean, 2), round(epst_desc.variance**0.5, 2)
    emis_mean, emis_sd = round(emis_desc.mean, 2), round(emis_desc.variance**0.5, 2)
    epst_norm = data['EPST'][measure]['normality']
    emis_norm = data['EMIS'][measure]['normality']
    row = [measure, f"{epst_mean} ({epst_sd})", epst_norm, f"{emis_mean} ({emis_sd})", emis_norm]
    df.loc[len(df)] = row

# Print DataFrame
print(df)
# Write output table to Excel file
df.to_excel('output3.xlsx', index=False)

# Print message indicating success
print('Output table written to output.xlsx')

# ########################### within group compare_prepared ###########################
with open('corpus_prepared_.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
# Loop through the data and extract the variables of interest
for item in data:
    investment_variable = {
        'Capital_Economic_Capital': item['Capital_Economic_Capital'],
        'Capital_Cultural_Capital': item['Capital_Cultural_Capital'],
        'Capital_Social_Capital': item['Capital_Social_Capital'],
        'Capital_Symbolic_Capital': item['Capital_Symbolic_Capital'],
        'Identity': item['Identity'],
        'Ideology': item['Ideology']
    }
    item['investment_variable'] = investment_variable

# Output the updated data to a new JSON file
with open('corpus_prepared_0406.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

########################### within group compare ###########################
with open('corpus_prepared_0406.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Divide the items into two groups
epst_data = []
emis_data = []

for item in data:
    if item['Site'] == 'EPST':
        # Convert the investment_variable dictionary to a frozen set
        item['investment_variable'] = frozenset(item['investment_variable'].items())
        epst_data.append(item)
    elif item['Site'] == 'EMIS':
        item['investment_variable'] = frozenset(item['investment_variable'].items())
        emis_data.append(item)

# Convert the epst_data list into a pandas DataFrame
df = pd.DataFrame(epst_data)

# Perform Welch's ANOVA
result = pg.welch_anova(data=df, dv='NO.', between='investment_variable')

# Print the result
print(result)

# Interpret the result
if result['p-unc'][0] < 0.05:
    print("There is a significant difference in the mean NO. between the different levels of investment_variable.")
    # get the unique levels of investment_variable
    levels = np.unique(df['investment_variable'])
    # calculate the variance for each level
    variances = []
    for level in levels:
        variances.append(np.var(df[df['investment_variable'] == level]['NO.']))
    # filter out levels with zero variance
    non_zero_var_levels = levels[np.array(variances) > 0]
    # perform Tukey's HSD test
    tukey = pairwise_tukeyhsd(df[df['investment_variable'].isin(non_zero_var_levels)]['NO.'],
                              df[df['investment_variable'].isin(non_zero_var_levels)]['investment_variable'])
    # convert the TukeyHSDResults object to a pandas DataFrame
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    # filter out non-significant pairs
    tukey_signif_pairs = tukey_df[tukey_df['p-adj'] < 0.05].reset_index()
    # check if the DataFrame is not empty before printing the significant pairs
    if not tukey_signif_pairs.empty:
        print(f"The following pairs have a significant difference in mean NO.: {tukey_signif_pairs['group1'][0]} and {tukey_signif_pairs['group2'][0]}.")
    else:
        print("No significant differences in mean NO. were found.")
else:
    print("There is no significant difference in the mean NO. between the different levels of investment_variable.")
