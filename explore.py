import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

def ngrams_creator(input_string, n_grams = 2):
    """
    This function takes in a list and returns a list of grams.
    """
    ngrams = nltk.ngrams(input_string.split(), n_grams)
    return list(ngrams)

def explore_visual_1(df):
    """
    Visual to show the average word count by programming language
    """
    
    df['readme_length'] = df['readme_contents'].apply(len)
    
    avg_len_ruby =df[df['language']== 'Ruby']['readme_length'].mean()
    avg_len_python= df[df['language']== 'Python']['readme_length'].mean()
    avg_len_java= df[df['language']== 'Java']['readme_length'].mean()
    avg_len_js = df[df['language']== 'JavaScript']['readme_length'].mean()
    avg_len_c= df[df['language']== 'C_based']['readme_length'].mean()
    
    avg_df=pd.DataFrame({'Ruby': [avg_len_ruby],
                    'Python': [avg_len_python],
                    'Java':[avg_len_java],
                    'JavaScript': [avg_len_js],
                    'C_based': [avg_len_c]})
    plt.subplots(facecolor="gainsboro")
    sns.barplot(avg_df, palette='Paired')
    plt.title('Average Word Count of Readme by Programming Language', fontsize=15)
    plt.xticks(ticks=[0,1,2,3,4], labels=['Ruby','Python', 'Java', 'JavaScript', 'C Based'])
    plt.xlabel('Programming Languages', fontsize=12)
    plt.ylabel('Average', fontsize=12)
    
def explore_visual_2(train):
    """
    Creates a visual to show bi grams.
    """
    big_rams_stem = []
    for row in train['readme_stem_no_swords'].apply(ngrams_creator):
        big_rams_stem.extend(row)
        
    bi_stem_series = pd.Series(big_rams_stem)
    
    top_25_readme_bigrams = bi_stem_series.value_counts().head(25)
    top_25_readme_bigrams.sort_values(ascending=True).plot.barh(color='royalblue', width=.9, figsize=(10, 6))
    
    plt.title('25 Most frequently occuring readme bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_25_readme_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
def series_words_func(input_string):
    """
    Takes an input string and returns a list
    """
    flat_lst = []

    for word in input_string:
        flat_lst.extend(word.split())

    return flat_lst

def ruby_unique_words(df):
    """
    Takes in a dataframe and displayes the top unique words for ruby.
    """
    #Seperates dataframes by their programming language
    ruby = df[df['language']== 'Ruby'].readme_stem_no_swords
    python = df[df['language']== 'Python'].readme_stem_no_swords
    java = df[df['language']== 'Java'].readme_stem_no_swords
    javascript = df[df['language']== 'JavaScript'].readme_stem_no_swords
    c_based = df[df['language']== 'C_based'].readme_stem_no_swords
    all_words = df.readme_stem_no_swords
    
    #Takes all the strings and turns them into list of words
    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    all_words = series_words_func(all_words)   
    
    #Gets the unique words from each list of words    
    ruby_unique = pd.Series(ruby_words).unique()
    python_unique = pd.Series(python_words).unique()
    java_unique = pd.Series(java_words).unique()
    javascript_unique = pd.Series(javascript_words).unique()
    c_based_unique = pd.Series(c_based_words).unique()
    all_unique = pd.Series(all_words).unique()
    
    
    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    
    
    #Creates a DataFrame of all the word counts for each language
    word_counts = (pd.concat([ruby_freq, python_freq, java_freq,
                              javascript_freq, c_based_freq, all_freq],
                             axis=1, sort=True).set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'All'], axis=1, inplace=False).fillna(0).apply(lambda s: s.astype(int)))
    
    #Creates a DataFrame with only words in ruby
    top_ruby_words = word_counts[(word_counts.Python == 0) & (word_counts.Java == 0) & (word_counts.Javascript == 0) & (word_counts.C_based == 0)].sort_values(by=['Ruby'], ascending=False).head(25)
    
    #Builds the plot
    plt.subplots(facecolor="gainsboro")
    plt.bar(height = top_ruby_words.Ruby.head(), x=top_ruby_words.index[:5],color='royalblue')
    plt.title('Total Count for Unique of Words In Ruby')
    plt.xlabel('Ruby words')
    plt.ylabel('Total Count')
    plt.show()
    
def python_unique_words(df):
    """
    Takes in a dataframe and displayes the top unique words for python.
    """
    #Seperates dataframes by their programming language
    ruby = df[df['language']== 'Ruby'].readme_stem_no_swords
    python = df[df['language']== 'Python'].readme_stem_no_swords
    java = df[df['language']== 'Java'].readme_stem_no_swords
    javascript = df[df['language']== 'JavaScript'].readme_stem_no_swords
    c_based = df[df['language']== 'C_based'].readme_stem_no_swords
    all_words = df.readme_stem_no_swords
    
    #Takes all the strings and turns them into list of words
    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    all_words = series_words_func(all_words)   
    
    #Gets the unique words from each list of words    
    ruby_unique = pd.Series(ruby_words).unique()
    python_unique = pd.Series(python_words).unique()
    java_unique = pd.Series(java_words).unique()
    javascript_unique = pd.Series(javascript_words).unique()
    c_based_unique = pd.Series(c_based_words).unique()
    all_unique = pd.Series(all_words).unique()
    
    
    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    
    
    #Creates a DataFrame of all the word counts for each language
    word_counts = (pd.concat([ruby_freq, python_freq, java_freq,
                              javascript_freq, c_based_freq, all_freq],
                             axis=1, sort=True).set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'All'], axis=1, inplace=False).fillna(0).apply(lambda s: s.astype(int)))
    
    #Creates a DataFrame with only words in ruby
    top_python_words = word_counts[(word_counts.Ruby == 0) & (word_counts.Java == 0) & (word_counts.Javascript == 0) & (word_counts.C_based == 0)].sort_values(by=['Python'], ascending=False).head(25)
    
    #Builds the plot
    plt.subplots(facecolor="gainsboro")
    plt.bar(height = top_python_words.Python.head(), x=top_python_words.index[:5],color='royalblue')
    plt.title('Total Count for Unique Words In Python')
    plt.xlabel('Python words')
    plt.ylabel('Total Count')
    plt.show()
    
def javascript_unique_words(df):
    """
    Takes in a dataframe and displayes the top unique words for python.
    """
    #Seperates dataframes by their programming language
    ruby = df[df['language']== 'Ruby'].readme_stem_no_swords
    python = df[df['language']== 'Python'].readme_stem_no_swords
    java = df[df['language']== 'Java'].readme_stem_no_swords
    javascript = df[df['language']== 'JavaScript'].readme_stem_no_swords
    c_based = df[df['language']== 'C_based'].readme_stem_no_swords
    all_words = df.readme_stem_no_swords
    
    #Takes all the strings and turns them into list of words
    ruby_words= series_words_func(ruby)
    python_words= series_words_func(python)
    java_words= series_words_func(java)
    javascript_words= series_words_func(javascript)
    c_based_words= series_words_func(c_based)
    all_words = series_words_func(all_words)   
    
    #Gets the unique words from each list of words    
    ruby_unique = pd.Series(ruby_words).unique()
    python_unique = pd.Series(python_words).unique()
    java_unique = pd.Series(java_words).unique()
    javascript_unique = pd.Series(javascript_words).unique()
    c_based_unique = pd.Series(c_based_words).unique()
    all_unique = pd.Series(all_words).unique()
    
    
    ruby_freq = pd.Series(ruby_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    c_based_freq = pd.Series(c_based_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    
    
    #Creates a DataFrame of all the word counts for each language
    word_counts = (pd.concat([ruby_freq, python_freq, java_freq,
                              javascript_freq, c_based_freq, all_freq],
                             axis=1, sort=True).set_axis(['Ruby', 'Python', 'Java', 'Javascript', 'C_based', 'All'], axis=1, inplace=False).fillna(0).apply(lambda s: s.astype(int)))
    
    #Creates a DataFrame with only words in ruby
    top_javascript_words = word_counts[(word_counts.Ruby == 0) & (word_counts.Java == 0) & (word_counts.Python == 0) & (word_counts.C_based == 0)].sort_values(by=['Javascript'], ascending=False).head(25)
    
    #Builds the plot
    plt.subplots(facecolor="gainsboro")
    plt.bar(height = top_javascript_words.Javascript.head(), x=top_javascript_words.index[:5],color='royalblue')
    plt.title('Total Count for Unique Words In JavaScript')
    plt.xlabel('JavaScript words')
    plt.ylabel('Total Count')
    plt.show()