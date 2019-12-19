# Song Lyric Analysis and Classification by Decade

## Necessary Libraries

This program requires the following libraries to be imported:

- import argparse

- import csv

- import os

- import json

- import re


- import lyricsgenius

- import pandas

- import matplotlib

- import matplotlib.pyplot as plt

- import pronouncing

- from nltk.corpus import words

- from sklearn.metrics import accuracy_score

- from sklearn.model_selection import train_test_split

- from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

## How to Run

This program has three commands

##### Analyzing Data

This builds a classifier for which decade a song is from based on its lyrics. It prints all the data it finds and creates three graphs based on this data.
```
py final_project.py --analyze
```

##### Classifying Songs

This builds a decade classifier and also takes one argument that is the name of a txt file in the data folder. There are three of these files provided, called `song1.txt`, `song2.txt`, and `song3.txt`, which all contain lyrics that are formatted the same way that the training data is. The following command classifies a .txt file of lyrics as a specific decade, and also prints the accuracy of the classifier and the average difference between the decade predicted and the actual decade (e.g. '80s and '90s have a difference of 1).
```
py final_project.py --run song1.txt
```

##### Creating Data

This finds and creates all the data and puts it into a JSON file located in `data/song_data.txt`. This is already created and takes several hours to run because it scrapes about 6000 songs worth of data from Genius.com one by one, so I would recommend not running it since the data is already provided. This data is used for analysis and classification. The data is created by also using the data in `data/billboard-master/billboard`, which provides the title and artist of the Top 100 songs of each year between 1950 and 2015.
```
py final_project.py --create-data
```
