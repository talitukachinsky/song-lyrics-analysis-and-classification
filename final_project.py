import argparse
import csv
import os
import json
import re

import musicbrainzngs
import lyricsgenius
import pronouncing
from nltk.corpus import words
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

genius = lyricsgenius.Genius("WHgN5OPMIE02Bkkm57YNPvG5mKC4LkJhacVsOXLbcoZng3jwphOkn0IFUmvjOlkx")


# musicbrainzngs.set_useragent()

def clean(song_dict):  # cleans data
    for song in list(song_dict.keys()):
        if song_dict[song][2] == '':  # removes songs without lyrics
            del song_dict[song]
        if 'Song Title' in song:  # removes first line of csv file
            del song_dict[song]
    return song_dict


def create_json(songs_dict):
    data = {}
    data['songs'] = []
    for song in songs_dict:
        data['songs'].append({
            'lyrics': songs_dict[song][2],
            'song': song,
            'artist': songs_dict[song][0],
            'position': songs_dict[song][1],
            'year': songs_dict[song][3]
        })
    with open('data/song_data.txt', 'w') as outfile:
        json.dump(data, outfile)


class DataCreation:

    def get_data(self):
        song_dict = {}  # dict of songs with attributes as values
        genius.remove_section_headers = True  # removes section headers like "chorus" from songs
        files = os.listdir('data/billboard-master/billboard')  # list of files in folder
        randomindex = 72364  # random number to use as song title in case there are duplicates
        files.remove('.DS_Store')
        for year in files:  # each file contains csv of top 100 songs for that year
            csv_file = open('data/billboard-master/billboard/' + year)
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                print(year)
                print(row)
                duplicate_song = False
                value_array = []  # array of attributes
                position = row[0]  # position of song in charts
                artist = row[1]  # name of artist
                # genius often can't find lyrics with more than one artist. use first artist listed in this case
                if 'and' in artist:
                    artist = artist.split(' and ')[0]
                if 'feat.' in artist:
                    artist = artist.split(' feat. ')[0]
                song = row[2]  # name of song
                songgenius = None
                try:
                    songgenius = genius.search_song(song, artist)  # find the song in genius API
                except:
                    pass
                if songgenius is None:  # some song lyrics can't be found
                    lyrics = ''
                else:
                    lyrics = songgenius.lyrics  # get lyrics of song
                lyrics = lyrics.lower()  # put all the lyrics in lowercase
                year = year[:4]  # first four character of csv file refer to year
                value_array = [artist, position, lyrics, year]
                if song in song_dict.items():  # if two songs have the same name. going to assume that the same
                    # songs aren't in here more than once, but if they are then that's indicative of continued
                    # popularity so i'm fine with it. since it's a dict, i have to have different names for the
                    # songs, so everything after the first iteration will be followed by a string of ints
                    song = song + str(randomindex)
                    randomindex += 1
                song_dict.update({song: value_array})
        song_dict = clean(song_dict)
        create_json(song_dict)


class SongClassifier:

    def __init__(self):
        self.songs_dict = {}
        self.trait = ''
        self.x = []
        self.y = []
        self.x2 = []
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

    def year_classifier(self):
        self.trait = 'year'
        self.create_features()
        self.year_feature()
        self.classifier()
        self.year_analysis()

    def popularity_classifier(self):
        self.trait = 'position'
        self.create_features()
        self.classifier()

    def artist_classifier(self):
        self.trait = 'artist'
        self.create_features()
        self.classifier()

    def create_features(self):
        self.json_reader()
        self.lyrics_format()
        self.distinct_words()
        self.repeated_stanzas()
        self.repeated_lines()
        self.rhyme_percentage()
        self.rhyme_scheme()

    # makes two corresponding lists. x is the list of lyrics and y is the list of traits. get this from json file
    def json_reader(self):
        with open('data/song_data.txt') as json_file:
            data = json.load(json_file)
            for song in data['songs']:
                lyrics = song['lyrics']
                if self.trait == 'year':  # years are counted as decades instead of individual years
                    year = song['year']
                    year = year[2] + '0s'  # for example, '1955' gets turned into '50s'
                    trait = year
                elif self.trait == 'position':  # positions are split into groups of 20
                    position = song['position']
                    if position == '100':  # makes sure 100 doesn't get sorted with 11-19
                        trait = '80-100'
                    elif len(position) == 1:  # makes sure 1-9 are sorted in top 20
                        trait = '1-19'
                    else:
                        position = position[0]  # the rest can be sorted based on first digit
                        if position == '1':
                            trait = '1-19'
                        if position in ('2', '3'):
                            trait = '20-39'
                        if position in ('4', '5'):
                            trait = '40-59'
                        if position in ('6', '7'):
                            trait = '60-79'
                        if position in ('8', '9'):
                            trait = '80-100'
                elif self.trait == 'artist':  # no intuitive way to group artists together
                    trait = song['artist']
                self.x.append(lyrics)
                self.y.append(trait)

    # divides lyrics into stanzas, lines, and words. list within a list within a list.
    def lyrics_format(self):
        for i, song in enumerate(self.x):
            song_lyric_list = []
            stanzas_list = song.split('\n\n')  # each song is split into stanzas
            for stanza in stanzas_list:
                stanza_list = []
                lines_list = stanza.split('\n')  # each stanza is split into lines
                for line in lines_list:
                    words_list = re.split("[^a-zA-Z0-9']+", line)  # each line is split into words, including
                    # apostrophes for contractions
                    line_list = words_list  # a line is a list of words
                    stanza_list.append(line_list)  # a stanza is a list of lines
                song_lyric_list.append(stanza_list)  # a song is a list of stanzas
            self.x[i] = song_lyric_list

    def distinct_words(self):
        for i, song in enumerate(self.x):
            feature_array = []
            word_set = set()
            word_list = []
            line_counter = 0
            for stanza in song:
                for line in stanza:
                    line_counter += 1
                    for word in line:
                        word_set.add(word)
                        word_list.append(word)
            percentage_distinct = len(word_set) / len(word_list)  # number of distinct words over number of total words
            feature_array.append(percentage_distinct)
            feature_array.append(len(word_list))  # also total words could be a useful feature
            feature_array.append(len(word_list) / line_counter)  # average number of words per line
            self.x2.append(feature_array)  # corresponding array of just features

    def repeated_stanzas(self):
        for i, song in enumerate(self.x):
            stanza_count = 0
            stanza_list = []
            stanza_set = set()
            for stanza in song:
                stanza_count += 1
                stanza_list.append(str(stanza))
                stanza_set.add(str(stanza))
            percentage_repeated = 1 - (len(stanza_set) / len(stanza_list))
            self.x2[i].append(percentage_repeated)
            self.x2[i].append(stanza_count)

    def repeated_lines(self):
        for i, song in enumerate(self.x):
            line_count = 0
            line_list = []
            line_set = set()
            for stanza in song:
                for line in stanza:
                    line_count += 1
                    line_set.add(str(line))
                    line_list.append(str(line))
            percentage_repeated = 1 - (len(line_set) / len(line_list))  # 1 minus percentage of unique lines
            self.x2[i].append(percentage_repeated)
            self.x2[i].append(line_count)  # total number of lines

    # percentage of lines that have a rhyme
    def rhyme_percentage(self):
        for i, song in enumerate(self.x):
            rhyme_count = 0
            last_word_list = []
            for stanza in song:
                for line in stanza:
                    if line[-1] == '':  # sometimes there's an extra empty word at the end of some lines
                        if len(line) > 1:
                            last_word = line[-2]
                        else:
                            last_word = ''
                    else:
                        last_word = line[-1]
                    last_word_list.append(last_word)
            for word in last_word:  # counts how many lines rhyme with something
                rhyme_list = pronouncing.rhymes(word)
                if any(w in last_word for w in rhyme_list):
                    rhyme_count += 1
                else:  # if word is not in rhyming dictionary, also count same word as a rhyme
                    counter = 0
                    while counter < 2:
                        for w in last_word:
                            if w == word:
                                counter += 1
                            if counter == 2:  # if word occurs more than once at the end of a line, counts as rhyme
                                rhyme_count += 1
            rhyme_percentage = rhyme_count / len(last_word_list)
            self.x2[i].append(rhyme_percentage)

    def percentage_english_words(self):
        for i, song in enumerate(self.x):
            word_counter = 0
            english_word_counter = 0
            for stanza in song:
                for line in stanza:
                    for word in line:
                        word_counter += 1
                        if word in words.words():
                            english_word_counter += 1
            percentage_english_words = english_word_counter / word_counter
            self.x2[i].append(percentage_english_words)

    # defines the rhyme scheme within a stanza
    def rhyme_scheme(self):
        for i, song in enumerate(self.x):
            rhyme_array = [0, 0, 0, 0, 0, 0, 0,
                           0]  # there are 8 rhyme schemes that I'll be looking for. If present, it'll go into this array as a 1. If not, it'll stay as 0.
            for stanza in song:
                last_word_list = []
                if len(stanza) > 4:
                    stanza = stanza[:4]  # only going to look at first four lines of each stanza
                for line in stanza:
                    if line[-1] == '':  # sometimes there's an extra empty word at the end of some lines
                        if len(line) > 1:
                            last_word = line[-2]
                        else:
                            last_word = ''
                    else:
                        last_word = line[-1]
                    last_word_list.append(last_word)
                if len(last_word_list) < 4:  # can only look at groups of 4
                    rhyme_scheme = 'none'
                else:
                    rhyme_scheme = self.rhyme_scheme_test(last_word_list)
                if rhyme_scheme == 'AAAA':
                    rhyme_array[0] = 1
                elif rhyme_scheme == 'AAAX':
                    rhyme_array[1] = 1
                elif rhyme_scheme == 'AABB':
                    rhyme_array[2] = 1
                elif rhyme_scheme == 'AAXA':
                    rhyme_array[3] = 1
                elif rhyme_scheme == 'ABAB':
                    rhyme_array[4] = 1
                elif rhyme_scheme == 'AXAA':
                    rhyme_array[5] = 1
                elif rhyme_scheme == 'XAXA':
                    rhyme_array[6] = 1
                elif rhyme_scheme == 'XXXX':
                    rhyme_array[7] = 1
            for value in rhyme_array:  # will be added as separate values to the actual vector
                self.x2[i].append(value)

    def rhyme_scheme_test(self, word_list):
        if self.is_rhyme(word_list[0], word_list[1]):  # AA
            if self.is_rhyme(word_list[1], word_list[2]):  # AAA
                if self.is_rhyme(word_list[2], word_list[3]):  # AAAA
                    return 'AAAA'
                else:
                    return 'AAAX'
            else:
                if self.is_rhyme(word_list[2], word_list[3]):
                    return 'AABB'
                elif self.is_rhyme(word_list[1], word_list[3]):
                    return 'AAXA'
                else:
                    return 'none'
        else:
            if self.is_rhyme(word_list[0], word_list[2]):  # AXA/ABA
                if self.is_rhyme(word_list[1], word_list[3]):
                    return 'ABAB'
                elif self.is_rhyme(word_list[0], word_list[3]):
                    return 'AXAA'
                else:
                    return 'none'
            else:
                if self.is_rhyme(word_list[1], word_list[3]):
                    return 'XAXA'
                else:
                    return 'XXXX'

    def is_rhyme(self, word1, word2):
        rhyme_list = pronouncing.rhymes(word1)
        if word2 in rhyme_list:
            return True
        elif word1 == word2:
            return True
        else:
            return False

    # binary feature for whether a year between 1950-2019 gets mentioned
    def year_feature(self):
        for i, song in enumerate(self.x):
            year_array = [0, 0, 0, 0, 0, 0, 0]  # separate feature for each decade
            for stanza in song:
                for line in stanza:
                    for word in line:
                        if '19' in word[:2] and len(word) > 3:
                            if word[2] == '5':
                                year_array[0] = 1
                            elif word[2] == '6':
                                year_array[1] = 1
                            elif word[2] == '7':
                                year_array[2] = 1
                            elif word[2] == '8':
                                year_array[3] == 1
                            elif word[2] == '9':
                                year_array[4] == 1
                        elif '20' in word[:2] and len(word) > 3:
                            if word[2] == '0':
                                year_array[5] == 1
                            elif word[2] == '1':
                                year_array[6] == 1
            for value in year_array:  # will be added as separate values to the actual vector
                self.x2[i].append(value)

    def year_analysis(self):
        fifties = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], '50s']
        sixties = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], '60s']
        seventies = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], '70s']
        eighties = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], '80s']
        nineties = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], '90s']
        zeros = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], '2000s']
        tens = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], '2010s']
        for i, feature_list in enumerate(self.x2):
            if self.y[i] == '50s':
                array = fifties
            elif self.y[i] == '60s':
                array = sixties
            elif self.y[i] == '70s':
                array = seventies
            elif self.y[i] == '80s':
                array = eighties
            elif self.y[i] == '90s':
                array = nineties
            elif self.y[i] == '00s':
                array = zeros
            elif self.y[i] == '10s':
                array = tens
            for j, feature in enumerate(feature_list):
                array[j].append(feature)
        data_array = [fifties, sixties, seventies, eighties, nineties, zeros, tens]
        for data in data_array:
            print(data[-1])
            averages_list = []
            for array in data[:-1]:
                average = sum(array)/len(array)
                averages_list.append(str(average))
            print('Percentage of Distinct Words: '+averages_list[0])
            print('Number of Words: '+averages_list[1])
            print('Average Number of Word Per Line: '+averages_list[2])
            print('Percentage of Repeated Stanzas: '+averages_list[3])
            print('Number of Stanzas: '+averages_list[4])
            print('Percentage of Repeated Line: '+averages_list[5])
            print('Number of Lines: '+averages_list[6])
            print('Percentage of Rhyming Lines: '+averages_list[7])
            print('Contains AAAA Rhyme Pattern: '+averages_list[8])
            print('Contains AAAX Rhyme Pattern: '+averages_list[9])
            print('Contains AABB Rhyme Pattern: '+averages_list[10])
            print('Contains AAXA Rhyme Pattern: '+averages_list[11])
            print('Contains ABAB Rhyme Pattern: '+averages_list[12])
            print('Contains AXAA Rhyme Pattern: '+averages_list[13])
            print('Contains XAXA Rhyme Pattern: '+averages_list[14])
            print('Contains XXXX Rhyme Pattern: '+averages_list[15])
            print('Mentions 1950s: '+averages_list[16])
            print('Mentions 1960s: '+averages_list[17])
            print('Mentions 1970s: '+averages_list[18])
            print('Mentions 1980s: '+averages_list[19])
            print('Mentions 1990s: '+averages_list[20])
            print('Mentions 2000s: '+averages_list[21])
            print('Mentions 2010s: '+averages_list[22])




    def classifier(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x2, self.y, test_size=0.1,
                                                                                random_state=0)
        classifier = BernoulliNB()
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        accuracy = str(accuracy_score(self.y_test, y_pred))
        print('Accuracy:' + accuracy)


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", action='store_true', help="train a classifier")
parser.add_argument("-r", "--run", nargs=2, help="run classifier")
parser.add_argument("--test", action='store_true')
parser.add_argument('--test2', action='store_true')

# py final_project.py --test2
DataCreator = DataCreation()
SongClassifier = SongClassifier()
args = parser.parse_args()
if args.test:
    DataCreator.get_data()
if args.test2:
    SongClassifier.year_classifier()
