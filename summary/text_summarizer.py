import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
import math

def summarization(text):
    sent_tokens = nltk.sent_tokenize(text)
    word_tokens = nltk.word_tokenize(text)
    word_tokens_lower=[word.lower() for word in word_tokens]
    stopWords = list(set(stopwords.words("english")))
    word_tokens_refined=[x for x in word_tokens_lower if x not in stopWords]
    #print(len(word_tokens_refined))
    stemmed = [ ]
    stemmer = PorterStemmer( )
    for w in word_tokens_refined:
        stemmed.append(stemmer.stem(w))

    word_tokens_refined=stemmed
    cue=["example", "anyway", "furthermore", "first", "second", "then", "now", "therefore", "hence", "lastly", "finally", "summary","in conclusion"]
    cue_phrases={}

    for s in sent_tokens:
        cue_phrases[s]=0
        word_tokens=word_tokenize(s)
        for w in word_tokens:
            if w.lower() in cue:
                cue_phrases[s]+=1

    max_freq = max(cue_phrases.values())
    for k in cue_phrases.keys():
        try:
            cue_phrases[k] = (cue_phrases[k] / max_freq)
            cue_phrases[k]=round(cue_phrases[k],3)
        except ZeroDivisionError:
            x=0

    sent_len_score={}
    for sentence in sent_tokens:
        sent_len_score[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        if len(word_tokens) in range(0,10):
            sent_len_score[sentence]=1-0.05*(10-len(word_tokens))
        elif len(word_tokens) in range(7,20):
            sent_len_score[sentence]=1
        else:
            sent_len_score[sentence]=1-(0.05)*(len(word_tokens)-20)
    for k in sent_len_score.keys():
        sent_len_score[k]=round(sent_len_score[k],4)

    sentence_position={}
    d=1
    no_of_sent=len(sent_tokens)
    for i in range(no_of_sent):
        a=1/d
        b=1/(no_of_sent-d+1)
        sentence_position[sent_tokens[d-1]]=max(a,b)
        d=d+1
    for k in sentence_position.keys():
        sentence_position[k]=round(sentence_position[k],3)

    freqTable = {}
    for word in word_tokens_refined:
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    for k in freqTable.keys():
        freqTable[k]= math.log10(1+freqTable[k])
    #Compute word frequnecy score of each sentence
    word_frequency={}
    for sentence in sent_tokens:
        word_frequency[sentence]=0
        e=word_tokenize(sentence)
        f=[]
        for word in e:
            f.append(stemmer.stem(word))
        for word,freq in freqTable.items():
            if word in f:
                word_frequency[sentence]+=freq
    maximum=max(word_frequency.values())
    for key in word_frequency.keys():
        try:
            word_frequency[key]=word_frequency[key]/maximum
            word_frequency[key]=round(word_frequency[key],3)
        except ZeroDivisionError:
            x=0

    upper_case={}
    for sentence in sent_tokens:
        upper_case[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k.isupper():
                upper_case[sentence] += 1
    maximum_frequency = max(upper_case.values())
    for k in upper_case.keys():
        try:
            upper_case[k] = (upper_case[k]/maximum_frequency)
            upper_case[k] = round(upper_case[k], 3)
        except ZeroDivisionError:
            x=0

    proper_noun={}
    for sentence in sent_tokens:
        tagged_sent = pos_tag(sentence.split())
        propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
        proper_noun[sentence]=len(propernouns)
    maximum_frequency = max(proper_noun.values())
    for k in proper_noun.keys():
        try:
            proper_noun[k] = (proper_noun[k]/maximum_frequency)
            proper_noun[k] = round(proper_noun[k], 3)
        except ZeroDivisionError:
            x=0

    head_match={}
    heading=sent_tokens[0]
    for sentence in sent_tokens:
        head_match[sentence]=0
        word_tokens = word_tokenize(sentence)
        for k in word_tokens:
            if k not in stopWords:
                k = stemmer.stem(k)
                if k in stemmer.stem(heading):
                    head_match[sentence] += 1
    maximum_frequency = max(head_match.values())
    for k in head_match.keys():
        try:
            head_match[k] = (head_match[k]/maximum_frequency)
            head_match[k] = round(head_match[k], 3)
        except ZeroDivisionError:
            x=0

    numeric_data={}
    for sentence in sent_tokens:
        numeric_data[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k.isdigit():
                numeric_data[sentence] += 1
    maximum_frequency = max(numeric_data.values())
    for k in numeric_data.keys():
        try:
            numeric_data[k] = (numeric_data[k]/maximum_frequency)
            numeric_data[k] = round(numeric_data[k], 3)
        except ZeroDivisionError:
            x=0

    total_score={}
    for k in cue_phrases.keys():
        total_score[k]=cue_phrases[k]+numeric_data[k]+sent_len_score[k]+sentence_position[k]+word_frequency[k]+upper_case[k]+proper_noun[k]+head_match[k]

    sumValues = 0
    for sentence in total_score:
        sumValues += total_score[sentence]
    average = sumValues / len(total_score)
    #print(average)
    # Storing sentences into our summary.
    summ = ''
    for sentence in sent_tokens:
        if (sentence in total_score) and (total_score[sentence] > (1.2*average)):
            summ += " " + sentence

    return summ
