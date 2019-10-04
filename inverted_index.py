import numpy as np
import os
import os.path
from os import path
import operator
import nltk
from nltk.corpus import stopwords
import pickle
import string
from os.path import isfile, join
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from collections import OrderedDict
from operator import itemgetter
from natsort import natsorted
from nltk.stem import *
from nltk.corpus import stopwords

def read_file(filename):
    with open(filename, 'r', encoding="ascii", errors="surrogateescape") as f:
        stuff=f.read()
    f.close()
    stuff=remove_header_footer(stuff)		# Remove header and footer.
    return stuff

def open_file():
	final_string=""
	folder_path="20_newsgroups/comp.graphics" #comp.graphics rec.motorcycles
	onlyfiles = [f for f in os.listdir(folder_path)]
	for f in onlyfiles:
		# Read the current file
		with open(folder_path+"/"+f,encoding="utf8", errors='ignore') as myfile:
			file_content=myfile.read()
			file_content=remove_header_footer(file_content)		# Remove header and footer.
			final_string+=file_content
	return final_string

def remove_header_footer(final_string):
	new_final_string=""
	tokens=final_string.split('\n\n')
	# Remove tokens[0] and tokens[-1]
	for token in tokens[1:-1]:
		new_final_string+=token+" "
	return new_final_string

def remove_punctuations(final_string):
	tokenizer=TweetTokenizer()
	token_list=tokenizer.tokenize(final_string)
	table = str.maketrans('', '', '\t')
	token_list = [word.translate(table) for word in token_list]
	punctuations = (string.punctuation).replace("'", "")
	trans_table = str.maketrans('', '', punctuations)
	stripped_words = [word.translate(trans_table) for word in token_list]
	token_list = [str for str in stripped_words if str]
	token_list=[word.lower() for word in token_list]
	return token_list

def or_command(x,y, inv_index):
    docs_with_x=[]
    docs_with_y=[]
    try:
        docs_with_x=inv_index[x][1]
        docs_with_y=inv_index[y][1]
    except:
        pass
    return list(set(docs_with_x+docs_with_y))           # Merge the lists
    

def and_command(x,y, inv_index):
    docs_with_x=[]
    docs_with_y=[]
    try:
        docs_with_x=inv_index[x][1]
        docs_with_y=inv_index[y][1]
    except:
        pass
    return list(set(docs_with_x) & set(docs_with_y))                        # Intersection of the lists

def not_command(x, inv_index, doc_mapping):
    docs_without_x=doc_mapping.keys()                 # Take all the documents initially
    try:
        docs_with_x=inv_index[x]
    except:
        return docs_without_x                           # x doesnt exist in any document
    return [docid for docid in docs_without_x if docid not in docs_with_x]

def map_to_filename(doc_list, doc_mapping):
    final_doc_list=[]
    for doc in doc_list:
        final_doc_list.append(doc_mapping[doc])
    return final_doc_list

inv_index=np.load('inv_index.npy').item()
doc_mapping=np.load('doc_mapping.npy').item()
fileno=0                                                # Map the doc name to a single integer (ID).

folder_names=natsorted(os.listdir("20_newsgroups"))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')) 

## FORMING INVERTED INDEX ##
if not path.exists('inv_index.npy') or not path.exists('doc_mapping.npy'):
	for folder_name in folder_names:
	    file_names=natsorted(os.listdir("20_newsgroups/"+folder_name))
	    print(folder_name)
	    for file_name in file_names:
		#doc_mapping[fileno]=folder_name+"/"+file_name
		stuff=read_file("20_newsgroups/"+folder_name+"/"+file_name)
		final_token_list=remove_punctuations(stuff)	# This is the list of words in order of the text.
		final_token_list=set(final_token_list)
		for term in final_token_list:
		    if not term in stop_words:                  # If word is not in stopwords
		        # First stem the term
		        term=stemmer.stem(term)
		        if term in inv_index:
		            # Increment doc freq by 1
		            inv_index[term][0]=inv_index[term][0]+1
		            # Add doc ID to postings list
		            inv_index[term][1].append(fileno)
		        else:
		            inv_index[term]=[1, [fileno]]
		fileno+=1                                       # Increment the file no. counter for document ID mapping
	np.save('inv_index.npy', inv_index)
	np.save('doc_mapping.npy', doc_mapping)


print("Select query type:")
print("1. x OR y\n2. x AND y\n3. x AND NOT y\n4. x OR NOT y")
option=input()

print("Enter x")
x=input()
x=stemmer.stem(x)
print("Enter y")
y=input()
y=stemmer.stem(y)

if (option=="1"):
    doc_list=or_command(x,y, inv_index)
elif (option=="2"):
    doc_list=and_command(x,y, inv_index)
elif (option=="3"):
    docs_with_y=not_command(y, inv_index, doc_mapping)
    docs_with_x=[]
    try:
        docs_with_x=inv_index[x][1]
    except:
        pass
    doc_list=list(set(docs_with_x) & set(docs_with_y))
elif (option=="4"):
    docs_with_y=not_command(y, inv_index, doc_mapping)
    docs_with_x=[]
    try:
        docs_with_x=inv_index[x][1]
    except:
        pass
    doc_list=list(set(docs_with_x+docs_with_y)) 
else:
    print("Invalid")

print(map_to_filename(doc_list, doc_mapping))
