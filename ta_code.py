import os
import csv
import nltk
import random
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# Gathering the names of the corpus
corpus_names = next(os.walk('Data/test_docs'))[2]

# Collecting the company names, only focusing on 1-word company names

f = open('Data/all_entities/u_1w_companies.csv')
csv_f = csv.reader(f)
uc = []
for row in csv_f:
  uc.append(row[0])

# Part of speech tags
adjectives = ['JJ','JJR','JJS']
nouns = ['NN','NNS','NNP','NNPS']
verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']
adverbs = ['RB','RBR','RBS']
ending_punct = ['.']
possessive = ['POS']

bus_words = ['stock','share','shares']

# Populate the features for the company names
def get_features():
	
	res = []
	feat = []
	for text in corpus_names:

		print 'Now processing ' + text

		document = open('Data/test_docs/' + text, 'rb')

		text = document.read().decode('ascii','ignore')
		text = text.encode('utf-8')

		tkns = nltk.word_tokenize(text)
		tkns_pos = nltk.pos_tag(tkns)

		# Collect the 1's for classification
		for comp in uc:

			uc_indices = [i for i, word in enumerate(tkns_pos) if word[0] == comp]

			if uc_indices:
				
				for i in uc_indices:

					# The nearest 5 words on either side
					sample_section_pos = tkns_pos[i-5:i+6]

					# Feature 1: Followed by a verb?
					f1 = feature_1(tkns_pos, i)

					# Feature 2: Followed by a business word?
					f2 = feature_2(tkns_pos, i)

					# Feature 3: Followed by sentence ending punctuation?
					f3 = feature_3(tkns_pos, i)

					# Feature 4: Two or more capitalized words in the sample_section
					f4 = feature_4(sample_section_pos)

					# Feature 5: Followed by an 's ?
					f5 = feature_5(tkns_pos, i)

					# The 1 is to indicate that this is a 1-word company name
					res.append([1])
					feat.append([f1, f2, f3, f4, f5])

		# Collect the 0's for classification
		num_zeros = 6 * len(feat)
		zero_indices = []
		for z in range(0,num_zeros):

			i = random.randrange(10, len(tkns_pos)-10)
			zero_indices.append(i)

		# Remove any duplicates
		zero_indices = sorted(list(set(zero_indices)))

		for i in zero_indices:

			# The nearest 5 words on either side
			sample_section_pos = tkns_pos[i-5:i+6]

			# Feature 1: Followed by a verb?
			f1 = feature_1(tkns_pos, i)

			# Feature 2: Followed by a business word?
			f2 = feature_2(tkns_pos, i)

			# Feature 3: Followed by sentence ending punctuation?
			f3 = feature_3(tkns_pos, i)

			# Feature 4: Two or more capitalized words in the sample_section?
			f4 = feature_4(sample_section_pos)

			# Feature 5: Followed by an 's ?
			f5 = feature_5(tkns_pos, i)

			# The 0 is to indicate that this is a random, non-company word
			res.append([0])
			feat.append([f1, f2, f3, f4, f5])


	res = numpy.asarray(res)
	feat = numpy.asarray(feat)

	document.close()
	return res, feat

def fit_model(feat, res):

	res = numpy.ravel(res)

	model = LogisticRegression()
	model.fit(feat,res)

	score = model.score(feat,res)

	feat_train, feat_test, res_train, res_test = train_test_split(feat, res, test_size=0.3, random_state=0)

	model2 = LogisticRegression()
	model2.fit(feat_train, res_train)

	predicted = model2.predict(feat_test)

	stats = metrics.classification_report(res_test, predicted)

	return score, stats

def feature_1(text, index):
	
	f1 = 1 if text[index+1][1] in verbs else 0

	return f1

def feature_2(text, index):
	
	f2 = 1 if text[index+1][0] in bus_words else 0

	return f2

def feature_3(text, index):
	
	f3 = 1 if text[index+1][1] in ending_punct else 0

	return f3

def feature_4(text):

	count = 0

	for word in text:
		
		if not word[0] == word[0].lower():
			count += 1
			
	f4 = 1 if count >= 2 else 0

	return f4

def feature_5(text, index):
	
	f5 = 1 if text[index+1][1] in possessive else 0

	return f5