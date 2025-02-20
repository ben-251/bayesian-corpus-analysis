from bentests import asserts, testGroup, test_all
from bentests.tester import TestResult
import numpy as np
from main import Author, BayesianModeler

class DataTests(testGroup):
	def test_sentence_import(self):
		author = Author("A", file_name="data/test_data/A.txt")
		asserts.assertEquals(
			author.sentences,
			["Go","Well, you may if you wish", "Maybe", "Since the monotone convergence theorem holds, then the sequence must converge", "Or so you think"]
		)

class AuthorInitialisationTests(testGroup):

	def test_sentence_lengths_for_one_author(self):
		author = Author("A", file_name="data/test_data/A.txt")
		sentence_lengths = author.find_sentence_lengths()
	
		asserts.assertEquals(
			sentence_lengths,
			[1,6,1, 11,4]
		)

	def test_max_sentence_length_one_author(self):
		authorA = Author("A", file_name="data/test_data/A.txt")
		length = authorA.max_sentence_length()

		asserts.assertEquals(length, 11)

	def test_get_max_length_mult_authors(self):
		authorA = Author("A", file_name="data/test_data/A.txt")
		authorB = Author("B", file_name="data/test_data/B.txt")
	
		modeler = BayesianModeler([authorA, authorB])
		max_length = modeler.get_max_length()
		
		asserts.assertEquals(
			max_length, 11
		)		
	
	def test_initialise_data_for_one(self):
		authorA = Author("A", file_name="data/test_data/A.txt")		
		modeler = BayesianModeler([authorA])

		asserts.assertEquals(
			modeler.authors[0].sentence_length_counts,
		  #           1  2  3  4  5  6  7  8  9  10 11
			np.array([2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1 ])
		)

	def test_initialise_data_mult_authors(self):
		authorC = Author("C", file_name="data/test_data/C.txt")	
		authorD = Author("D", file_name="data/test_data/D.txt")	
		modeler = BayesianModeler([authorC, authorD])

		asserts.assertEquals(
			modeler.authors[0].sentence_length_counts,
		              #  1  2  3  4 
			np.array([1, 1, 1, 0 ])
		)

class BayesianMathTests(testGroup):
	def test_prior(self):
		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)

		prior = modeller.prior(C)
		asserts.assertEquals(prior, 0.5)	

	def test_get_matching_total_zero(self):
		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		matching = modeller.get_total_matching_sentences(4, C)
		asserts.assertEquals(
			matching,
			0
		)

	def test_get_matching_total_nonzero(self):
		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		matching = modeller.get_total_matching_sentences(4, D)
		asserts.assertEquals(
			matching,
			1
		)

	def test_likelihood_zero(self):
		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		likelihood = modeller.likelihood(4, C)
		asserts.assertEquals(
			likelihood,
			0
		)

	def test_likelihood_nonzero(self):
		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		likelihood = modeller.likelihood(4, D)
		asserts.assertEquals(
			likelihood,
			0.5
		)

	def test_area_Zero(self):
		# we're expecting half of the likelihood,
		# since C and D are equally distributed a priori 
		# In this case, the likelihood is 0, so we should get 0
		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		area = modeller.find_area(3, D)
		asserts.assertEquals(
			area,
			0
		)

	def test_area_nonzero(self):
		# we're expecting half of the likelihood, since C and D are equally distributed a priori
		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		area = modeller.find_area(4, D)
		asserts.assertEquals(
			area,
			0.25
		)

	def test_evidence(self, skip=False):
		# A sum of the areas across all authors. we're expecting 0 + 0.25 = 0.25, 
		# where the 0.25 comes from 0.5 * 1/2, where 1/2 is the prior, 
		# where the prior is uniform.
		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)

		prior = modeller.evidence(4)
		asserts.assertEquals(prior, 0.25)		

	def test_basic_posterior(self, skip=False):
		# C: [1, 1, 1, 0]
		# D: [1, 0, 0, 1]

		# P(D|4) should be 1

		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		posterior = modeller.find_posterior(4, D)
		asserts.assertEquals(
			posterior,
			1.0
		)

	def test_zero_posterior(self, skip=False):
		# C: [1, 1, 1, 0]
		# D: [1, 0, 0, 1]

		# P(C|4) should be 0

		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		posterior = modeller.find_posterior(4, C)
		asserts.assertEquals(
			posterior,
			0
		)

	def test_messier_posterior(self, skip=False):
		# C: [1, 1, 1, 0]
		# D: [1, 0, 0, 1]

		# P(D|1) should be 1/2, since the prior is 1/2, 
		# and the likelihood is exactly the same for both C and D, so 
		# the formula ends up as (likelihood * prior) / likelihood + likelihood 
		# = 1/2 likelihood / 2likelihood = 1/4 likelihood??? uh mistake in logic somewhre

		# wait so for Posterior(1, D), the likelihood is 0.5 (1 out of 2 is a 1)
		# the prior is again 0.5, so the numerator is 0.25
		# p(L) is going to be that top number, 0.25, + (likelihood for C which is 1/3 times prior which is 0.5)
		# so 0.25/(1/4 + 1/6) = 0.25/(6/24 + 4/24) = 1/4 * 24/10 = 24/40 = 0.6! woah! that's clean but also that means D is more likely

		# Which makes sense because D has less options, so even though they both have one 1 word sentence, 
		# D is more likely to have a one-word sentence than C

		C = Author("C", file_name="data/test_data/C.txt")
		D = Author("D", file_name="data/test_data/D.txt")
		
		authors = [C,D]
		modeller = BayesianModeler(authors)
		posterior = modeller.find_posterior(1, D)
		asserts.assertAlmostEquals(
			posterior,
			0.6
		)


test_all(
	DataTests,
	BayesianMathTests,
	AuthorInitialisationTests,
	skip_passes=True
)