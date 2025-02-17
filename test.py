from bentests import asserts, testGroup, test_all
from main import Author, get_sentence_lengths, get_max_length, initialise_author_data

class DataTests(testGroup):
	def test_sentence_import(self):
		author = Author("A", file_name="data/test_data/A.txt")
		asserts.assertEquals(
			author.sentences,
			["Go","Well, you may if you wish", "Maybe", "Since the monotone convergence theorem holds, then the sequence must converge", "Or so you think"]
		)

	def test_sentence_lengths_for_one_author(self):
		author = Author("A", file_name="data/test_data/A.txt")
		get_sentence_lengths([author])
	
		asserts.assertEquals(
			author.sentence_lengths,
			[1,6,1, 11,4]
		)

	def test_sentence_lengths_for_two_authors(self):
		authorA = Author("A", file_name="data/test_data/A.txt")
		authorB = Author("B", file_name="data/test_data/B.txt")
		
		get_sentence_lengths([authorA, authorB])
	
		asserts.assertEquals(
			(authorA.sentence_lengths, authorB.sentence_lengths),
			([1,6,1,11,4], [8, 6, 4, 2])
		)

	def test_get_max_length(self):
		authorA = Author("A", file_name="data/test_data/A.txt")
		authorB = Author("B", file_name="data/test_data/B.txt")
		
		authors = [authorA, authorB]

		get_sentence_lengths(authors)
		max_length = get_max_length(authors)
	
		asserts.assertEquals(
			max_length, 11
		)		
	
	def test_initialise_data_for_one(self):
		authorA = Author("A", file_name="data/test_data/A.txt")		
		authors = [authorA]

		initialise_author_data(authors)

		asserts.assertEquals(
			authorA.sentence_length_counts,
		  #  1  2  3  4  5  6  7  8  9  10 11
			[2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1 ]
		)

class BayesianTests(testGroup):
	...

test_all(
	DataTests
)