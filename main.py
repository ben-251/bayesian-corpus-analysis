from typing import Iterator, List, Optional
from functools import reduce
import numpy as np
import re

class Author:
	def __init__(self, name, file_name:Optional[str]=None):
		self.name = name
		self.file_path = f"data/{name}.txt" if file_name is None else file_name
		self.import_sentences()
		self.words = []
		self.sentence_length_counts = []

	def __str__(self) -> str:
		return self.name

	def import_sentences(self):
		self.sentences = []
		with open(self.file_path, "r") as f:
			for line in f:
				line = line.strip()
				split_line = re.split("[.!?]\\s+", line)
				split_line = [s.rstrip('.!?') for s in split_line]
				self.sentences.extend(split_line)

	def find_sentence_lengths(self) -> List[int]:
		sentence_lengths = []
		for sentence in self.sentences:
			words = sentence.split(" ")
			sentence_lengths.append(len(words))
		return sentence_lengths

	def max_sentence_length(self) -> int:
		return max(self.find_sentence_lengths())

class BayesianModeler:
	def __init__(self, authors:List[Author]) -> None:
		'''
		`authors`: a pre-initialised author group. 
		'''
		self.authors = authors
		self.initialise_author_data()
	
	def initialise_author_data(self) -> None:
		max_length = self.get_max_length()
		for author in self.authors:
			sentence_lengths = author.find_sentence_lengths()
			author.sentence_length_counts = [0]*max_length
			for word_count in sentence_lengths:
				author.sentence_length_counts[word_count-1] += 1

	def get_total_matching_sentences(self, length:int, author:Author):
		return author.sentence_length_counts[length-1]
	
	def get_max_length(self):
		return reduce(lambda current_max, author: max(current_max, author.max_sentence_length()), self.authors, 0) # thanks @voklen!

	def prior(self, identity):
		return 1/len(self.authors)

	def likelihood(self, length: int, author:Author) -> float:
		total_matching = self.get_total_matching_sentences(length, author)
		total_sentences = len(author.sentences)
		return total_matching / total_sentences

	def evidence(self, length: int):
		total = 0
		for author in self.authors:
			total += self.find_area(length, author)
		return total

	def find_area(self, length: int, author:Author):
		return self.likelihood(length, author) * self.prior(author)

	def find_posterior(self, sentence_length:int, author:Author):
		# P(I|L) = A(L,I) / sum ( A(L, I) )
		evidence = self.evidence(sentence_length)
		if evidence == 0:
			return 0 # there's no chance of this length occuring at all!
		return self.find_area(sentence_length, author) / self.evidence(sentence_length)



def main():
	Harryette = Author("Harryette", file_name="data/test_data/Harryette.txt")
	David = Author("David", file_name="data/test_data/David.txt")
	Will = Author("Shakespeare", file_name="data/test_data/shakespeare.txt")
	
	authors = [Harryette, David, Will]
	modeller = BayesianModeler(authors)
	target_author = Harryette
	target_length = 9

	m_length = modeller.get_max_length()
	for length in range(1, m_length+1):
		posteriors = list(map(lambda author: modeller.find_posterior(length, author), authors))
		if all([posterior == 0 for posterior in posteriors]):
			continue
		print(f"\n\nLENGTH OF {length}")
		for author, posterior in zip(authors, posteriors):			
			print(f"{author}: {round(posterior, 2)}")

if __name__ == "__main__":
	main()