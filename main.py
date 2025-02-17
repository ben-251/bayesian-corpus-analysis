from typing import Iterator, List, Optional
import re

class Author:
	def __init__(self, name, file_name:Optional[str]=None):
		self.name = name
		self.file_path = f"data/{name}.txt" if file_name is None else file_name
		self.import_sentences()
		self.words = []
		self.sentence_lengths = []
		self.sentence_length_counts = []

	def import_sentences(self):
		self.sentences = []
		with open(self.file_path, "r") as f:
			for line in f:
				line = line.strip()
				split_line = re.split("[.!?]\\s+", line)
				split_line = [s.rstrip('.!?') for s in split_line]
				self.sentences.extend(split_line)


class AuthorGroup:
	def __init__(self, authors:List[Author]) -> None:
		self.authors = authors
		self.initialise_author_data()

	def __getitem__(self, index: int) -> Author:
		return self.authors[index]

	def __iter__(self) -> Iterator[Author]:
		return iter(self.authors)

	def __len__(self) -> int:
		return len(self.authors)

	def get_sentence_lengths(self) -> None:
		for author in self.authors:
			author.sentence_lengths = []
			for sentence in author.sentences:
				words = sentence.split(" ")
				author.sentence_lengths.append(len(words))

	def get_max_length(self) -> int:
		maximum_length = 0
		for author in self.authors:
			current_max = max(author.sentence_lengths)
			if current_max > maximum_length:
				maximum_length = current_max
		return maximum_length

	def initialise_author_data(self) -> None:
		self.get_sentence_lengths()
		max_length = self.get_max_length()
		for author in self.authors:
			author.sentence_length_counts = [0]*max_length
			for word_count in author.sentence_lengths:
				author.sentence_length_counts[word_count-1] += 1


class BayesianModeler:
	def __init__(self, authors:AuthorGroup) -> None:
		'''
		`authors`: a pre-initialised author group. 
		'''
		self.authors = authors
	
	def get_total_matching_sentences(self, length:int, author:Author):
		total_matching = 0
		for sentence in author.sentences:
			if sentence.length == length:
				total_matching += 1
		return total_matching	

	def prior(self, identity):
		return 1/len(self.authors)

	def likelihood(self, length: int, author:Author):
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
		return self.find_area(sentence_length, author) / self.evidence(sentence_length)


if __name__ == "__main__":
	print("script")