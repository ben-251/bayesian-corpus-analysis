from typing import List, Optional
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


def get_sentence_lengths(authors:List[Author]):
	for author in authors:
		for sentence in author.sentences:
			words = sentence.split(" ")
			author.sentence_lengths.append(len(words))

def get_max_length(authors:List[Author]):
	maximum_length = 0
	for author in authors:
		current_max = max(author.sentence_lengths)
		if current_max > maximum_length:
			maximum_length = current_max
	return maximum_length

def initialise_author_data(authors:List[Author]):
	get_sentence_lengths(authors)
	max_length = get_max_length(authors)
	for author in authors:
		author.sentence_length_counts = [0]*max_length
		for word_count in author.sentence_lengths:
			author.sentence_length_counts[word_count-1] += 1

if __name__ == "__main__":
	print("script")