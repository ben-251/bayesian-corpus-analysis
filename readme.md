Use a bayesian model to predict person by word/sentence lengths?
Data would be the average word/sentence length, or maybe all the word lengths.
Hypothesis would be the identity.

We can have a table like:

|     | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | …   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A   | 23  | 26  | 23  | 2   | 234 | 45  | 23  | 24  | 2   | …   |
| B   | 34  | 46  | 53  | 22  | 23  | 42  | 223 | 63  | 12  | …   |

So we see how long each sentence length is distributed for each person, and then can predict who that person is from the sentence length later on.

# Bayesian logic
Let $L$ be a sentence length, and $I$ be the identity of the user. then:
$$
P(I|L) = \frac{P(L|I)P(I)}{P(L)}
$$
## Prior
the probability of an identity occurring at all, which we’ll make uniform for the time being.
## Likelihood
the probability of the sentence length occurring for some identity, l: $$P(L|I)=\frac{\#(L)}{\sum L}$$
## Data
the sum of the probabilities of getting any sentence length:
$$P(L)= \sum\limits_{I_{n}} P(L|I_{n})\,P(I_{n})$$
![[Sentence length prediction 2025-02-17 13.58.55.excalidraw]]
The area of each bars is just a likelihood-prior product computed on different hypotheses (Identities). For example the probability of Shakespeare producing a specific sentence length, then Dickinson, etc.

# (pseudo)-Code
## Bayesian Logic
The likelihood function should be generalised to be reused in the denominator and numerator:
```python
# identity table defined at some point

def find_posterior(length):
	# P(I|L) = A(L,I) / sum ( A(L, I) )
	return find_area(length, identity) / sum(map(find_area, identities))

def find_area(length, identity):
	return likelihood(length, identity) * prior(identity)

def likelihood(length, identity):
	total_matching = get_total_matching_sentences(length, identity)
	total_sentences = len(identity).sentences
	return total_matching / total_sentences

def get_total_matching_sentences(length, identity)
	total_matching = 0
	for sentence in identity.sentences:
		if sentence.length == length:
			total_matching += 1
	return total_matching	

def prior(identity):
	return 1/len(identities) # later modify this to be non-uniform.
```

## Table generation
I still need to define the way we generate the table from a corpus. At a basic level, it’d look like this:
- repeat for each identity:
	- import text for identity
	- go through text:
		- each sentence:
			- add to a list of sentence lengths (`identity.sentences[n-1]` is equal to the number of sentences written by `identity` of length n)
in code:

```python
for author in authors:
	sentences = import_sentences(author)
	for sentence in sentences:
		words = sentence.split(" ")
		word_count = len(words)
		author.sentences[word_count-1] += 1
```

oh wait this won’t work. for example consider this example text:
```txt
Don't! No, do consider this text.
```
This has two sentences: one with length of 1, and another with a length of 5. but the program doesn’t have a sense for what those positions *mean* yet, since it doesn’t have `[0, 0, 0, 0, 0]` waiting to be filled. Thus, we require a method for getting the maximum sentence length required for consideration, across all authors, before we even start:

```python
def extract_sentences(authors):
	for author in authors:
		for sentence in import_sentences(author):
			words = sentence.split(" ")
			author.words.append(words)
			author.sentence_lengths.append(len(words))

def get_max_length(authors)
	maximum_length = 0
	for author in authors:
		current_max = max(author.sentence_lengths)
		if current_max > maximum_length:
			maximum_length == current_max
	return maximum_length

def initialise_author_data(authors):
	for author in authors:
		for word_count in author.sentence_lengths:
			author.sentence_length_counts[word_count-1] += 1
		
```
# Evaluation and such
I predict that this model will do worse as more identities are added, since it’s easy to tell apart an overwriter from an underwriter, but not a slight-overwriter from a slightly-less-overwriter, where “-over-” and “-under-” refer to the expected sentence lengths from an identity.

Second flaw is that this will be *severely*  “overfitted”. By that I mean that if someone happens to write a lot of sentences specifically at, say 11 and 13 words, then given a length of, say 12, the score will be quite low, even though that should be just as lik— actually no this is good, cuz it could pick on otherwise uncatcahble patterns