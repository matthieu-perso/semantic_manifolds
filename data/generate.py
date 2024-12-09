import nltk
from nltk.corpus import wordnet as wn
from data.embed import get_embeddings_openai, get_embeddings_sentence_transformer

nltk.download('wordnet')

class HyponymGenerator:
    def __init__(self, words):
        self.words = words
        self.synsets_dict = self._get_synsets_dict()
        self.all_hyponyms = self._get_all_hyponyms()
        self.clean_hyponyms = self._get_clean_hyponyms()
        self.sentences = self._create_sentences_for_hyponyms()
        self.sentences_list = list(self.sentences.values())

    def _get_synsets_dict(self):
        """Get the synsets for the words in a dictionary."""
        return {word: wn.synsets(word)[0] for word in self.words if wn.synsets(word)}

    def _get_all_hyponyms(self):
        """Retrieve all hyponyms recursively for each synset."""
        return {word: self._get_all_hyponyms_clean(synset) for word, synset in self.synsets_dict.items()}

    def _get_all_hyponyms_clean(self, synset):
        """Recursively retrieve all hyponyms of a given synset and format their names."""
        hyponyms = synset.hyponyms()
        all_hyponyms = set(hyponyms)

        for hyponym in hyponyms:
            all_hyponyms.update(self._get_all_hyponyms_clean(hyponym))
        return all_hyponyms

    def _get_clean_hyponyms(self):
        """Get cleaned names of all hyponyms."""
        return {
            word: [
                self._clean_name(lemma.name()) 
                for synset in synset_set 
                for lemma in synset.lemmas()
            ]
            for word, synset_set in self.all_hyponyms.items()
        }

    def _clean_name(self, name):
        """Clean lemma names (remove underscores, capitalize if necessary)."""
        return name.replace("_", " ").capitalize()

    def _create_sentences_for_hyponyms(self):
        """Create sentences for each clean hyponym of each word."""
        sentences = {}
        for word, hyponyms in self.clean_hyponyms.items():
            for hyponym in hyponyms:
                sentences[hyponym] = create_sentence(hyponym)
        return sentences

def create_sentence(word):
    """Create a sentence for a given word."""
    article = 'an' if word[0].lower() in 'aeiou' else 'a'
    return f"Someone has {article} {word}"

def main(words, model="sentence_transformer"):
    hyponym_generator = HyponymGenerator(words)
    print(len(hyponym_generator.sentences_list))
    print(hyponym_generator.sentences_list)
    if model == "openai":
        embeddings = get_embeddings_openai(hyponym_generator.sentences_list)
    else:
        embeddings = get_embeddings_sentence_transformer(hyponym_generator.sentences_list)
    print(len(embeddings))
    return embeddings

if __name__ == "__main__":
    main(["cat", "dog", "car", "house"])