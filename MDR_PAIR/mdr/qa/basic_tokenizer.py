# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Base tokenizer/tokens classes and utilities."""

import copy

class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


import regex
import logging

logger = logging.getLogger(__name__)


class RegexpTokenizer(Tokenizer):
    DIGIT = r'\p{Nd}+([:\.\,]\p{Nd}+)*'
    TITLE = (r'(dr|esq|hon|jr|mr|mrs|ms|prof|rev|sr|st|rt|messrs|mmes|msgr)'
             r'\.(?=\p{Z})')
    ABBRV = r'([\p{L}]\.){2,}(?=\p{Z}|$)'
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]++'
    HYPHEN = r'{A}([-\u058A\u2010\u2011]{A})+'.format(A=ALPHA_NUM)
    NEGATION = r"((?!n't)[\p{L}\p{N}\p{M}])++(?=n't)|n't"
    CONTRACTION1 = r"can(?=not\b)"
    CONTRACTION2 = r"'([tsdm]|re|ll|ve)\b"
    START_DQUOTE = r'(?<=[\p{Z}\(\[{<]|^)(``|["\u0093\u201C\u00AB])(?!\p{Z})'
    START_SQUOTE = r'(?<=[\p{Z}\(\[{<]|^)[\'\u0091\u2018\u201B\u2039](?!\p{Z})'
    END_DQUOTE = r'(?<!\p{Z})(\'\'|["\u0094\u201D\u00BB])'
    END_SQUOTE = r'(?<!\p{Z})[\'\u0092\u2019\u203A]'
    DASH = r'--|[\u0096\u0097\u2013\u2014\u2015]'
    ELLIPSES = r'\.\.\.|\u2026'
    PUNCT = r'\p{P}'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
            substitutions: if true, normalizes some token types (e.g. quotes).
        """
        self._regexp = regex.compile(
            '(?P<digit>%s)|(?P<title>%s)|(?P<abbr>%s)|(?P<neg>%s)|(?P<hyph>%s)|'
            '(?P<contr1>%s)|(?P<alphanum>%s)|(?P<contr2>%s)|(?P<sdquote>%s)|'
            '(?P<edquote>%s)|(?P<ssquote>%s)|(?P<esquote>%s)|(?P<dash>%s)|'
            '(?<ellipses>%s)|(?P<punct>%s)|(?P<nonws>%s)' %
            (self.DIGIT, self.TITLE, self.ABBRV, self.NEGATION, self.HYPHEN,
             self.CONTRACTION1, self.ALPHA_NUM, self.CONTRACTION2,
             self.START_DQUOTE, self.END_DQUOTE, self.START_SQUOTE,
             self.END_SQUOTE, self.DASH, self.ELLIPSES, self.PUNCT,
             self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()
        self.substitutions = kwargs.get('substitutions', True)

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Make normalizations for special token types
            if self.substitutions:
                groups = matches[i].groupdict()
                if groups['sdquote']:
                    token = "``"
                elif groups['edquote']:
                    token = "''"
                elif groups['ssquote']:
                    token = "`"
                elif groups['esquote']:
                    token = "'"
                elif groups['dash']:
                    token = '--'
                elif groups['ellipses']:
                    token = '...'

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)



