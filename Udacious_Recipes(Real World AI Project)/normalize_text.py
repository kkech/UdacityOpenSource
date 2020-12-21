import ftfy as ft
from nltk.tokenize import sent_tokenize
import re
def to_lowercase(words):
  
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def normalize(sent):
    
    try:
        split_sent=ft.fix_text(sent).split()
    except Exception as e:
        return ""
    words=to_lowercase(split_sent)
    words=" ".join(words)
    words=strip_non_ascii(re.sub('\s+',' ',re.sub(r'[\n\r]+','',words))).strip('[!-[]{};:|,<>?@#$%^&*_~]')
    return re.sub('\s+',' ',re.sub(r'[^\[\]\w\s.{}&()/]','',words))

def tokenize_sents(sents):
    
    sents_tokenized=[]
    for text in sents:
        sents_tokenized+=sent_tokenize(text)

    return sents_tokenized

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)