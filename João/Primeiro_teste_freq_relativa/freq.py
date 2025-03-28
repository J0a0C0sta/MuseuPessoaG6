#!/usr/bin/env python
"""
Usage:
freq [OPTIONS] files
OPTIONS:
- l lemmas
- w words
- m 10 just the first 10 most common
- r relative freq per million
"""
import spacy
from jjcli import *
from collections import Counter

pln = spacy.load("pt_core_news_lg")

def printtab(f,m=1000000):
    for pal,oco in f.most_common(m):
        print(f"{pal}\t{oco}")

def main():
    cl = clfilter("wlrm:", doc=__doc__)
    freq = Counter()

    for txt in cl.text():
        dt = pln(txt)
        for tok in dt:
            if tok.pos_ == "SPACE" or tok.is_punct:
                continue
            freq[tok.lemma_] += 1
        
            print(tok.pos_, tok.lemma_)
    #print(freq.most_common(20))
    
    tot =freq.total()
    print(tot)
    freqrel = Counter()
    for pal,oco in freq.items():
        freqrel[pal] = oco/tot * 1000000
    #print(freqrel.most_common(20))
    num = int(cl.opt.get("-m", 1000000))
    if "-r" in cl.opt:
        printtab(freq, num)
    else:
        printtab(freqrel, num)
    
if __name__=="__main__":main()
