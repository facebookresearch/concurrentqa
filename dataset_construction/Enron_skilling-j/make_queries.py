import os
import csv
import ujson
import json
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np
import time
import ast
import random
from collections import Counter, defaultdict, OrderedDict

INBOX = "skilling-j"
def add_entry(q="", idx="", answer=[], sp1={}, sp2={}, typ="", domain=[]):
    entry = {
        'question': q,
        '_id': idx,
        'answer': answer,
        'sp': [sp1, sp2],
        'type': typ, # comparison or bridge
        'domain': domain, # 0, 1
    }
    
    original_entry = {
        '_id':idx, 
        'answer':answer[0], 
        'question':q, 
        'supporting_facts':[[sp1['title'], 0], [sp2['title'], 0]],
        'context':[], 
        'type':typ, 
        'level':'hard'
    }
    
    return entry, original_entry


local_global_queries = []
original_queries= []

with open("/checkpoint/simarora/mdr/wikipassages2sents.json") as f:
    wikipassages2sents = json.load(f)

with open(f"/checkpoint/simarora/PersonalDatasets/Enron/qa_runs/{INBOX}/subject2sents.json") as f:
    subject2sents = json.load(f)

# question
entry, original_entry = add_entry(q="The company providing natural gas transmission between Western US states such as New Mexico and Texas is helping support credit lines worth how much money?",
                                  idx="01P",
                                  answer=["$1 billion"], 
                                  sp1={'title': 'Transwestern Pipeline', 
                                       "sents": wikipassages2sents['Transwestern Pipeline'],
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL daily update', 
                                       "sents":subject2sents['PERSONAL daily update'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())  


# question
entry, original_entry = add_entry(q="The Texas Pacific Group is known for investments in what motorcycle company?",
                                  idx="02P",
                                  answer=["Ducati"], 
                                  sp1={'title': 'TPG Capital', 
                                       "sents": wikipassages2sents['TPG Capital'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL re: jeff skilling for tpg ceo conference', 
                                       "sents":subject2sents['PERSONAL re: jeff skilling for tpg ceo conference'], 
                                       'sp_sent_ids': [5]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="What type of partnership does Enron want to form with the media entertainment conglomerate, which is based in Burbank, California?",
                                  idx="03P",
                                  answer=["broadband"], 
                                  sp1={'title':'The Walt Disney Company', 
                                       "sents":wikipassages2sents['The Walt Disney Company'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL re: broadband partnership with walt disney corp', 
                                       "sents":subject2sents['PERSONAL re: broadband partnership with walt disney corp'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="How many times per year can the exam that the Enron candidate from Princeton took be taken?",
                                  idx="04P",
                                  answer=["five", "5"], 
                                  sp1={'title': 'PERSONAL enron candidate', 
                                       "sents":subject2sents['PERSONAL enron candidate'], 
                                       'sp_sent_ids': [0, 1, 2]}, 
                                  sp2={'title':'Graduate Management Admission Test', 
                                       "sents":wikipassages2sents['Graduate Management Admission Test'], 
                                       'sp_sent_ids': [0, 4]},
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="What is the current growth rate of the Fortune 500 company originally called Metropolitan Pathology Labs?",
                                  idx="05",
                                  answer=["50%", '50% per year'], 
                                  sp1={'title':'Quest Diagnostics', 
                                       "sents":wikipassages2sents['Quest Diagnostics'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL mischer-interfit health', 
                                       "sents":subject2sents['PERSONAL mischer-interfit health'], 
                                       'sp_sent_ids': [20]}, 
                                  typ="bridge", 
                                  domain=[0,1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="How much is the Atlanta based contact technology company offering per newly referred customer?",
                                  idx="06P",
                                  answer=["$1,000.00", "$1,000.00 plus"], 
                                  sp1={'title':'Noble Systems Corporation', 
                                       "sents":wikipassages2sents['Noble Systems Corporation'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL noble systems executive update + opportunities for you', 
                                       "sents":subject2sents['PERSONAL noble systems executive update + opportunities for you'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[0,1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="Jim Kelly is the CEO participating in the Mastermind Keynote Interview. How many customers does the company Jim Kelly is from have?",
                                  idx="07P",
                                  answer=["7.9 million", "more than 7.9 million"], 
                                  sp1={'title':'United Parcel Service', 
                                       "sents":wikipassages2sents['United Parcel Service'], 
                                       'sp_sent_ids': [0, 3]}, 
                                  sp2={'title':'PERSONAL re: invitation', 
                                       "sents":subject2sents['PERSONAL re: invitation'], 
                                       'sp_sent_ids': [5, 8]}, 
                                  typ="bridge", 
                                  domain=[1,0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Does Enron earn money from projects conducted at the U.S. Navy facility located 100 km from Manila Bay?",
                                  idx="08P",
                                  answer=["yes"], 
                                  sp1={'title':'Subic Bay', 
                                       "sents":wikipassages2sents['Subic Bay'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL end of an era', 
                                       "sents":subject2sents['PERSONAL end of an era'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="When was the dean just elected to the Enron Board of Directors born?",
                                  idx="09P",
                                  answer=["May 30, 1946", "1946"], 
                                  sp2={'title':'William Powers, Jr.', 
                                       "sents":wikipassages2sents['William Powers, Jr.'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp1={'title':'PERSONAL enron update0', 
                                       "sents":subject2sents['PERSONAL enron update0'], 
                                       'sp_sent_ids': [5]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="After Enron announced the $1 billion credit line, it’s S&P rating was the same as Hyundai Haesang’s S&P rating?",
                                  idx="10P",
                                  answer=["yes"], 
                                  sp1={'title':'Hyundai Marine &amp; Fire Insurance', 
                                       "sents":wikipassages2sents['Hyundai Marine &amp; Fire Insurance'], 
                                       'sp_sent_ids': [0, 5]}, 
                                  sp2={'title':'PERSONAL enron update', 
                                       "sents":subject2sents['PERSONAL enron update'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Did any of the selected guests who will be at the Insight Capital May 15th dinner work for Goldman Sachs?",
                                  idx="11P",
                                  answer=["yes"], 
                                  sp2={'title':'Robert Rubin', 
                                       "sents":wikipassages2sents['Robert Rubin'], 
                                       'sp_sent_ids': [0, 2]}, 
                                  sp1={'title':'PERSONAL re: telephone call with jerry murdock15', 
                                       "sents":subject2sents['PERSONAL re: telephone call with jerry murdock15'], 
                                       'sp_sent_ids': [7, 8, 10]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Are any of my fellow invitees for the Insight Capital dinner chemical engineers?",
                                  idx="12P",
                                  answer=["yes"], 
                                  sp2={'title':'Jack Welch', 
                                       "sents":wikipassages2sents['Jack Welch'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp1={'title':'PERSONAL re: telephone call with jerry murdock15', 
                                       "sents":subject2sents['PERSONAL re: telephone call with jerry murdock15'], 
                                       'sp_sent_ids': [8, 10]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="On what day is the upcoming luncheon with the co-founder of Netscape, Hewlett-Packard, and Mosaic?",
                                  idx="13P",
                                  answer=["Friday, June 22nd"], 
                                  sp1={'title':'Marc Andreessen', 
                                       "sents":wikipassages2sents['Marc Andreessen'], 
                                       'sp_sent_ids': [0, 1, 2, 3]}, 
                                  sp2={'title':'PERSONAL marc andreessen in dallas 6/22...0', 
                                       "sents":subject2sents['PERSONAL marc andreessen in dallas 6/22...0'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Will there be lunch at the event for the Cambridge Ivy League institution?",
                                  idx="14P",
                                  answer=["no"], # there will be dinner... 
                                  sp1={'title':'Harvard University', 
                                       "sents":wikipassages2sents['Harvard University'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL re: harvard forum 05/18/01 - second invite1', 
                                       "sents":subject2sents['PERSONAL re: harvard forum 05/18/01 - second invite1'], 
                                       'sp_sent_ids': [5,6]}, 
                                  typ="bridge", 
                                  domain=[0,1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Where is the headquarters for the association with an upcoming advertisement in 'On Scene' magazine?",
                                  idx="15P",
                                  answer=["Fairfax, Virginia"], 
                                  sp2={'title':"International Association of Fire Chiefs" , 
                                       "sents":wikipassages2sents["International Association of Fire Chiefs"], 
                                       'sp_sent_ids': [0, 3]}, 
                                  sp1={'title':'PERSONAL the list, legal opinion & other news', 
                                       "sents":subject2sents['PERSONAL the list, legal opinion & other news'], 
                                       'sp_sent_ids': [4, 5]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 



# question 
entry, original_entry = add_entry(q="When visiting the affluent summer colony located south of Cape Cod, Jeff mentioned mentioned he wanted to walk through what?",
                                  idx="16P",
                                  answer=["house", "our house"], 
                                  sp1={'title':"Martha\'s Vineyard", 
                                       "sents":wikipassages2sents["Martha\'s Vineyard"], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title': 'PERSONAL re: christmas gathering2', 
                                       "sents":subject2sents['PERSONAL re: christmas gathering2'], 
                                       'sp_sent_ids': [4]}, 
                                  typ="bridge", 
                                  domain=[0,1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="When was the speaker for Commercial and Political Perspectives luncheon born?",
                                  idx="17P",
                                  answer=["1956"], 
                                  sp2={'title':'Bernard Harris (disambiguation)', 
                                       "sents":wikipassages2sents['Bernard Harris (disambiguation)'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL re: hbs april 25 luncheon - reminder1', 
                                       "sents":subject2sents['PERSONAL re: hbs april 25 luncheon - reminder1'], 
                                       'sp_sent_ids': [4,5,6,7]}, 
                                  typ="bridge", 
                                  domain=[1,0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="At the golf tournament on Fazio Course, the New York eCommerce Association will dedicate proceeds to an organization affiliated with which International Federation?",
                                  idx="18P",
                                  answer=["International Federation of Red Cross and Red Crescent Societies"], 
                                  sp1={'title':'American Red Cross', 
                                       "sents":wikipassages2sents['American Red Cross'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL upcoming golf tournament and venture capital conference', 
                                       "sents":subject2sents['PERSONAL upcoming golf tournament and venture capital conference'], 
                                       'sp_sent_ids': [0, 2]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="For how many days is Cheryl going to the city historically known as Constantinople and Byzantium?",
                                  idx="19P",
                                  answer=["3"], 
                                  sp1={'title':'Istanbul', 
                                       "sents":wikipassages2sents['Istanbul'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL a trip to turkey', 
                                       "sents":subject2sents['PERSONAL a trip to turkey'], 
                                       'sp_sent_ids': [2, 7]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The 2002 television film The Junction Boys was based on a book by an author who signed a book for who?",
                                  idx="20P",
                                  answer=["Jim Bavouset"], 
                                  sp1={'title':'The Junction Boys (film)', 
                                       "sents":wikipassages2sents['The Junction Boys (film)'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL re: [smu-betas] 76ers - it\'s time to hump it3', 
                                       "sents":subject2sents['PERSONAL re: [smu-betas] 76ers - it\'s time to hump it3'], 
                                       'sp_sent_ids': [7]}, 
                                  typ="bridge", 
                                  domain=[0,1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The popular author in Beta suggested which dinner location for Thursday of homecoming weekend?",
                                  idx="21P",
                                  answer=["The Double Tree and Central Expressway"], 
                                  sp1={'title':'PERSONAL re: [smu-betas] 76ers - it\'s time to hump it4', 
                                       "sents":subject2sents['PERSONAL re: [smu-betas] 76ers - it\'s time to hump it4'], 
                                       'sp_sent_ids': [3]}, 
                                  sp2={'title':'PERSONAL -kai-3', 
                                       "sents":subject2sents['PERSONAL -kai-3'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The Beta who elaborated on another dinner idea for Thursday of HC is also organizing an outing for which sport?",
                                  idx="22P",
                                  answer=["golf"], 
                                  sp1={'title':'PERSONAL -kai-16', 
                                       "sents":subject2sents['PERSONAL -kai-16'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL -kai-19', 
                                       "sents":subject2sents['PERSONAL -kai-19'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Is the guy who Chuck Paul added to the Beta list arriving to HC weekend with his son?",
                                  idx="23P",
                                  answer=["no"], 
                                  sp1={'title':'PERSONAL -kai-3', 
                                       "sents":subject2sents['PERSONAL -kai-3'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL -kai-5', 
                                       "sents":subject2sents['PERSONAL -kai-5'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Is the newspaper which blasted Mayor Brown the largest in the United States by Sunday circulation?",
                                  idx="24P",
                                  answer=["no"], 
                                  sp1={'title':'PERSONAL chronicle article on hfd', 
                                       "sents":subject2sents['PERSONAL chronicle article on hfd'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'Houston Chronicle', 
                                       "sents":wikipassages2sents['Houston Chronicle'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The substituted CEO of Mexican Petroleums is close to which politician?",
                                  idx="25P",
                                  answer=["Francisco Labastida"], 
                                  sp1={'title':'Pemex', 
                                       "sents":wikipassages2sents['Pemex'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL change at pemex', 
                                       "sents":subject2sents["PERSONAL change at pemex"], 
                                       'sp_sent_ids': [0, 2]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Did Rogelio's close friend who was running for presidency win the 2000 presidential election?",
                                  idx="26P",
                                  answer=["no"], 
                                  sp2={'title':'Francisco Labastida', 
                                       "sents":wikipassages2sents['Francisco Labastida'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL change at pemex', 
                                       "sents":subject2sents['PERSONAL change at pemex'], 
                                       'sp_sent_ids': [2]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The agency that that \"manages pension and health benefits for millions of California employees\" owns how many Enron shares?",
                                  idx="27P",
                                  answer=["2.6 million"], 
                                  sp2={'title':'PERSONAL jedi ii', 
                                       "sents":subject2sents['PERSONAL jedi ii'], 
                                       'sp_sent_ids': [1]}, 
                                  sp1={'title':'CalPERS', 
                                       "sents":wikipassages2sents['CalPERS'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0,1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Do Tonn Ostergard from YPO and Jim Dent live in the same state?",
                                  idx="28P",
                                  answer=["no"], 
                                  sp1={'title':'PERSONAL parent child mountain adventure, july 21-25, 2001', 
                                       "sents":subject2sents['PERSONAL parent child mountain adventure, july 21-25, 2001'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL re: [smu-betas] 76ers - it\'s time to hump it4', 
                                       "sents":subject2sents['PERSONAL re: [smu-betas] 76ers - it\'s time to hump it4'], 
                                       'sp_sent_ids': [3, 6]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Who supposedly made up lies about the player who won 47 straight games in college football?",
                                  idx="29P",
                                  answer=["Dent"], 
                                  sp1={'title':'Bud Wilkinson', 
                                       "sents":wikipassages2sents['Bud Wilkinson'], 
                                       'sp_sent_ids': [0, 2]}, 
                                  sp2={'title':'PERSONAL [smu-betas] dent pisses on bud wilkinson\'s grave', 
                                       "sents":subject2sents['PERSONAL [smu-betas] dent pisses on bud wilkinson\'s grave'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="In what year was the athletic director of the Big 12 Conference Sooners born?",
                                  idx="30P",
                                  answer=["1957"], 
                                  sp1={'title':"Oklahoma Sooners", 
                                       "sents":wikipassages2sents["Oklahoma Sooners"], 
                                       'sp_sent_ids': [0, 2, 3]}, 
                                  sp2={'title':'Joe Castiglione (athletic director)', 
                                       "sents":wikipassages2sents['Joe Castiglione (athletic director)'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Who wrote the song that Bill Miller will need to sing in pink skivvies?",
                                  idx="31P",
                                  answer=["Arthur M Alden"], 
                                  sp2={'title':'Boomer Sooner', 
                                       "sents":wikipassages2sents['Boomer Sooner'], 
                                       'sp_sent_ids': [0, 1, 2]}, 
                                  sp1={'title':'PERSONAL re: [smu-betas] dent\'s wrath', 
                                       "sents":subject2sents['PERSONAL re: [smu-betas] dent\'s wrath'], 
                                       'sp_sent_ids': [1, 3]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The fight song of the New Haven based Ivy League University is borrowed from a song written in which year?",
                                  idx="32P",
                                  answer=["1898"], 
                                  sp1={'title':'Yale University', 
                                       "sents":wikipassages2sents['Yale University'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':"Boomer Sooner", 
                                       "sents":wikipassages2sents["Boomer Sooner"], 
                                       'sp_sent_ids': [3]}, 
                                  typ="bridge", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="For the Astros vs. Giants game at Enron Field, the Enron sign will feature the logo of a nonprofit organization that has how many offices throughout the country?",
                                  idx="33P",
                                  answer=["1,200"], 
                                  sp2={'title':'United Way of America', 
                                       "sents":wikipassages2sents['United Way of America'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL enron and united way\'s continued partnership', 
                                       "sents":subject2sents['PERSONAL enron and united way\'s continued partnership'], 
                                       'sp_sent_ids': [4]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="All communications between Enron and LJM must be preserved under an Act created in what year?",
                                  idx="34P",
                                  answer=["1995"], 
                                  sp2={'title':'Private Securities Litigation Reform Act', 
                                       "sents":wikipassages2sents['Private Securities Litigation Reform Act'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL important announcement regarding document preservation', 
                                       "sents":subject2sents['PERSONAL important announcement regarding document preservation'], 
                                       'sp_sent_ids': [0, 2]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="What is the approximate population of the city Mark visited in Georgia?",
                                  idx="35P",
                                  answer=["1.5 million people"], 
                                  sp2={'title':'Tbilisi', 
                                       "sents":wikipassages2sents['Tbilisi'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'<6289674.1075845512831.JavaMail.evans@thyme>', 
                                       "sents":subject2sents['<6289674.1075845512831.JavaMail.evans@thyme>'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Was the singer of Baby One More Time requested as a speaker for the Enron eSpeak event?",
                                  idx="36P",
                                  answer=["yes"], 
                                  sp1={'title':"Britney Spears", 
                                       "sents":wikipassages2sents["Britney Spears"], 
                                       'sp_sent_ids': [0, 2, 4]}, 
                                  sp2={'title':'PERSONAL espeak survey: the results are in!0', 
                                       "sents":subject2sents['PERSONAL espeak survey: the results are in!0'], 
                                       'sp_sent_ids': [0, 2, 3]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Do Steve Spar of ESX Engineering and Bruce Wrobel of EnCom hold the same position at their companies?",
                                  idx="37P",
                                  answer=["yes"], # CEO
                                  sp1={'title':'PERSONAL status report on enron\'s investment in encom0', 
                                       "sents":subject2sents['PERSONAL status report on enron\'s investment in encom0'], 
                                       'sp_sent_ids': [0, 2]}, 
                                  sp2={'title':'PERSONAL referred by jeff spar (mck - ny)', 
                                       "sents":subject2sents['PERSONAL referred by jeff spar (mck - ny)'], 
                                       'sp_sent_ids': [10]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="What is the nearest hospital to the area where Reliance is developing a liquid fuel fired power plant?",
                                  idx="38P",
                                  answer=["Dhirubhai Ambani Hospital"], 
                                  sp2={'title':'Patalganga', 
                                       "sents":wikipassages2sents['Patalganga'], 
                                       'sp_sent_ids': [3]}, 
                                  sp1={'title':'PERSONAL re: maharashtra plants', 
                                       "sents":subject2sents['PERSONAL re: maharashtra plants'], 
                                       'sp_sent_ids': [3, 4]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="Are Dabhol and Patalganga located in the same state in India?",
                                  idx="39P",
                                  answer=["yes"], 
                                  sp1={'title':'Patalganga', 
                                       "sents":wikipassages2sents['Patalganga'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'Dabhol', 
                                       "sents":wikipassages2sents['Dabhol'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="What is the name of the online magazine of the Oil Patch District Federal Reserve Bank?",
                                  idx="40P",
                                  answer=["e-Perspectives"], 
                                  sp1={'title':'Federal Reserve Bank of Dallas', 
                                       "sents":wikipassages2sents['Federal Reserve Bank of Dallas'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL welcome to the federal reserve bank of dallas community affairs\n announcement list', 
                                       "sents":subject2sents['PERSONAL welcome to the federal reserve bank of dallas community affairs\n announcement list'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="The meeting with Merrill Lynch about the Houston based water services company is on what date?",
                                  idx="41P",
                                  answer=["Monday February 28"], 
                                  sp1={'title':"Azurix", 
                                       "sents":wikipassages2sents["Azurix"], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL re: azurix investment banking meeting0', 
                                       "sents":subject2sents['PERSONAL re: azurix investment banking meeting0'], 
                                       'sp_sent_ids': [3, 6]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="Are Azurix and Enron headquartered in the same city?",
                                  idx="42P",
                                  answer=["yes"], 
                                  sp1={'title':'Azurix', 
                                       "sents":wikipassages2sents['Azurix'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'Enron', 
                                       "sents":wikipassages2sents['Enron'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="Skilling booked which suite to watch the MLB game for the team based in Houston?",
                                  idx="43P",
                                  answer=["Drayton McLane's"], 
                                  sp1={'title':"Houston Astros", 
                                       "sents":wikipassages2sents["Houston Astros"], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL astros game', 
                                       "sents":subject2sents['PERSONAL astros game'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="A professor at the Pasadena based university co-founded which coporation that's selling a software framework to Enron?",
                                  idx="44P",
                                  answer=["iSpheres"], 
                                  sp1={'title':"California Institute of Technology", 
                                       "sents":wikipassages2sents["California Institute of Technology"], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL re: advanced arbitrage enabling technology out of caltech', 
                                       "sents":subject2sents['PERSONAL re: advanced arbitrage enabling technology out of caltech'], 
                                       'sp_sent_ids': [5]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="To support the longest-serving Republican senator in Montana history, to whom should the checks be made payable?",
                                  idx="45P",
                                  answer=["Friends of Conrad Burns"], 
                                  sp1={'title':"Conrad Burns", 
                                       "sents":wikipassages2sents["Conrad Burns"], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL senator conrad burns contribution', 
                                       "sents":subject2sents['PERSONAL senator conrad burns contribution'], 
                                       'sp_sent_ids': [0, 1, 2]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="The partner Enron is in litigation with in Federal court proposed to acquire what Corp from Enron?",
                                  idx="46P",
                                  answer=["Enron Renewable Energy Corp"], 
                                  sp1={'title':'PERSONAL important announcement regarding document preservation', 
                                       "sents":subject2sents['PERSONAL important announcement regarding document preservation'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL project aura; draft disclosures re ljm2', 
                                       "sents":subject2sents['PERSONAL project aura; draft disclosures re ljm2'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Did Rebecca Carter's replacement for corporate secretary receive her bachelor's degree at a university located in Houston?",
                                  idx="47P",
                                  answer=["no"], # College Station
                                  sp2={'title':"Texas A&amp;M University", 
                                       "sents":wikipassages2sents["Texas A&amp;M University"], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL enron board elects new corporate secretary', 
                                       "sents":subject2sents['PERSONAL enron board elects new corporate secretary'], 
                                       'sp_sent_ids': [0, 1, 6]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The Austin based conservative think tank that's part of the State Policy Network is similar to which foundation in D.C.?",
                                  idx="48P",
                                  answer=["Heritage Foundation"], 
                                  sp1={'title':'Texas Public Policy Foundation', 
                                       "sents":wikipassages2sents['Texas Public Policy Foundation'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL texas public policy foundation dinner - thursday, february 15', 
                                       "sents":subject2sents['PERSONAL texas public policy foundation dinner - thursday, february 15'], 
                                       'sp_sent_ids': [0, 1, 2]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The man who said Broadband held the future for delivery of Disney's entertainment product was born in what year?",
                                  idx="49P",
                                  answer=["1942"], 
                                  sp2={'title':'Michael Eisner', 
                                       "sents":wikipassages2sents['Michael Eisner'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL re: broadband partnership with walt disney corp', 
                                       "sents":subject2sents['PERSONAL re: broadband partnership with walt disney corp'], 
                                       'sp_sent_ids': [2]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Are the two businessmen Jack Welch and Michael Eisner both of the same nationality?",
                                  idx="50P",
                                  answer=["yes"], 
                                  sp1={'title':'Jack Welch', 
                                       "sents":wikipassages2sents['Jack Welch'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'Michael Eisner', 
                                       "sents":wikipassages2sents['Michael Eisner'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Are emmployees able to access the web portal, which spun off from Time Warner in 2009, on Enron computers?",
                                  idx="51P",
                                  answer=["no"], 
                                  sp1={'title':'AOL', 
                                       "sents":wikipassages2sents['AOL'], 
                                       'sp_sent_ids': [0, 7]}, 
                                  sp2={'title':'PERSONAL external e-mail sites', 
                                       "sents":subject2sents['PERSONAL external e-mail sites'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Why couldn't Wade attend the meeting with the Indian politician who serves as president of the Nationalist Congress Party?",
                                  idx="52P",
                                  answer=["he fell sick"], 
                                  sp1={'title':'Sharad Pawar', 
                                       "sents":wikipassages2sents['Sharad Pawar'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL re: meetings with sharad pawar', 
                                       "sents":subject2sents['PERSONAL re: meetings with sharad pawar'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Which was founded first, the party founded by Sharad Pawar or the Indian National Congress?",
                                  idx="53P",
                                  answer=["Indian National Congress"], 
                                  sp2={'title':'Indian National Congress', 
                                       "sents":wikipassages2sents['Indian National Congress'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp1={'title':'Sharad Pawar', 
                                       "sents":wikipassages2sents['Sharad Pawar'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Enron signed the manufacturer founded by Sidney and Bernard in 1953 to manufacture which offering?",
                                  idx="54P",
                                  answer=["CD/DVD"], 
                                  sp1={'title':'Harman Kardon', 
                                       "sents":wikipassages2sents['Harman Kardon'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL congrats & zapmedia...', 
                                       "sents":subject2sents['PERSONAL congrats & zapmedia...'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The Houston Astros have played as members of the Junior and \"Senior Circuit\"?",
                                  idx="55P",
                                  answer=["yes"], 
                                  sp1={'title':'Houston Astros', 
                                       "sents":wikipassages2sents['Houston Astros'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'American League', 
                                       "sents":wikipassages2sents['American League'], 
                                       'sp_sent_ids': [0, 2]}, 
                                  typ="bridge", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Do Azurix Corp and Wessex Water both operate in North America?",
                                  idx="56P",
                                  answer=["no"], 
                                  sp1={'title':'Wessex Water', 
                                       "sents":wikipassages2sents['Wessex Water'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'Azurix', 
                                       "sents":wikipassages2sents['Azurix'], 
                                       'sp_sent_ids': [0, 2]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Will the sewerage utility company that serves 1.3 million people in England be part of the Enron Global Assets and Services unit?",
                                  idx="57P",
                                  answer=["no"], 
                                  sp1={'title':'Wessex Water', 
                                       "sents":wikipassages2sents['Wessex Water'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL organizational changes3', 
                                       "sents":subject2sents['PERSONAL organizational changes3'], 
                                       'sp_sent_ids': [0, 4]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Who chose to donate to the company that's \"the equivalent of off-Broadway in Houston\" in the Enron Matching Gift Program?",
                                  idx="58P",
                                  answer=[" Rebecca Skupin"], 
                                  sp1={'title':'Stages Repertory Theatre', 
                                       "sents":wikipassages2sents['Stages Repertory Theatre'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL enron matching gift program winners', 
                                       "sents":subject2sents['PERSONAL enron matching gift program winners'], 
                                       'sp_sent_ids': [0, 6]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The employee focusing on distressed counterparties in RAC will represent Enron in which case?",
                                  idx="59P",
                                  answer=["Pacific Gas and Electric Company bankruptcy case"], 
                                  sp2={'title':'PERSONAL pg&e bankruptcy case-- important', 
                                       "sents":subject2sents['PERSONAL pg&e bankruptcy case-- important'], 
                                       'sp_sent_ids': [1, 2]}, 
                                  sp1={'title':'PERSONAL new legal team to assist rac', 
                                       "sents":subject2sents['PERSONAL new legal team to assist rac'], 
                                       'sp_sent_ids': [5]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Are Kevin Hannon and Lisa Mellencamp part of the same Enron business units?",
                                  idx="60P",
                                  answer=["no"], 
                                  sp1={'title':'PERSONAL organizational changes3', 
                                       "sents":subject2sents['PERSONAL organizational changes3'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL new legal team to assist rac', 
                                       "sents":subject2sents['PERSONAL new legal team to assist rac'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Did any of the investors in the KnowledgeCube venture capital firm get caught for insider trading?",
                                  idx="61P",
                                  answer=["yes"], 
                                  sp1={'title':'PERSONAL mckinsey alums/energy fund', 
                                       "sents":subject2sents['PERSONAL mckinsey alums/energy fund'], 
                                       'sp_sent_ids': [3, 5]}, 
                                  sp2={'title':'Rajat Gupta', 
                                       "sents":wikipassages2sents['Rajat Gupta'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="How many Grand Prix wins does Luca Baldisserri's driver at the Monaco Grand Prix have?",
                                  idx="62P",
                                  answer=["91"], 
                                  sp2={'title':'Michael Schumacher', 
                                       "sents":wikipassages2sents['Michael Schumacher'], 
                                       'sp_sent_ids': [2]}, 
                                  sp1={'title':'PERSONAL monaco grand prix', 
                                       "sents":subject2sents['PERSONAL monaco grand prix'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The championship Michael Schumaker has won seven times is sanctioned by which Federation?",
                                  idx="63P",
                                  answer=["Fédération Internationale de l'Automobile"], 
                                  sp1={'title':'Michael Schumacher', 
                                       "sents":wikipassages2sents['Michael Schumacher'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'Formula One', 
                                       "sents":wikipassages2sents['Formula One'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The conference which has been held annually at Reliant Park since 1969 has drawn close to how many participants?",
                                  idx="64P",
                                  answer=["50,000"], 
                                  sp1={'title':'Offshore Technology Conference', 
                                       "sents":wikipassages2sents['Offshore Technology Conference'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL speaking opportunity, otc - may2001', 
                                       "sents":subject2sents['PERSONAL speaking opportunity, otc - may2001'], 
                                       'sp_sent_ids': [8]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="For which event did Jeff Skilling invite the man who developed the market for \"Junk Bonds\" to speak?",
                                  idx="65P",
                                  answer=["Key Executive breakfast"], 
                                  sp1={'title':'Michael Milken', 
                                       "sents":wikipassages2sents['Michael Milken'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL re: michael milken', 
                                       "sents":subject2sents['PERSONAL re: michael milken'], 
                                       'sp_sent_ids': [0, 8]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="When did the ISO that works with the Texas Reliability Entity begin processing switch requests according to PUCT?",
                                  idx="66P",
                                  answer=["July 31, 2001"], 
                                  sp1={'title':'Electric Reliability Council of Texas', 
                                       "sents":wikipassages2sents['Electric Reliability Council of Texas'], 
                                       'sp_sent_ids': [0, 1, 2]}, 
                                  sp2={'title':'PERSONAL important update on your newpower service', 
                                       "sents":subject2sents['PERSONAL important update on your newpower service'], 
                                       'sp_sent_ids': [3, 4]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The November Rally Against Terrorism will be held at a Hotel which is accross the street from which subway station?",
                                  idx="67P",
                                  answer=["Pennsylvania Station"], 
                                  sp2={'title':'Hotel Pennsylvania', 
                                       "sents":wikipassages2sents['Hotel Pennsylvania'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL rally againt terrorism', 
                                       "sents":subject2sents['PERSONAL rally againt terrorism'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Are the Mastermind Keynote Interview in May and Creating Value with Internet Technologies conference in October located in the same city?",
                                  idx="68P",
                                  answer=["no"], 
                                  sp1={'title':'PERSONAL re: invitation', 
                                       "sents":subject2sents['PERSONAL re: invitation'], 
                                       'sp_sent_ids': [5]}, 
                                  sp2={'title':'PERSONAL speaker invitation to economist conference 24-25 october', 
                                       "sents":subject2sents['PERSONAL speaker invitation to economist conference 24-25 october'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Are Southern California Edison and Pacific Gas & Electric and San Diego Gas & Electric based in the same city?",
                                  idx="69P",
                                  answer=["no"], 
                                  sp1={'title':'San Diego Gas &amp; Electric', 
                                       "sents":wikipassages2sents['San Diego Gas &amp; Electric'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'Pacific Gas and Electric Company', 
                                       "sents":wikipassages2sents['Pacific Gas and Electric Company'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Enron was referred to as Williams Companies in the information request for an Act administered by which U.S. Agency?",
                                  idx="70P",
                                  answer=["Environmental Protection Agency"], 
                                  sp2={'title':'Clean Water Act', 
                                       "sents":wikipassages2sents['Clean Water Act'], 
                                       'sp_sent_ids': [0, 3, 4]}, 
                                  sp1={'title':'PERSONAL fw: 308 information request', 
                                       "sents":subject2sents['PERSONAL fw: 308 information request'], 
                                       'sp_sent_ids': [0, 2, 3]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="How many mutual fund offerings does the firm Philippe Bibi is resigning from Enron to join have?",
                                  idx="71P",
                                  answer=["79"], 
                                  sp2={'title':'Putnam Investments', 
                                       "sents":wikipassages2sents['Putnam Investments'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp1={'title':'PERSONAL philippe bibi', 
                                       "sents":subject2sents['PERSONAL philippe bibi'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Which firm manages more assets, Galleon Group or Putnam Investments?",
                                  idx="72P",
                                  answer=["Putnam Investments"], 
                                  sp1={'title':'Galleon Group', 
                                       "sents":wikipassages2sents['Galleon Group'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'Putnam Investments', 
                                       "sents":wikipassages2sents['Putnam Investments'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The 37th Governer of California has aggressively blamed companies from which state for California\'s energy meltdown?",
                                  idx="73P",
                                  answer=["Texas"], 
                                  sp1={'title':'Gray Davis', 
                                       "sents":wikipassages2sents['Gray Davis'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL la times article', 
                                       "sents":subject2sents['PERSONAL la times article'], 
                                       'sp_sent_ids': [3, 4]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())  


# question 
entry, original_entry = add_entry(q="The man whos leadership leads CA to purchase power at $135 per megawatt hour appeared on which late-night talk show?",
                                  idx="74P",
                                  answer=["The Tonight Show with Jay Leno"], 
                                  sp1={'title':'PERSONAL encouraging poll results', 
                                       "sents":subject2sents['PERSONAL encouraging poll results'], 
                                       'sp_sent_ids': [3]}, 
                                  sp2={'title':'PERSONAL the "dark" side of popular culture', 
                                       "sents":subject2sents['PERSONAL the "dark" side of popular culture'], 
                                       'sp_sent_ids': [1,4, 7, 8]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="The man who sent Wes Carberry an email about Jeff's departure has what street address?",
                                  idx="75P",
                                  answer=["1440 Smith Street"], 
                                  sp1={'title':'PERSONAL hope all is well...', 
                                       "sents":subject2sents['PERSONAL hope all is well...'], 
                                       'sp_sent_ids': [0, 8]}, 
                                  sp2={'title':'PERSONAL fw: business development opportunity', 
                                       "sents":subject2sents['PERSONAL fw: business development opportunity'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Are the Texas Venture Capital Conference and Southwest Venture Capital Conference supported by any of the same organizations?",
                                  idx="76P",
                                  answer=["yes"], # Houston Technology Center
                                  sp1={'title':'PERSONAL texas venture capital conference - 5.16.01', 
                                       "sents":subject2sents['PERSONAL texas venture capital conference - 5.16.01'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL upcoming golf tournament and venture capital conference', 
                                       "sents":subject2sents['PERSONAL upcoming golf tournament and venture capital conference'], 
                                       'sp_sent_ids': [4]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Frank Ianna reports to a leader from AT&T who is interested in Enron's value propositions from which team specifically?",
                                  idx="77P",
                                  answer=["Enron-Adventis team"], 
                                  sp1={'title':'PERSONAL talking points - at&t', 
                                       "sents":subject2sents['PERSONAL talking points - at&t'], 
                                       'sp_sent_ids': [2, 5]}, 
                                  sp2={'title':'PERSONAL moving forward: urgent, urgent.', 
                                       "sents":subject2sents['PERSONAL moving forward: urgent, urgent.'], 
                                       'sp_sent_ids': [1, 6]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="In which year was the individual on the Cogent Communications Advisory Committee from Coca Cola born?",
                                  idx="78P",
                                  answer=["1945"], 
                                  sp1={'title':'PERSONAL cogent communications', 
                                       "sents":subject2sents['PERSONAL cogent communications'], 
                                       'sp_sent_ids': [2, 3, 6]}, 
                                  sp2={'title':'Sergio Zyman', 
                                       "sents":wikipassages2sents['Sergio Zyman'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# # question 
entry, original_entry = add_entry(q="Sergio Zyman is known for his failure to launch a product which was later renamed to what?",
                                  idx="79P",
                                  answer=["Coke II"], 
                                  sp1={'title':'Sergio Zyman', 
                                       "sents":wikipassages2sents['Sergio Zyman'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'New Coke', 
                                       "sents":wikipassages2sents['New Coke'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Which state has is larger by area, Calfornia or Texas?",
                                  idx="80P",
                                  answer=["Texas"], 
                                  sp1={'title':'California', 
                                       "sents":wikipassages2sents['California'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'Texas', 
                                       "sents":wikipassages2sents['Texas'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Did the organization which Nasim spoke to McMahon about exceed its income targets during this first half?",
                                  idx="81P",
                                  answer=["yes"], 
                                  sp1={'title':'PERSONAL follow-up on my conversation in november', 
                                       "sents":subject2sents['PERSONAL follow-up on my conversation in november'], 
                                       'sp_sent_ids': [2, 11]}, 
                                  sp2={'title':'PERSONAL accomplishments-june 2001', 
                                       "sents":subject2sents['PERSONAL accomplishments-june 2001'], 
                                       'sp_sent_ids': [2]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Who is the assistant to the man responsible for Enron's e-commerce systems development including ClickPaper.com?",
                                  idx="82P",
                                  answer=["Tina Spiller"], 
                                  sp1={'title':'PERSONAL re: fw: eworldtradex', 
                                       "sents":subject2sents['PERSONAL re: fw: eworldtradex'], 
                                       'sp_sent_ids': [12]}, 
                                  sp2={'title':'PERSONAL your correspondence', 
                                       "sents":subject2sents['PERSONAL your correspondence'], 
                                       'sp_sent_ids': [3, 4, 5]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Do Jeff Skilling and Greg Piper have the same assistant at Enron?",
                                  idx="83P",
                                  answer=["no"], 
                                  sp1={'title':'PERSONAL your correspondence', 
                                       "sents":subject2sents['PERSONAL your correspondence'], 
                                       'sp_sent_ids': [8]}, 
                                  sp2={'title':'PERSONAL re: fw: eworldtradex', 
                                       "sents":subject2sents['PERSONAL re: fw: eworldtradex'], 
                                       'sp_sent_ids': [12]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# # question 
entry, original_entry = add_entry(q="The location for Eyeforenergy Asia 2001 is how many degrees north of the equator?",
                                  idx="84P",
                                  answer=["one"], 
                                  sp2={'title':'Singapore', 
                                       "sents":wikipassages2sents['Singapore'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp1={'title':'PERSONAL t,h: eyeforenergy briefing', 
                                       "sents":subject2sents['PERSONAL t,h: eyeforenergy briefing'], 
                                       'sp_sent_ids': [7]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# # question 
entry, original_entry = add_entry(q="Where dit the organizer of EEO Europe: Energy Trading in the New Economy hold its Asia 2001 conference?",
                                  idx="85P",
                                  answer=["Singapore"], 
                                  sp1={'title':'PERSONAL eeo europe:  energy trading in the new economy', 
                                       "sents":subject2sents['PERSONAL eeo europe:  energy trading in the new economy'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL t,h: eyeforenergy briefing', 
                                       "sents":subject2sents['PERSONAL t,h: eyeforenergy briefing'], 
                                       'sp_sent_ids': [7]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# # question 
entry, original_entry = add_entry(q="The country that had a population of 14 million at its birth in 1923 is bordered by how many countries?",
                                  idx="86P",
                                  answer=["eight"], 
                                  sp2={'title':'Turkey', 
                                       "sents":wikipassages2sents['Turkey'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp1={'title':'<32040359.1075840066357.JavaMail.evans@thyme>', 
                                       "sents":subject2sents['<32040359.1075840066357.JavaMail.evans@thyme>'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())


# question 
entry, original_entry = add_entry(q="After the earthquake in Turkey, Mark suggested sending contributions to an organization with national headquarters built between which years?",
                                  idx="87P",
                                  answer=["1915 and 1917"], 
                                  sp2={'title':'American Red Cross National Headquarters', 
                                       "sents":wikipassages2sents['American Red Cross National Headquarters'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp1={'title':'<32040359.1075840066357.JavaMail.evans@thyme>', 
                                       "sents":subject2sents['<32040359.1075840066357.JavaMail.evans@thyme>'], 
                                       'sp_sent_ids': [0, 3, 6]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Mark compared the Bosphorous Bridge to a silver version of a bridge that spans a straight how many miles long?",
                                  idx="88P",
                                  answer=["1 mi"], 
                                  sp1={'title':'Golden Gate Bridge', 
                                       "sents":wikipassages2sents['Golden Gate Bridge'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL fwd: picture!', 
                                       "sents":subject2sents['PERSONAL fwd: picture!'], 
                                       'sp_sent_ids': [4]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="How many people visit the 1545000 sqft retail space in Buckhead Atlanta annually?",
                                  idx="89P",
                                  answer=["25 million"], 
                                  sp1={'title':'Lenox Square', 
                                       "sents":wikipassages2sents['Lenox Square'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp2={'title':'PERSONAL lenox title sponsorship', 
                                       "sents":subject2sents['PERSONAL lenox title sponsorship'], 
                                       'sp_sent_ids': [0, 1, 2, 4]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Which has a larger revenue, Genscape Inc., the two year old energy info provider, or Midcoast Energy Resources with its 4,100 miles of pipe?",
                                  idx="90P",
                                  answer=["Midcoast Energy Resources"], 
                                  sp1={'title':'PERSONAL david doctor & genscape, inc.', 
                                       "sents":subject2sents['PERSONAL david doctor & genscape, inc.'], 
                                       'sp_sent_ids': [6]}, 
                                  sp2={'title':'PERSONAL acg october 9 lunch - reminder', 
                                       "sents":subject2sents['PERSONAL acg october 9 lunch - reminder'], 
                                       'sp_sent_ids': [7, 8]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Is the trustworthy organization Mark sugggested for Turkey's recovery one of the organizations Enron contributed to for September 11 relief efforts?",
                                  idx="91P",
                                  answer=["yes"], 
                                  sp1={'title':'<32040359.1075840066357.JavaMail.evans@thyme>', 
                                       "sents":subject2sents['<32040359.1075840066357.JavaMail.evans@thyme>'], 
                                       'sp_sent_ids': [0, 6]}, 
                                  sp2={'title':'PERSONAL our response to the u.s. tragedy', 
                                       "sents":subject2sents['PERSONAL our response to the u.s. tragedy'], 
                                       'sp_sent_ids': [0, 4]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="How many people were killed on the plane with Nick Humber, the Enron employee who was traveling to Los Angeles?",
                                  idx="92P",
                                  answer=["92"], 
                                  sp2={'title':'American Airlines Flight 11', 
                                       "sents":wikipassages2sents['American Airlines Flight 11'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  sp1={'title':'PERSONAL tragedy claims life of enron employee', 
                                       "sents":subject2sents['PERSONAL tragedy claims life of enron employee'], 
                                       'sp_sent_ids': [0, 1, 2, 3]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# # question 
entry, original_entry = add_entry(q="Who is the new Chief Financial Officer of the Enron group where Nick Humber was a director?",
                                  idx="93P",
                                  answer=["Tod Lindholm"], 
                                  sp1={'title':'PERSONAL tragedy claims life of enron employee', 
                                       "sents":subject2sents['PERSONAL tragedy claims life of enron employee'], 
                                       'sp_sent_ids': [2]}, 
                                  sp2={'title':'PERSONAL enron wind', 
                                       "sents":subject2sents['PERSONAL enron wind'], 
                                       'sp_sent_ids': [4]}, 
                                  typ="bridge", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Which has more miles, the GST transmission lines in the Carolinas or Midcoast's pipelines in 10 states?",
                                  idx="94P",
                                  answer=["transmission lines"], 
                                  sp1={'title':'PERSONAL gridsouth appointment', 
                                       "sents":subject2sents['PERSONAL gridsouth appointment'], 
                                       'sp_sent_ids': [4]}, 
                                  sp2={'title':'PERSONAL acg october 9 lunch - reminder', 
                                       "sents":subject2sents['PERSONAL acg october 9 lunch - reminder'], 
                                       'sp_sent_ids': [8]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# # question 
entry, original_entry = add_entry(q="Votenet Solutions, Inc released a Spanish language version of its software in celebration of a event which starts when?",
                                  idx="95P",
                                  answer=["September 15"], 
                                  sp2={'title':'National Hispanic Heritage Month', 
                                       "sents":wikipassages2sents['National Hispanic Heritage Month'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL votenet announces online voter registration software in spanish', 
                                       "sents":subject2sents['PERSONAL votenet announces online voter registration software in spanish'], 
                                       'sp_sent_ids': [1, 2]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="When was the company that acquired the EBS portfolio company, Amber Networks, founded?",
                                  idx="96P",
                                  answer=["1865"], 
                                  sp2={'title':'Nokia', 
                                       "sents":wikipassages2sents['Nokia'], 
                                       'sp_sent_ids': [0]}, 
                                  sp1={'title':'PERSONAL amber and storageapps acquired', 
                                       "sents":subject2sents['PERSONAL amber and storageapps acquired'], 
                                       'sp_sent_ids': [1]}, 
                                  typ="bridge", 
                                  domain=[1, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy())

# question 
entry, original_entry = add_entry(q="How many million dollars is the relationship between the largest U.S. newspaper publisher by daily circulation and Eric's company?",
                                  idx="97P",
                                  answer=["$270  million"], 
                                  sp1={'title':'Gannett Company', 
                                       "sents":wikipassages2sents['Gannett Company'], 
                                       'sp_sent_ids': [0, 2, 3]}, 
                                  sp2={'title':'PERSONAL congrats & zapmedia...0', 
                                       "sents":subject2sents['PERSONAL congrats & zapmedia...0'], 
                                       'sp_sent_ids': [1, 2]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Did Enron raise more money for United Way Scholars last year than the contribution amount for Enron's September 11 relief efforts?",
                                  idx="98P",
                                  answer=["yes"], 
                                  sp1={'title':'PERSONAL alexis de tocqueville breakfast and solicitation', 
                                       "sents":subject2sents['PERSONAL alexis de tocqueville breakfast and solicitation'], 
                                       'sp_sent_ids': [4, 5, 6]}, 
                                  sp2={'title':'PERSONAL our response to the u.s. tragedy', 
                                       "sents":subject2sents['PERSONAL our response to the u.s. tragedy'], 
                                       'sp_sent_ids': [0, 4]}, 
                                  typ="comparison", 
                                  domain=[1, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 

# # question 
entry, original_entry = add_entry(q="The team owned by Bruce McCaw partnered with Enron in which year?",
                                  idx="99P",
                                  answer=["2001"], 
                                  sp1={'title':'PacWest Racing', 
                                       "sents":wikipassages2sents['PacWest Racing'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'PERSONAL 2001 texaco/havoline grand prix', 
                                       "sents":subject2sents['PERSONAL 2001 texaco/havoline grand prix'], 
                                       'sp_sent_ids': [0]}, 
                                  typ="bridge", 
                                  domain=[0, 1])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 


# question 
entry, original_entry = add_entry(q="Are the New York City Subway and Washington Metro administered by the same Transit Authority agency?",
                                  idx="100P",
                                  answer=["no"], 
                                  sp1={'title':'New York City Subway', 
                                       "sents":wikipassages2sents['New York City Subway'], 
                                       'sp_sent_ids': [0]}, 
                                  sp2={'title':'Washington Metro', 
                                       "sents":wikipassages2sents['Washington Metro'], 
                                       'sp_sent_ids': [0, 1]}, 
                                  typ="comparison", 
                                  domain=[0, 0])
local_global_queries.append(entry.copy())
original_queries.append(original_entry.copy()) 

 


# save the queries
all_queries = []
all_queries.extend(local_global_queries.copy())
with open(f"/checkpoint/simarora/PersonalDatasets/Enron/qa_runs/{INBOX}/enron_wiki_qas_val_all.json", "w") as f:
    for query in all_queries:
        json.dump(query, f)
        f.write("\n")

with open(f"/checkpoint/simarora/PersonalDatasets/Enron/qa_runs/{INBOX}/enron_wiki_qas_original.json", "w") as f:
    json.dump(original_queries, f)

print(f"Saved all {len(all_queries)} queries!")
