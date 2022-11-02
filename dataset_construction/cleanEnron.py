import re 
import sys
import os
import os.path
import random
import json
import time
import nltk.data
import spacy 
import pandas as pd
import random
from multiprocessing import Pipe, Pool
from functools import partial
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.append("/checkpoint/simarora/KILT/")
# from kilt.knowledge_source import KnowledgeSource
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# https://github.com/egerber/spaCy-entity-linker
# initialize language model
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker", last=True)
random.seed(1)
INBOX = "dasovich-j"
MY_PATH = "/private/home/simarora/pqa/PersonalDataDemo/" # SET YOUR PATH!
VALID_NER_TYPES = ['ORG', 'PERSON', 'LOC', 'EVENT', 'PRODUCT', 'LANGUAGE', 'LAW']
NER_TYPES_DICT = {
    'ORG': 'ORGANIZATION',
    'PERSON': "PEOPLE", 
    'LOC': "LOCATION", 
    'EVENT': "EVENT", 
    'PRODUCT': "PRODUCT", 
    'LANGUAGE': "LANGUAGES", 
    'LAW': "LEGAL"
}
PUNCT = ["'", ";", ":", ".", ",", '"', "|", ">", "<", "/", "?", ":", ";", "(", ")"]
OVERRIDE = []

# CREATE THE LOCAL CORPUS (approximately 5500 seconds)
def remove_structure_tokens(body):
    string_encode = body.encode("ascii", "ignore")
    body = string_encode.decode()
    body = body.strip()
    body = body.strip("]")
    body = body.strip("[")

    CLEAN_PAIRS = [("\no", " "), ("\n", " "), ("\\n", " "), ("\\t", " "), ("\\", ""), 
                    (" /", " "), (">", " "), ("=09", " "), ("=01", " "), ("=09=09=09=09", " "), ("---", " "),("|", " "),
                    ("___", " "), ("[IMAGE]", " "), ("= ", " "), ("???", " "), ("**", " "), ("??", " "), ("\xa0", " "),
                    ("=20", " "), ("0F", " "), (' " ', " "), (" ' ", " "), (". ?", ". "), ("=01", ""), ("=07", ""), 
                    ("National Assn.", "National Association")]
    for clean in CLEAN_PAIRS:
        body = body.replace(clean[0], clean[1])

    # floating quotes
    body_toks = body.split()
    if body_toks and body_toks[0] in ['"', "'", "?"]:
        body_toks = body_toks[1:]
    
    clean_body_toks = []
    for ind, tok in enumerate(body_toks):
        filt = 0
        if len(tok) == 1 and tok in PUNCT:
            filt = 1
        if all(ch == "?" for ch in tok): # or all(ch == "_" for ch in tok):
            filt = 1
        if ind > 0 and '.com' in body_toks[ind-1] and tok == 'o':
            filt = 1
        if len(tok) > 2 and tok[0] == "?":
            tok = tok[1:]
        if not filt:
            clean_body_toks.append(tok.strip())

    # get rid of 't o' and 'o f' type splits
    combined_tok = ''
    combined_toks = []
    for ind, tok in enumerate(clean_body_toks):
        if combined_tok:
            if len(tok) == 1 and tok.islower():
                combined_tok = combined_tok + tok
                combined_toks.append(combined_tok)
                combined_tok = ''
            else:
                combined_toks.append(combined_tok)
                combined_toks.append(tok)
                combined_tok = ''
        else:
            if len(tok) == 1 and tok.islower():
                combined_tok = tok
            else:
                combined_toks.append(tok)
                combined_tok = ''

    body = " ".join(combined_toks)

    # step 4: Wikiextractor cleaning steps
    body = body.replace('\t', ' ')
    body = body.replace('...', ' ')
    body = re.sub(u' (,:\.\)\]»)', r'\1', body)
    body = re.sub(u'(\[\(«) ', r'\1', body)
    body = re.sub(r'\n\W+?\n', '\n', body, flags=re.U)  # lines with only punctuations
    body = body.replace(',,', ',').replace(',.', '.')

    # Common abbreviations
    body = body.replace("U.S. ", "United States ")
    body = body.replace("Dept. ", "Department ")

    body = body.replace("  ", " ")

    return body


def identify_duplicates_by_text():
    first_sentences = []
    first_sentence_map = defaultdict(list)
    duplicates_map = {}
    num_duplicates = 0
    sentences_matched = 0

    with open(f"Enron_{INBOX}/EmailsCorpus.json") as f:
        EnronPassages = json.load(f)

    EnronPassages_New = {}
    for key, passage in tqdm(EnronPassages.items()):
        sents = passage['sents']

        # check if it's a duplicate
        is_duplicate = 0
        
        for sent in sents:
            if sent in first_sentences:
                is_duplicate = 1
                sentences_matched += 1
                first_sentence_map[sent].append(key)
                break

        # save whether it's a duplicate or not
        if not is_duplicate:
            for sent in sents:
                if len(sent.split()) > 1:
                    first_sentences.append(sent)
                    break
            first_sentence_map[sents[0]].append(key)
            duplicates_map[key] = False
        else:
            duplicates_map[key] = True
            num_duplicates += 1

        if not duplicates_map[key]:
            EnronPassages_New[key] = passage

    print(f"Marked {num_duplicates} passages as duplicates.")
    print(f"For {sentences_matched} passages, the first sentences matched exactly.")

    with open("first_sentence_map.json", "w") as f:
        json.dump(first_sentence_map, f)

    # only save the non-duplicates
    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json", "w") as f:
        json.dump(EnronPassages_New, f)

    return duplicates_map


def identify_linked_entities(bodies_lst):
    # want one mapping on entities to passages
    linked_entities_lst = []
    for body in bodies_lst:
        doc = nlp(body)
        # iterates over sentences and prints linked entities
        linked_entities = []
        for sent in doc.sents:
            for entity in sent._.linkedEntities.__dict__['entities']:
                entity_title = entity.__dict__['label']
                identifier = entity.__dict__['identifier']
                description = entity.__dict__['description']

                entity_details = {
                    'title': entity_title,
                    'identifier': identifier,
                    'description': description
                }

                linked_entities.append(entity_details)
        linked_entities_lst.append(linked_entities)
    return linked_entities_lst


def get_ner_tags(bodies_lst):
    ner_tags_lst = []
    for body in bodies_lst:
        doc = nlp(body)
        ner_tags = []
        for ent in doc.ents:
            ner_tag = {
                'text': ent.text, 
                'start_char': ent.start_char, 
                'end_char': ent.end_char, 
                'ner': ent.label_
            }
            ner_tags.append(ner_tag)
        ner_tags_lst.append(ner_tags)
    return ner_tags_lst


def split_body_to_sents(body):
    MAXIMUM_WORDS = 150
    MINIMUM_WORDS = 50
    num_words = 0
    body_sents, body_sents_lst = [], []

    EDGE_CASES = ["Assn.", "Abbrev.", "Var.", "Gov.", "Mass.", "No.", 
                  "Corp.", "Co.", "Cos.", "Inc.", "Pg.", "etc.", "?Pg.", "II.", 
                  "Mr.", "Mrs.", "Ms.", "CH.", "Ch.", "Md.", "Cup."]
    
    # split body into sentences
    all_sents = tokenizer.tokenize(body)
    new_all_sents = []
    current_sent = []
    for sent in all_sents:
        if sent and sent != " ":
            if (len(sent) > 1 and sent[-1] == "." and sent[-2].isdigit()) or (
                len(sent) ==2 and sent[-1] == "." and sent[-2].isupper()) or (
                len(sent) ==2 and sent[-1] == "(") or (
                sent.split()[-1] in EDGE_CASES) or (
                len([ch for ch in sent.split()[-1] if ch == "."]) > 1) or (
                len(sent) > 2 and sent[-1] == "." and sent[-2].isupper() and sent[-3] == " "):
                current_sent.append(sent)
            else:
                current_sent.append(sent)
                sent = " ".join(current_sent.copy())
                new_all_sents.append(sent)
                current_sent = []
    all_sents = new_all_sents.copy()

    # split into chunks of some maximum length
    for sent in all_sents:
        if sent:
            body_sents.append(sent)
            num_words += len(sent.split())
            if num_words > MAXIMUM_WORDS:
                body_sents_lst.append(body_sents.copy())
                body_sents = []
                num_words = 0

    # add the trailing/passages
    if num_words >= MINIMUM_WORDS:
        body_sents_lst.append(body_sents.copy())
        body_sents = []
        num_words = 0
    
    bodies_lst = []
    for body_sents in body_sents_lst:
        body = " ".join(body_sents)
        bodies_lst.append(body) 

    return bodies_lst.copy(), body_sents_lst.copy()


def create_local_documents(data, index):
    passage2sents = {}
    finalEntries = {}
    entity2emailid = defaultdict(list)
    email2entities = defaultdict(list)
    
    email_key = index 
    assert type(index) == int, print("index is not the correct format")
    psg_key = 0
    row = data[index]

    body = row["Body"]
    if body.strip():
        email_title = "EMAIL_" + str(email_key)
        body = remove_structure_tokens(body)

        # split the email into the MAX SEQ LENGTH sized chunks
        bodies_lst, body_sents_lst = split_body_to_sents(body)

        # get entity annotations
        ner_tags_lst = get_ner_tags(bodies_lst)
        linked_entities_lst = identify_linked_entities(bodies_lst) 
        
        for body, body_sents, linked_entities, ner_tags in zip(bodies_lst, body_sents_lst, linked_entities_lst, ner_tags_lst):
            psg_title = f"PERSONAL_e{str(email_key)}_p{str(psg_key)}"
            passage2sents[psg_title] = body_sents
            new_id = f"e{str(email_key)}_p{str(psg_key)}"
            finalEntries[new_id] = {
                            "id": new_id,
                            "email_title":email_title,
                            "title":psg_title, 
                            "text":body,
                            "sents":body_sents,
                            "ner_tags_lst":ner_tags,
                            "linked_entities_lst":linked_entities
                        }
            for ent in linked_entities:
                entity2emailid[ent['title']].append(psg_title)
                email2entities[psg_title].append(ent['title'])
            psg_key += 1
    return finalEntries, entity2emailid, email2entities, passage2sents


def create_local_passages_wrapper():
    # unzips the raw data
    pool = Pool(8)

    passage2sents = {}
    entity2emailid = defaultdict(list)
    email2entities = defaultdict(list)
    
    # load the correct inbox and the mappings
    with open(f"/checkpoint/simarora/PersonalDatasets/Enron/parsed_maildir/{INBOX}_09082021.csv") as f:
        data = pd.read_csv(f)
    print(f"Length of inbox: {INBOX} is {len(data)}")
    st = time.time()

    # select entries with an existent body and message id
    data = data[pd.notnull(data['Body'])]
    data = data[pd.notnull(data['Message-ID'])]
    data = data.to_dict('records')
    # data = data[0:100]
    data_indices = range(len(data))
    
    entries_lst, entity2emailid_lst, email2entities_lst, passage2sents_lst = zip(*pool.map(partial(create_local_documents, data), data_indices))
    finalEntries = {}
    for entries_dict in entries_lst:
        for key, entry in entries_dict.items():
            finalEntries[key] = entry
    with open(f"Enron_{INBOX}/EmailsCorpus.json", "w") as f:
        json.dump(finalEntries, f)

    for passage2sents_subdict in passage2sents_lst:
        for psg_key, sents in passage2sents_subdict.items():
            passage2sents[psg_key] = sents

    for email2entities_subdict in email2entities_lst:
        for psg_key, entities_list in email2entities_subdict.items():
            email2entities[psg_key] = entities_list

    for entity2emailid_subdict in entity2emailid_lst:
        for entity_name, psgs_list in entity2emailid_subdict.items():
            if entity_name in entity2emailid:
                entity2emailid[entity_name].extend(psgs_list)
            else:
                entity2emailid[entity_name] = psgs_list


    # # save the mappings
    with open(f"/checkpoint/simarora/PersonalDatasets/Enron/qa_runs/{INBOX}/subject2sents.json", "w") as f:
        json.dump(passage2sents, f)
        print(f"Saved passages 2 sents for {len(passage2sents)} passages.")
    
    with open(f"{MY_PATH}/Enron_{INBOX}/entity2emailid.json", "w") as f:
        json.dump(entity2emailid, f)
        print(f"Saved entity2emailid for {len(entity2emailid)} entities.")

    with open(f"{MY_PATH}/Enron_{INBOX}/email2entities.json", "w") as f:
        json.dump(email2entities, f)
        print(f"Saved email2entities for {len(email2entities)} emails.")

    print(f"Generate full set of personal documents in time: {time.time() - st}")
    print(f"There are: {len(finalEntries)} passages created.")


def extra_cleaning():
    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json") as f:
        EnronPassages = json.load(f)
    
    EnronPassages_New = {}
    for key, passage in tqdm(EnronPassages.items()):
        new_sents = []
        for sent in passage['sents']:
            sent = remove_structure_tokens(sent)
            if sent and sent != " ":
                if sent[0] == "?" and len(sent) > 1:
                    sent = sent[1:]
                new_sents.append(sent)
        passage["sents"] = new_sents
        passage['text'] = " ".join(new_sents)
        EnronPassages_New[key] = passage

    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json", "w") as f:
        json.dump(EnronPassages, f)


# FILTER POOR QUALITY NED TAGS AND GENERATE FINAL LISTS OF LOCAL / GLOBAL ENTITIES 
def ner_alias_replacements(tag_text):
    tag_toks = tag_text.split()
    tag_toks = [tok.replace("\\'s", "") for tok in tag_toks]
    tag_toks = [tok.replace("'s", "") for tok in tag_toks]
    tag_toks = [tok for tok in tag_toks if tok not in ['RE', 'F1', 'To:', "PS", "Subject", "Sent"]]
    tag_toks = [tok.replace("=20","").replace("=","").strip() for tok in tag_toks if tok not in ['the'] and tok not in PUNCT]
    tag_text = " ".join(tag_toks)

    tag_text = tag_text.replace("Enron", "")
    tag_text = tag_text.replace("U.S.", "United States")
    tag_text = tag_text.replace("US", "United States")
    tag_text = tag_text.replace("LA", "Los Angeles")
    tag_text = tag_text.replace("L.A.", "Los Angeles")
    tag_text = tag_text.replace("SF", "San Francisco")
    tag_text = tag_text.replace("NY", "New York")
    tag_text = tag_text.replace("N.Y.", "New York")

    # punct
    tag_text = tag_text.replace("**", "").strip()
    tag_text = tag_text.replace("-", "").strip()
    tag_text = tag_text.replace("\\t", " ").strip()
    tag_text = tag_text.replace("\\", "")
    tag_text = tag_text.replace(":", " ").strip()
    tag_text = tag_text.replace("  ", " ")
    return tag_text


def filter_named_entities():
    print("Running some filtering on tagged entities to remove poor quallity!")

    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json") as f:
        EnronPassages = json.load(f)

    linkedentities2nertags_global = {}
    for k, v in tqdm(EnronPassages.items()):
        text = v['text']
        ner_tag_text = []
        ner_tag_to_item = {}

        for tag in v['ner_tags_lst']:
            tag_text = ner_alias_replacements(tag['text'])
            if tag_text:
                ner_tag_text.append(tag_text)
                ner_tag_to_item[tag_text] = tag

        filtered_ents = []
        for ent in v['linked_entities_lst']:
            filter = 0

            # FILTER 1: exact match of alias and entity title
            if not ent or not ent['title'] or ent['title'] not in text:
                filter = 1

            # FILTER 2: if it's an an entity title that's not in the ner tagged spanned text at all
            if ent['title'] not in ner_tag_text:
                filter = 1

            # FILTER 3: if it's a PERSON NER tag, and not the full name (first, last) then drop it
            if not filter and ner_tag_to_item[ent['title']]['ner'] == "PERSON":
                if len(ent['title'].split()) == 1:
                    filter = 1

                # sometimes the second word is just an initial e.g., "Richard B."
                elif len(ent['title'].split()) == 2 and len(ent['title'].split()[1]) < 3:
                    filter = 1

            # FILTER 4: do any of the entity linking description words match the text? e.g., Nokia Chairman 

            if not filter:
                linkedentities2nertags_global[ent['title']] = ner_tag_to_item[ent['title']]['ner']
                filtered_ents.append(ent)

        v['linked_entities_lst'] = filtered_ents
    
    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json", "w") as f:
        json.dump(EnronPassages, f)

    with open(f"{MY_PATH}/Enron_{INBOX}/linkedentities2nertags_global.json", "w") as f:
        json.dump(linkedentities2nertags_global, f)


# PRODUCE A LIST OF THE LOCAL AND GLOBAL ENTITIES
def get_wiki_df():
    st = time.time()
    passages_path = '/checkpoint/simarora/mdr/data/hotpot_index/wiki_id2doc.json'
    with open(passages_path) as f:
        wiki_id2doc = json.load(f)
        passages = []
        for k, v in wiki_id2doc.items():
            v['id'] = k
            passages.append(v)
    print(f"Loaded full set of documents in {time.time() - st}")
    st = time.time()

    st = time.time()
    df = pd.DataFrame(passages)
    print(f"Loaded full set of documents in {time.time() - st}")
    st = time.time()
    wikititles = [psg['title'] for psg in passages]
    return df, wikititles 


def get_variations_lst(titles, wikititles=[], cache=None, text="", sents=[]):
    global_titles = {}
    remaining = []

    for tup in titles:
        title, tag = tup[0], tup[1]
        filt = 1
        if " Cor" in title: 
            USE_REPLACEMENTS = [("Corp.",""), ("Corporation", ""), ("Corp.", "Corporation"), ("Corp.", "Company")]
        elif " Co" in title:
            USE_REPLACEMENTS = [("Co.",""), ("Co", ""), ("Co.", "Company"), ("& Co.", ""), ("Computer", "")]
        elif "The " in title:
            USE_REPLACEMENTS = [("The ", "")]
        elif "Inc" in title:
            USE_REPLACEMENTS = [("Inc. ", ""), ("Inc.", "")]
        elif "Venture" in title:
            USE_REPLACEMENTS = [("Ventures", " "), ("Venture Fund", " ")]
        elif any(wd in title for wd in ['URL', 'Ltd.', '&', "Venture", "Capital", "News"]):
            USE_REPLACEMENTS = [("Ltd.", ""), ("URL", ""), ("&", "&amp;"),  ("Limited", ""), ("Newspaper", " "), ("Capital", " ")] 
        else:
            USE_REPLACEMENTS = []

        if USE_REPLACEMENTS:
            title = title.replace(",", " ")
            title = title.replace("  ", " ")
            for replace in USE_REPLACEMENTS:
                title_new = title.replace(replace[0], replace[1]).strip()
                if title == title_new:
                    continue
                elif title_new in cache and cache[title_new]:
                    filt = 0
                    break
                elif title_new in wikititles:
                    filt = 0
                    cache[title_new] = 1
                    break

            if not filt:
                text = text.replace(title, title_new)
                text = text.replace("  ",  " ")
                new_sents = []
                for sent in sents:
                    new_sents.append(sent.replace(title, title_new).replace("  ", " "))
                sents = new_sents.copy()
                global_titles[title_new] = tag
            else:
                remaining.append(title)

    return global_titles, remaining, text, sents, cache


def local_ents_refilter_by_wikipassages():
    df, wikititles = get_wiki_df()

    THRESH = 9
    freq_local = []
    with open(f"{MY_PATH}/Enron_{INBOX}/local_entities.json") as f:
        local_ents = json.load(f)
    for key, value in local_ents.items():
        if value > THRESH:
            freq_local.append(key)

    swapped_titles = []
    for local_title in freq_local:
        sents = len(df[df['title'].str.contains(local_title)]["id"].values)
        sents += len(df[df['text'].str.contains(local_title)]["id"].values)
        if sents >= 1:
            swapped_titles.append(local_title)
    
    with open(f"Enron_{INBOX}/local_ents_refilter.json", "w") as f:
        json.dump(swapped_titles, f)


def local_ents_checker(local_title, hard_coded_dictionary):
    # hard rules for which we want to exclude the local title as a local entity

    if any((len(tok) == 2 and tok.islower() and tok not in stop_words) for tok in local_title.split()):
        return False
    if any(tok in['PM', 'AM', 'EDT', 'EST', 'PST', 'AB', 'SB', 'Cc', 'RE', 'F1', 
                  'To:', "PS", "P.S.", "Subject", 'said', 'said.', "hasn\'t", 'has',
                  "doesn\'t", "does", "didn\'t", "did"] for tok in local_title.split()):
        return False
    if any((len(tok) == 1 and tok.islower() and tok not in ['a', 'i']) for tok in local_title.split()):
        return False
    if any(wd in local_title for wd in ['United States', "Dow Jones", 'New York', 'Committee',  "AT&T",
                                        "Associated Press", "Goldman Sachs", "Pacific Gas", "The Times", 
                                        "Financial Times", "Haas School", "Charles Schwab", 
                                        "Morgan Stanley", "J.P. Morgan", "Standard &", 
                                        "FERC", 'Los Angeles', "PG&E", "San Francisco", ".com"]):
        return False
    if local_title.split()[0] == '&' or local_title.split()[-1] == '&':
        return False
    if local_title.split()[0] in ['of', 'To', "or"] or local_title.split()[-1] == 'of':
        return False
    if "?" in local_title:
        return False
    if local_title.isupper() or local_title.islower():
        return False
    for tok in local_title.split():
        if all(t.isdigit() for t in tok):
            return False
    if hard_coded_dictionary[local_title]:
        OVERRIDE.append(local_title)
        return False
    return True


def hard_coded_remove_local_entities():
    hard_coded_dictionary = defaultdict(int)
    remove_local = [
        'Jeff Dasovich', 'Wash. Post', 'Private Company Business News', 'Public Service Company of New Mexico', 
        'Channing Way Berkeley', 'Universal Studios', 'California State', "National Assn.",
        'University of California, Berkeley Berkeley', 'AP Business Writer', 'Bad News', 'English News', 
        'West Coast', 'Haas Social Venture Competition', 'Haas Haas Celebrations',  
        'Electrical Engineering', 'Board of Directors', 'Pacific G&E', 'Calif Gov', 'California Senate', 
        'California Legislature', 'The Economic Times Copyright', 'Times Staff', 'Costa Times', 
        'Times Staff Writers', 'World Watch The Wall Street Journal', "Mobile Outlook",
        'The Wall Street Journal A2', 'Dear Haas Community', 'California State University and University of California', 
        'Jeff Dasovich NA', 'Justice Department', 'Energy Department', 'State Department', 'The Commerce Department', 
        'Department of Water', 'Department of Finance', 'Defense Department', 'Interior Department', 
        'Water Resources Department', 'Department of Commerce', 'The Energy Department', 'The Justice Department', 
        'The Department of Energy', 'Department of Education', 'Labor Department', 'The Department of Water Resources', 
        'The Labor Department', 'Treasury Department', 'Commerce Department', 'Northern and', 'Account and', 
        'Computer Science or Engineering', 'Participation in Roundtable on Lessons Learned', 
        'English News Service', 'Newport News', 'Domestic News', 'Eastern Time', 'Central Time', 'Govt Affairs', 
        'Evening MBA Program Office', 'General Accounting Office', 'III Chief of Staff Office of Assembly', 
        'Office of Emergency Services', 'Office of Government Ethics', 'The General Accounting Office', 'Docket Office', 
        'DSan Diego', 'The State Government', 'United Kingdom0F', 'Page A1', 'Gas & Electric', 'George W.', 
        'Gov Davis', 'Board BOSTON', 'Science & Technology', "Gov't Affairs", 'Section 19.3.2', 
        'Dow Jones)The California Independent System Operator','Corp. Cut', 'Securities & Exchange Commission',
        "Director Institute of Management, Innovation and Organization"
    ]

    print(f"Total remove local size: {len(remove_local)}")
    with open(f"{MY_PATH}/Enron_{INBOX}/hard_block_local_entities_v2.json", "w") as f:
        json.dump(remove_local, f)


    global_override = [
        'UC CSU', 'Enron Corp.', "Securities & Exchange Commission", "QUALCOMM, Inc.", 'UC Berkeley',
        'University of California Berkeley', 'Berkeley CA', 'University of California at Berkeley',
        'Merrill Lynch & Co.', 'Wells Fargo & Co.', 'Boeing Co.', 'U.C. Berkeley', 'Bain & Co.', 'Allen & Co.', 
        'Bear, Stearns & Co.', 'General Electric Co.', 'Ford Motor Co.', 'Walt Disney Co.', 'Transwestern Pipeline Co.',
        'Halliburton Co.', 'Portland General Electric Co.', 'Southern California Edison Co.', 
        'Transwestern Pipeline Co.', 'American Electric Power Co.', 'El Paso Natural Gas Co.','DTE Energy Co.',
        'Green Mountain Energy Co.','Commonwealth Edison Co.', 'Arizona Public Service Co.','Tata Power Co.',
        'Duke Energy Co.', 'DuPont Co.','Gas Co.','Gujarat Gas Co.', 'McKinsey & Co.', 'Goldman, Sachs & Co.',
        'Economic Times', 'New York Times', "New President & CEO", "President & CEO", "VC Fund", "Lays Off", 
        'UC San Diego', 'District of Columbia', 'JP Morgan Chase', 'Morgan Point', 'JP Morgan',  
        'Transwestern Pipeline Company', 'McKinsey & Company', 'The Gas Company', 'The Washington Post Co.',
        'El Paso Natural Gas Company', 'Portland General Electric Company', 'L.A. Times', 'Wall Street Journal',
        'Transwestern Pipeline Company', 'Southern California Edison Company', 'Chicago Tribune Company', 
        'Idaho Power Company', 'The Dabhol Power Company', "The Securities and Exchange Commission",
        'The New Power Company', 'San Diego Gas and Electric Company', 'Greenfield Shipping Company', 
        'Public Utility Holding Company Act', 'San Diego Gas & Electric Company', 'UC Davis', 'UC Irvine',
        'UC BERKELEY', 'Department of Water Resources', 'Exelon Corp.', "Chronicle Staff Writers",
        'Department of Energy', 'Department of Environmental Protection', "Department of Water Resources", 
        'TXU Corp.', 'Apache Corp.', 'Microsoft Corp.', 'Intel Corp.', 'Sony Corp.', 'News Corp.', 
        'General Motors Corp.', 'Exxon Mobil Corp.', 'Chevron Corp.', 'Compaq Computer Corp.', 
        'Nortel Networks Corp.', 'Enron North America Corp.', 'Enron Canada Corp.', 'Oracle Corp.', 'PPL Corp.',  
        'EMC Corp.', 'BellSouth Corp.', 'National Thermal Power Corp.', 'American Electric Power Service Corp.', 
        'Illinova Corp.', 'Electric Corp.', 'El Paso Energy Corp.', 'Indian Oil Corp.', 'TransAlta Corp.', 
        'Fluor Corp.', 'Dabhol Power Corp.', 'Mobil Corp.', 'Exxon Corp.', 'ChevronTexaco Corp.', 'E nron Corp.',
        'Questar Corp.', 'Qwest Corp.', 'Sprint Corp.', '- Enron Corp.', 'Bank of America Corp.', 
        'Bechtel Corp.', 'First Albany Corp.', 'Sempra Energy Corp.', 'Yellow Corp.', 'Sempra Energy Trading Corp.', 
        'Credit Suisse First Boston Corp.', 'VoiceStream Wireless Corp.', 'Oil & Natural Gas Corp.', 'Enron Corp. Cut', 
        'Enron Corporation', 'VC Personnel', "Time Warner Telecom, Inc.", "Time Warner Telecom", "Our Bureau Copyright",
        "Nortel Networks", "National Public Radio", "Independent Ene rgy Producers Association",
        "Cinergy Corp.", "Dynegy Inc.", "Dynegy Corp.", "Nasdaq Stock Market", "The Economist Newspaper", 
        "The Independent London FOREIGN", "Dell Computer",  "Viacom Inc.", "Compaq Computer", "Reuters Limited",
        "WalMart Stores Inc.", "Cisco Systems Inc.", "Royal Dutch Shell Group", "Occidental Petroleum Corp.", 
        "Marathon Oil Canada Inc.", "NRG Energy Inc.", "Barclays Global Investors",  "Deloitte Consulting", 
        "Financial Desk", "AP Business Writer DATELINE", "Financial Desk Markets", "Shiv SenaBJP",
        "AP Online", "Futu reTense", "Procter & Gamble", "Chronicle Staff",  "Environmental Strategies", "Editorial Desk", 
        "Johnson & Johnson", "Assembly Floor", "Assembly Energy", "Working Council", 
        "HewlettPackard Co.", "Board SAN FRANCISCO", "Angel Investors", "Your Account Settings", "McGrawHill, Inc.", 
        "Deutsche Bank AG", "Industrial Markets", "Verizon Communications, Inc.",  "Washington Post Staff",
        "Sun Microsystems Inc.", "Oil & Gas", "a Federal Energy Regulatory Commission", "UBS Capital", "AT&T Ventures", 
        "The Boston Consulting Group", "Oracle Venture Fund", "Gas Daily", 
        "The Supreme Court", "Internet Outlook", "Round Two", "NRG Energy, Inc.",  'Department of Justice',
        "Wireless Telecommunications", "a Securities and Exchange Commission", "Week Change", "Pacific, Boston",
        'Department of Water Resources.',"The Hindu Copyright (C", "PR Newswire (Copyright (c)", "Finance Ministry",    
    ]   

    swapped_titles = [
        'Enron Corp', 'Enron Corp.', 'Smith Street', 'Power Exchange', 'General Fund', 'Ken Lay', 'Dow Jones', 'Jim Foster', 'UBS Warburg', 
        'California Senate', 'Energy Committee', 'Universal Studios', 'Nevada Power Co.', 'Sierra Pacific Power', 'UC Berkeley', 'Bush Administration', 
        'Steve Baum', 'Dept. of', 'Water Resources', 'The Chronicle', 'Department of Water Resources', 'Legislative Analyst', 'Gordon Smith', 
        'Federal Energy Regulatory', 'Anne Kelly', 'Andy Brown', 'State Legislature', 'Quaker Oats', 'Advisory Group', 'San Diego Gas', 'Action Network', 
        'Government Affairs', 'Jeff D.', 'Utility Service', 'Williams Communications', 'Public Service Commission', 'Direct Access', 'California State', 
        'John Campbell', 'Chamber of Commerce', 'Sacramento Bee', 'San Jose Mercury News', 'Craig Rose', 'David Ward', 'Don Thompson', 'Public Affairs', 
        'Wall Street Journal', 'Independent System', 'Public Utilities Commission', 'Bill Campbell', 'John Nelson', 'Charles Schwab', 'Corporate Finance', 
        'California Assembly', 'Susan Davis', 'Pacific Gas', 'Proposition 9', 'Energy Commission', 'The Utility Reform Network', "Arthur O\\'Donnell", 
        'Electric Co.', 'Paul Patterson', 'Independent System Operator', 'Tom Higgins', 'Wheeler Ridge', 'Southern California Gas Co.', 'El Paso', 
        'Watson Wyatt', 'United States EPA', 'Business Development', 'David Young', 'Hewlett Packard', 'Bill Jones', 'Ray Hart', 'Pacific Gas &', 'California Edison', 
        'Senate Energy', 'Sony Computer Entertainment America', 'Reliant Energy', 'Pro Tem', 'Maharashtra Government', 'Salomon Smith Barney', 'West Coast', 
        'The White House', 'Claire Buchan', 'Halliburton Co.', 'Apache Corp.', 'Duke Energy Corp.', 'Dabhol Power Co.', 'Economic Times', 'Independent Energy', 
        'in California', 'Portland General Electric Co.', 'Portland General', 'Sierra Pacific', 'Mike Day', 'Rocky Mountain', 'Securities and Exchange Commission', 
        'AES Corp.', 'Michael Kahn', 'Dan Schnur', 'UC Davis', 'New York Times', 'John Stevens', 'Electric Company', 'Broadband Services', 'Ken Rice', 'Bay Area', 
        'New York Times Company', 'El Paso Energy', 'Rebecca Smith', 'Washington Post', 'Environmental Protection Agency', 'Southern Co.', 'Federal Reserve', 
        'International Business Machines', 'Microsoft Corp.', 'Intel Corp.', 'Walt Disney Co.', 'Verizon Communications Inc.', 'Sony Corp.', 'News Corp.', 'Big Board', 
        'George Bush', 'Entergy Corp.', 'Dabhol Power', 'Department of Energy', 'Portland General Electric Company', 'Phillips Petroleum Co.', 'Shell Oil Co.', 
        'John Chambers', 'Haas School', 'Utility Reform Network', 'Mark Cooper', 'North Field', 'State Government', 'Central Government', 'New Power', 'National Grid', 
        'Gulf Coast', 'John Anderson', 'General Motors Corp.', 'Home Depot', 'Exxon Mobil', 'MBA Program', 'Forest Service', 'Napa Valley', 'Carnegie Mellon', 
        'Washington University', 'John Edmiston', 'Quaker Oats Co.', 'American Electric Power Co.', 'Jeff Miller', 'Louis XIV', 't o', 'Joe Edwards', 'William S.', 
        'Energy Policy Act', 'General Electric Co.', 'International Business Machines Corp.', 'America Online', 'Wal-Mart Stores', 'Ford Motor', 'Bell Atlantic', 
        'SBC Communications', 'Fortune magazine', 'Exxon Mobil Corp.', 'Texaco Inc.', 'Chevron Corp.', 'Ford Motor Co.', 'Citigroup Inc.', 'Phillips Petroleum', 
        'J.C. Penney', 'Waste Management', 'Ethics Commission', 'Philip Morris', 'Union Government', 'Oversight Board', 'John Burton', 'County Board of Supervisors', 
        'Michael Katz', 'Jonathan Berk', 'University of Texas', 'Graduate School of Business', 'Wharton School', 'Mike Allen', 'California Commission', 'United States News', 
        'Andrew Rose', 'Ken Rosen', 'Urban Economics', 'Eugene E.', 'Business Administration', 'National Economic Council', 'Board of Directors', 'Asia Pacific', 
        'Marketing Group', 'John Morel', 'Electrical Engineering', 'External Affairs', 'Energy Services', 'New York', 'al l', 'New Economy', 'First Amendment', 'East Coast', 
        'Tracy Fairchild', 'Nevada Power', 'Amr Ibrahim', 'California Street', 'Republican Assembly', 'Supreme Court', 'Roger Salazar', 'Aaron Thomas', 'Joe Dunn', 
        'Tom Williams', 'John Sousa', 'east coast', 'Chapter 11', 'House Energy', 'Union Bank of California', 'Computer Center', 'District Court', 'Charles Robinson', 
        'State of California', 'J.P. Morgan', 'Golden State', 'Department of Environmental Protection', 'Natural Gas Act', 'Fortune 100', 'west coast', 'Dabhol Power Co', 
        'Lee Brown', 'City Council', 'City Hall', 'Digital Media', 'Edward Jones', 'Bank of New York', 'Bank One', 'Bankruptcy Court', 'Public Service Co.', 'United States Bank', 
        'Department of Water and Power', 'United States Bankruptcy Court', 'Southern California Gas', 'Eastern Time', 'Steve Johnson', 'Investors Service', 'Mercury News', 
        'Peter Cartwright', 'Securities Exchange Act', 'United States Supreme Court', 'PECO Energy Co.', 'Steve Wright', 'Cal State', 'Morro Bay', 'Southern Energy', 'AES Corp', 
        'Business Week', 'Mission Energy', 'Pacific Gas and Electric Co.', 'California Public Utilities', 'Henry Duque', 'United States Energy', 'Clean Air Act', 'Justice Department', 
        'Energy Department', 'Moss Landing', 'Chula Vista', 'United States House', 'Montana Power Co.', 'Montana Power', 'General Counsel', 'Pacific Gas and', 'Bankruptcy Code', 
        'College of Engineering', 'Federal Government', 'Squaw Valley', 'South Bay', 'Geoff Brown', 'Geoffrey Brown', 'Pat Wood', 'Oracle Corp.', 'Apple Computer', 'PPL Corp.', 
        'Wisconsin Energy', 'Stephen Oliver', "Los Angeles\\'", 'Cove Point', 'Williams Co.', 'United States Treasury', 'United States Circuit Court', 'Ras Laffan', 'Signature Services', 
        'customer s', 'United States Mail', 'United States Court of Appeals', 'Qualcomm Inc.', 'State Department', 'Bay area', 'Morgan Point', 'John Olson', 'Mike Smith', 'K Street', 
        'Richard Sanders', 'Bob Williams', 'Gary Fergus', 'Central Time', 'UC Irvine', 'Round One', 'Public Utility Commission', 'Energy Crisis', 'Energy Regulatory Commission', 
        'Rebecca Mark', 'Solar Power', 'Sierra Pacific Power Co.', 'Shell Oil', 'Sacramento Municipal Utility', 'Air Force', 'Workers Party', 'Peter Evans', 
        'Competitive Telecommunications Association', 'Richard Lyons', 'Commonwealth Edison Co.', 'Atal Bihari', 'Coyote Valley', 'Superior Court', 'Costa Times', 'Jack Scott', 
        'Jim Sanders', 'General Accounting Office', 'National Energy', 'Bill Morrow', 'Bob Foster', 'Bill Leonard', 'David Freeman', 'Dave Freeman', 'Board of Supervisors', 
        'Willie Brown', 'Communications Committee', 'Red Herring', 'Paul Carpenter', 'Harvey Morris', 'Market Surveillance Committee', 'State Auditor', 'The European Union', 
        'Electric Corp.', 'Utilities Commission', 'California Independent System', 'Joseph Dunn', 'John White', 'Robert Laurie', 'Richard Ellis', 
        'West Asia', 'Arizona Public Service Co.', 'Stephen Frank', 'Ross Johnson', 'Patrick Wood', 'David Hitchcock', 'Investor Service', 'ta ke', 'English News Service', 
        'Indian Oil Corp.', 'David Cox', 'Ben Campbell', 'John Wilson', 'Craig Barrett', 'William Wise', 'System Operator', 'East Bay', 'Fluor Corp.', 'sta te', 
        'Conference Board', 'San Francisco Chron', 'rat e', 'Dan Smith', 'Federal Energy', 'Clark Kelso', 'San Diego Gas &', 'Senate Select Committee', 'Public Utilities', 
        'Gray Dav', 'Department of Water', 'th e', 'Fair Oaks', 'Press Club', 'Tom Riley', 'Tamara Johnson', 'Air Resources Board', 'Regulatory Affairs', 'Marina del Rey', 
        'Desert Southwest', 'Franchise Tax Board', 'Investor Relations', 'General Assembly', 'High Point', 'Human Resources', 'ou r', 'Chase Manhattan', 'Ray Lane', 
        'Alex Brown', 'Venture Partners', 'Thomas White', 'Senate Appropriations', 'Robert C.', 'tha n', 'British Telecommunications plc', 'Health and Human Services', 
        'Harris Interactive', 'Kleiner Perkins', 'Mobil Corp.', 'Exxon Corp.', 'Steve Elliott', 'Board of Equalization', 'Department of Finance', 'Phi Beta Kappa', 'Richard Simon', 
        'Bank of Nova Scotia', 'Credit Lyonnais', 'Neil Stein', 'Wen Chen', 'Energy Conference', 'Undergraduate Program', 'Task Force', 'Legislative Counsel', 'Andersen Consulting', 
        'Indian Government', 'Ajit Kumar', 'Peter Behr', 'Kevin Murray', 'Carl Pope', 'Sean Gallagher', 'K. Lay', "Paul O\\'Neill", 'Chase Manhattan Bank', 'Maharashtra State', 'Banc of America', 
        'Ian Russell', 'Questar Corp.', 'State Senate', 'Republican Party', 'British Telecom', 'Salomon Smith', 'Defense Department', 'Wholesale Energy Market', 'Laurence Drivon', 'Western Power', 
        'John Hill', 'Regulatory Commission', 'o r', 'United States District Court', 'Air Quality', 'The Golden State', 'Boeing Co.', 'Social Security', 'UC San Diego', 'mor e', "Brian D\\'Arcy", 
        'the administration', 'n California', 'Northern and', 'yea r', 'International Power', 'California Chamber', 'Mike Briggs', 'California Independent', 'Elk Grove', 'wer e', 
        'Commonwealth Club', 'tha t', 'Los Angeles Department', 'stat e', 'Arctic National Wildlife', 'Diablo Canyon', 'District of Columbia', 'Pfizer Inc.', 'Jack Stewart', 'Keith McCrea', 
        'Barclays Capital', 'Qwest Corp.', 'Sprint Corp.', 'Enforcement Bureau', 'Financial Express', 'Business Council', 'Newport News', 'Press Trust', 'Nesbitt Burns', 'Brad Williams', 't he', 
        'Scott Reed', 'Chris Cox', 'Edwin Chen', 'Los Angeles Department of Water and ', 'Water Resources Department', 'at a', 'Randy Cunningham', 'Duke Power', 'Jeffrey A.', 'Jeff Brown', 
        'pa y', 'Joe Nation', 'Star Fleet', 'Montana Resources', 'Marine Corps', 'Office of Emergency Services', 'Otay Mesa', 'Rick Johnson', 'Societe Generale', 'Michael Hoffman', 
        'Blackstone Group', 'Community Energy', 'c Utilities Commission', 'Capital Investors', 'Venture Fund', 'Department of Commerce', 'Pinot Noir', 'Governing Board', 'vic e', 
        'Eastman Kodak', 'Carlyle Group', 'Grey Advertising', 'Model N', 'WR Hambrecht', 'North Slope', 'Energy Foundation', 'Christopher F.', 'Raymond James', 'Product Development', 
        'Dain Rauscher', 'Imperial Bank', 'Venture Capital', 'and Washington', 'Sevin Rosen', 'of Sales', 'Bank of America Corp.', 'n energy', 'Three Mile Island', 'Los Angeles Department of Water', 
        'Mark Baldwin', 'Global Coal', 'TL Ventures', 'George H.W. Bush', 'United States Power', 'for California', 'an d', 'control s', 'don e', 'the commission', 'Data Centers', 
        'Western Region', 'Capital Partners', 'Public Utility Holding Company Act', 'John Browne', 'Virodhi Andolan', 'are a', 'William Hogan', 'business development', 'Ken Smith', 
        'State Board of Equalization', 'Duke Energy Co.', 'Information Technology', 'William Blair', 'Technology Ventures', 'Capital Management', 'Growth Capital', 'Thomas Weisel', 
        'Investment Management', 'Union Pacific', 'Public Policy Institute', 'David Anderson', 'New West', 'supreme court', 'Susan Scott', 'Judiciary Committee', 'Eastman Chemical', 
        'Hummer Winblad', 'Draper Fisher', 'Arthur Andersen LLP', 'Department of Education', 'September 11th', 'S. David', 'Lloyds TSB', 'Republican party', 'for a', 'Amadeus Capital', 
        'Clay Johnson', 'Labor Department', 'Bill Wood', 'official s', 'Angeles Department of Water and Power', 'Florida Supreme Court', 'Governmental Affairs Committee', 'Royal Dutch', 
        'Alfred Kahn', 'World Affairs Council', 'Richard B.', 'Mechanical Engineering', 'Project Manager', 'The Independent Institute', 'Sony Music Entertainment', 'Texas Pacific', 
        'Providence Equity', 'Azure Capital', 'Page 2', 'Intel Corporation', 'Ministry of Defense', 'La Suer', 'Wind River', 'First Energy', 'Arts Alliance', 'Critical Path', 
        'Office of Government Ethics', 'Moore Capital', 'Desert Star', 'California Energy', 'United Way', 'Contra Costa', 'State Water Resources Control Board', 'West coast', 
        'Scott Miller', 'Channel 8', 'Rules Committee', 'Finance Group', 'PECO Energy', '2001 Los Angeles', 'Department of Justice', 'Contra Costa County', 'section 2', 'Pequot Capital', 
        'Bill Hall', 'William Hall', 'Royal Caribbean', 'Lee Friedman', 'Tom Gros', 'Blue Shield', 'Science Applications International', 'BMG Entertainment', 'Court of Appeals', 
        'Jeff Green', 'Bill Massey', 'Reed Elsevier', 'International Affairs', 'Professor of Public Policy', 'Computer Science', 'Data Warehouse', 'Michael Day', 'Dow Chemical', 
        'Fleur de Lys', 'Mona L', 'the Commission', 'First Fund', 'Discovery Capital', 'Applied Micro Circuits', 'California Report', 'Michael Ramsay', 'Tim Carter', 'Alpine Meadows', 
        'Order No', 'Salvation Army', 'Shaw Group', 'Michael M.', 'Chris H.', 'Williams III', 'Duke of', 'San Jose', 'David W', 'PS 2', 'Doug Smith', 'Securities and Exchange', 
        'Bonneville Power', 'Vol. 3', 'Steve Smith', 'Strategic Energy', 'Cal State Fullerton', 'Steve Hall', 'Phillip K.', 'Political Reform Act', 'Senate Committee', 'Glenn Johnson', 
        'Fair Political Practices Commission', 'Electric Board', 'Power Authority', 'Bill Ahern', 'John D. Dingell', 'John S.', 'New Energy', 'Northern Natural Gas', 'Michael Kirby', 
        'Gas Co.', 'Charlotte Observer', 'Stephen Moore', 'L.A. Times', 'Company, Inc.', 'Bob Anderson', 'William Mead', 'South Lake Tahoe', 'Wisconsin Gas', 'Mark Long', 
        'The Financial Express', "Brian O'Connell", 'Jim Fallon', 'Red Cross', 'Ann M.', 'James D.', 'Mark A.', 'Kevin Kelley', 'Steven J.', 'Linda J.', 'Coral Springs', 'P.O. Box', 
        'Steve C.', 'Susan M.', 'Cornell Club', 'Performance Management', 'Review Group', 'Robin Hill', 'Bad News', 'Opus One', 'Wireless Services', 'First Round', 
        'Kennedy School of Government', 'National Geographic', 'John Bowers', 'Optical Internet', 'Applied Physics', 'Implementation Group', 'Don Smith', 'Project Management', 
        'Community Choice', 'Power Pool', 'Press Conference', 'Treasury Department', 'Antitrust Act', 'Public Regulation Commission', 'Ray Williams', 'Facility Management', 'Ross Ain', 
        'Nord Pool', 'SBC Communications, Inc.', 'Global Telecom', 'Corporation Commission', 'Finance Committee', 'Valley Center', 'Motorola, Inc.', 'Fifth Circuit', 'Communications, Inc.', 
        'International Bureau', 'National Historic Preservation Act', 'Transportation Commission', 'Management Committee', 'South Slope', 'ris k', 'Dennis Harris', 'Public Affairs Committee', 
        'Data Quality', 'Murray P.', 'Rebecca W.', 'Hardy Jr', 'Barbara A.', 'Mona L.', 'World Trade Center', 'West Gas', 'English News', 'Nigel Shaw', 'Exchange Commission', 'Lisa M.', 
        'Commerce Department', 'American Water Works', 'American Water', 'Jane M.', 'Global Executive', 'Rob Nichol', 'Bill Ross', 'James Docker', 'Community Affairs', 'Project Lead', 
        'Mike Heim', 'Quinn Gillespie', 'William Barry', 'Milberg Weiss', '| | |', 'University Health Services', 'Adam N', 'Linda L.', 'Jo Ann', 'William Johnson', 'Blockbuster Inc.', 
        'Kenneth Rice', 'Commerzbank Securities', 'FPL Group', "Gray Davis'", 'San Diego Gas & Electric Co.', 'John Stout', 'Foundation for Taxpayer and Consumer Rights', 'MCI WorldCom', 
        'Covad Communications', 'Lucent Technologies', 'Jeff Skilling', 'San Diego Union Tribune', 'McGraw Hill', 'KGO Radio', 'San Diego Gas & Electric', 'Alpert Davis', 
        'Kern River Gas Transmission', 'Saber Partners', 'SoCal Gas', 'Con Edison', "Mike Day'", 'Technologic Partners', 'H&Q Asia Pacific', 'Law Ministry', 'Kasturi & Sons Ltd', 
        'Power Purchase Agreement', 'Calpine Corp.', 'Senate Floor', 'Delta Power', 'The California Energy Commission', 'Sierra Pacific Resources', 'Dan Richard', 
        'The Public Utilities Commission', 'Electronics Boutique', 'The California Public Utilities Commission', 'El Paso Corporation', 'William A. Wise', 'Tibco Software', 
        'Vivendi Universal', 'AOL Time Warner', 'Qwest Communications International Inc.', 'Gas Authority of India Ltd', 'Dominion Resources', 'Mirant Corp.', 'Michael Aguirre', 
        'British Petroleum', 'Valero Energy Corp.', 'Capstone Turbine Corp.', 'Conoco Inc.', 'Anadarko Petroleum Corp.', 'Schlumberger Ltd.', 'Deloitte & Touche', 'Japan Corp.', 
        'Finance Ministry', 'Lucent Technologies Inc.', 'CBS MarketWatch', 'Product Management', 'Jimmy Bean', 'Organization of Petroleum Exporting Countries', 'France Telecom', 
        'Dell Computer Corp.', 'Credit Lyonnais Securities', 'Azurix Corp.', 'Dow Jones & Company,', 'Illinois Power', 'Avista Corp.', 'Saks Inc.', 'Florida Power & Light', 
        'Northeast Utilities', 'Fisher Center for Real Estate and Urban Economics', 'Council of Economic Advisors', 'The Orange County Register', 'Mark Johnson', 
        'Lehman Brothers Holdings Inc.', 'Northwest Natural Gas', 'Comcast Interactive Capital', 'MSN Explorer', 'American Electronics Association', 'Richard Gephardt', 
        'Fortune Magazine', 'Hugo Chavez', 'Sycamore Networks', 'Corporate Communications', 'Duke Energy Corporation', 'Energy Intelligence Group', 'Montgomery Watson', 
        'Bertelsmann AG', 'Dresdner Kleinwort Wasserstein', 'Northern and Central California', 'Canada Corp.', 'National Desk', 'The Federal Energy Regulatory Commission', 
        'Calpine Corporation', '9th Circuit Court of Appeals', 'The Chronicle Publishing Co.', 'Stone & Webster', 'Pacific Gas and Electric', 'Bureau of Reclamation', 
        'John E. Bryson', 'Cingular Wireless', 'The Public Service Commission', 'Tyco International Ltd.', 'JDS Uniphase', 'Reliant Energy Services', 'Copley News Service', 
        'Columbia River Basin', 'Energy Services Inc.', 'British Wind Energy Association', 'Energy Systems Inc.', 'Phyllis Hamilton', 'UC Regents', 'National Thermal Power Corporation', 
        'Washington Bureau', 'Strategic Petroleum Reserve', 'Chuck Watson', 'Simmons & Co.', 'Energy Division', 'Vulcan Ventures', 'ING Barings', 'Science Communications', 
        'Anschutz Investment', 'Donaldson, Lufkin & Jenrette', 'Sigma Partners', 'Technology Crossover Ventures', 'Morgenthaler Ventures', 'New Millennium Partners', 
        'Internet Capital Group', 'Network Appliance', 'Hambrecht & Quist', 'Energy Services, Inc.', 'Larry Summers', 'Kohlberg Kravis Roberts & Co.', 'Blockbuster Video', 
        'Suez Lyonnaise des Eaux', 'John Heine', 'Lester Center for Entrepreneurship and Innovation', 'North American Electric Reliability Council', 'World Trade Organisation', 
        'Craig D.', 'Joseph Lieberman', 'Eli Lilly & Co.', 'Prudential Securities Inc.', 'Arter & Hadden', 'National Electric Power Authority', 'The Maharashtra Government', 
        'Judah Rose', 'Mirant Corp', 'Vestas Wind Systems', 'Global Crossing Ltd.', 'B.C. Hydro', 'The Brattle Group', 'The Energy Commission', 'The California Assembly', 
        'Global Markets', 'Career Services', "Department of Water Resources'", 'Western Energy', 'Ernst & Young', 'ABN Amro', 'Northwest Natural Gas Co.', 'Media Services', 
        'Steve Ballmer', 'Jeffrey Immelt', 'Wilson Sonsini Goodrich & Rosati', 'Duke Energy Corp', 'The Bonneville Power Administration', 'Regulatory Affairs Department', 
        'Industrial Development Bank of India', 'Paul Dawson', 'Giga Information', 'Crosspoint Venture Partners', 'Liberate Technologies', 'Chris Bowman', 'Barnes & Noble', 
        'Michael K. Powell', 'Bridgestone Firestone', 'Sofinnova Ventures', 'Ron Nichols', 'Navigant Consulting Inc.', 'Davis Administration', "Paul O'Neill", 'Joseph Pratt', 
        'Palm Computing', 'Industrial Finance Corporation', 'Utility Board', 'San Diego Superior Court', 'Con Ed', 'Carl Ingram', 'Pacific Bell Park', 'Mohave Generating Station', 
        'David Marshall', 'The Sacramento Municipal Utility District', 'U S WEST Communications, Inc.', 'Atal Behari', 'Dan Becker', 'James Woody', 'The City Council', 
        'The Public Utility Commission', 'Sun America', 'Middle East Economic Digest', 'National Energy Policy Development Group', 'Paul Kaufman', 'Jonathan Leonard', 
        'California Constitution', '11th Amendment', 'Canaan Partners', 'Whitney & Co.', 'Apollo Management', 'Blue Chip Venture', 'Kleiner Perkins Caufield & Byers', 
        'Scott Laughlin', 'CA Assembly', 'Labrador Ventures', 'J. & W. Seligman', 'Cable & Wireless', 'Crescendo Ventures', 'Jafco Ventures', 'Texas Pacific Group', 'with Davis', 
        'PA Consulting', 'Professional Services', 'Network Infrastructure', 'Benchmark Capital', 'Safeguard Scientifics', 'Zone Labs', 'Oxford Bioscience', 'Kodiak Venture Partners', 
        'Texas Public Utilities Commission', 'Christie Whitman', 'Low Income Home Energy Assistance Program', 'Williams Capital Group', 'Joseph Sent', 'William Blair Capital Partners', 
        'CNET Networks', 'Polaris Venture Partners', 'Bay Partners', 'Doll Capital Management', 'BP Plc', 'Joe Bob Perkins', 'Edward Kahn', 'Norman Y. Mineta', 'Sr. VP', 
        'Advent Venture Partners', 'Mark Fabiani', 'Independent Power Producers', 'Artemis Ventures', 'Trident Capital', 'Mohr Davidow Ventures', 'Ask Jeeves', 
        'The Electric Reliability Council of Texas', 'Democratic Assembly', 'OC Register', 'Gabriel Venture Partners', 'Challenge Fund', 'Insight Capital Partners', 
        'Sierra Ventures', 'Sandler Capital Management', 'Niagara Mohawk', 'Guy Phillips', 'Department of Health Services', 'John Flory', 'News World Communications, Inc.', 
        'VantagePoint Venture Partners', 'Walden International', 'Den Danske Bank', 'Lloyds TSB Development Capital', 'A.G. Edwards', 'Terra Lycos', 'SK Global', 
        'Gray Cary Ware & Freidenrich', 'Field Institute', 'Mexican Energy', 'Corporate Development', 'Willis Stein & Partners', 'Burrill & Co.', 'Prime Ventures', 
        'The Federal Energy Regulatory', 'Calpine Corp', 'Trinity Ventures', 'Mt. Tam', 'ARCH Venture Partners', 'First Union Capital Partners', 'Columbia Capital', '9th Circuit', 
        'Real Media', 'Sofinnova Partners', 'World Wide Packets', 'Netscape Communications', 'Department of Defense', 'Atal Behari Vajpayee', 'Holland & Knight', 'ETF Group', 
        'D.J. Smith', 'RRE Ventures', 'Boston Capital Ventures', 'New World Ventures', 'Global Switch', 'Horizon Ventures', 'Service Factory', 'CB Capital', 'GE Power Systems', 
        'Campesinos Unidos', 'Schroder Ventures', 'AT&T Canada', 'Coral Energy', 'Jupiter Communications', 'Venture Strategy Partners', 'Davidow Ventures', 'EchoStar Communications', 
        'AT&T Wireless', 'Itochu International', 'Mike Hansen', 'The California Department of Water Resources', 'GTCR Golder Rauner', "Ontario Teachers' Pension Plan Board", 
        'San Diego Gas & Electric Co', 'Lehman Brothers Venture Partners', 'MSN Hotmail', 'Mohr Davidow', 'J. & W. Seligman & Co.', 'Faculty Club', 'SAP Ventures', 'Capital Group', 
        'Pilgrim Baxter', 'Heather Cameron', 'ITC Holdings', 'NIB Capital', 'Datek Online', 'Freei Networks', 'Green Mountain Energy Company', 'Duquesne Light', 
        'Dell Computer Corporation', 'The Charles Schwab Corporation', 'Bayerische Landesbank', 'StarVest Partners', 'American Lawyer Media', 'Credit Suisse Group', 
        'Robert Mondavi Winery', 'Allegis Capital', 'Diego Gas & Electric Co.', 'Pervasive Computing', 'Lotus Notes', 'Mirant Corporation', 'Virginia Ellis', 
        'Electric Power Group', 'Jim Fleming', 'FPL Energy', 'Bechtel Group', 'Reliance Industries Ltd.', 'Richard Ferreira', 'Russell Hubbard', 'TransAlta Energy', 'Joel Newton', 
        'The Economist Group', 'Eugene Water & Electric Board', 'Qwest Communications', 'The Commission', 'AT&T Broadband', 'Rob Lamkin', 'California Supreme Court', 'Kasturi & Sons Ltd.', 
        'Kaufman, Paul', 'George H. Ryan', 'National Cable Television Association', 'Mobile Services', 'Public Utilities Act', 'Cambridge Silicon Radio', 'Clinton Administration', 
        'CSU Fresno', 'EBS, Inc.', 'Network Engineering', 'Common Carrier', 'BellSouth Telecommunications, Inc.', 'Nextel Communications, Inc.', 'Southwestern Bell Telephone Co.', 
        'Qwest Communications International, Inc.', 'WorldCom, Inc.', 'The State Corporation Commission', 'Lucent Technologies, Inc.', 'Cable Services', 
        'National Exchange Carrier Association, Inc.', 'John D. Rockefeller IV', 'FPL FiberNet', 'EOG Resources, Inc.', 'Catholic Health East', 'Christi L.', 'Mr Munde', 
        'Northern Natural Gas Co.', 'BSES Ltd.', 'BSES Ltd', 'Berkshire Hathaway Inc.', 'James J. Cramer', 'Robert Christensen', 'The Goldman Sachs Foundation', 'George Vaughn', 
        'David McManus', 'Gas Authority of India', 'Mary Lynne'
    ]

    global_override.extend(swapped_titles.copy())
    print(f"Added {len(swapped_titles)} to global override.")

    global_override = list(set(global_override))
    print(f"Total global override size: {len(global_override)}")
    with open(f"{MY_PATH}/Enron_{INBOX}/hard_block_global_override.json", "w") as f:
        json.dump(global_override, f)

    for title in remove_local:
        hard_coded_dictionary[title] = 1
    for title in global_override:
        hard_coded_dictionary[title] = 1

    return hard_coded_dictionary
    

def create_named_entity_maps():
    print("Creating named entity maps!")
    global_ents = Counter()
    local_ents = Counter()

    # load stuff
    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json") as f:
        EnronPassages = json.load(f)

    with open("wikititles.json") as f:
        wikititles = json.load(f)

    with open("/checkpoint/simarora/mdr/wikipassages2sents.json") as f:
        wikipassages2sents = json.load(f) 
    
    qid2types_wiki_filtered, title2qid, _, _, _, _ = global_entity_valid_types(wikititles)

    hard_coded_dictionary = hard_coded_remove_local_entities()

    with open(f"{MY_PATH}/Enron_{INBOX}/linkedentities2nertags_global.json") as f:
        linkedentities2nertags_global = json.load(f)
    nertags2psgs_global = defaultdict(list)
    nertags2psgs_local = defaultdict(list)

    fname = f"{MY_PATH}/Enron_{INBOX}/global_existence_cache.json"
    if os.path.isfile(fname):
        with open(fname) as f:
            global_existence_cache = json.load(f)
    else:
        global_existence_cache = {}

    # iterate through passages
    EnronPassages_New = {}
    for k, v in tqdm(EnronPassages.items()):

        # filter entities in passages with tons of emails, NED model does poorly on these names
        email_words = v['text'].count("@")
        email_words += v['text'].count("E-mail")
        if email_words > 1:
            v['GLOBAL_ENTITIES'] = []
            v['LOCAL_ENTITIES'] = []
            EnronPassages_New[k] = v.copy()
            continue

        # if the passage has global entities
        title_in_global = []
        title_not_global = []
        for ent in v['linked_entities_lst']:
            title = ent['title']
            if title in global_existence_cache:
                if global_existence_cache[title]:
                    title_in_global.append((title, linkedentities2nertags_global[title]))
                else:
                    title_not_global.append((title, linkedentities2nertags_global[title]))
            else:
                if title in wikititles:
                    global_existence_cache[title] = 1
                    title_in_global.append((title, linkedentities2nertags_global[title]))
                else:
                    global_existence_cache[title] = 0
                    title_not_global.append((title, linkedentities2nertags_global[title]))

        for tag in v['ner_tags_lst']:
            title = ner_alias_replacements(tag['text'])
            if len(title.split()) > 1 and tag['ner'] in VALID_NER_TYPES:
                if title in global_existence_cache:
                    if global_existence_cache[title]:
                        title_in_global.append((title, tag['ner']))
                    else:
                        title_not_global.append((title, tag['ner']))
                else:
                    if title in wikititles:
                        global_existence_cache[title] = 1
                        title_in_global.append((title, tag['ner']))
                    else:
                        global_existence_cache[title] = 0
                        title_not_global.append((title, tag['ner']))

        title_not_global = [t for t in title_not_global if not hard_coded_dictionary[t[0]] == 1]

        variations_lst, title_not_global, new_text, new_sents, global_existence_cache = get_variations_lst(title_not_global, wikititles=wikititles, cache=global_existence_cache, text=v['text'], sents=v['sents'])
        v['text'] = new_text 
        v['sents'] = new_sents

        for title, tag in variations_lst.items():
            if title not in global_existence_cache:
                global_existence_cache[title] = 1
                title_in_global.append((title, tag)) 

        # save local and global entities for the psg
        filtered_psg_local_ents = []
        filtered_psg_global_ents = [] 
        for tup in title_in_global:
            ent, nertag = tup[0], tup[1]
            global_ents[ent] += 1
            filtered_psg_global_ents.append(ent)
            filter_a = filter_global_ent(ent, wikipassages2sents, title2qid, qid2types_wiki_filtered)
            if not filter_a:
                nertags2psgs_global[nertag].append((v['id'], ent))  

        for tag in v['ner_tags_lst']:
            tag_text = ner_alias_replacements(tag['text'])
            if tag_text and tag_text not in filtered_psg_global_ents and tag['ner'] in VALID_NER_TYPES and local_ents_checker(tag_text, hard_coded_dictionary):
                if len(tag_text.split()) > 1: 
                    local_ents[tag_text] += 1
                    filtered_psg_local_ents.append(tag_text)
                    nertags2psgs_local[tag['ner']].append((v['id'], tag_text)) 

        v['GLOBAL_ENTITIES'] = filtered_psg_global_ents.copy()
        v['LOCAL_ENTITIES'] = filtered_psg_local_ents.copy()
        EnronPassages_New[k] = v.copy()

    # save stuff
    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json", "w") as f:
        json.dump(EnronPassages_New, f)

    with open(f"{MY_PATH}/Enron_{INBOX}/local_entities.json", "w") as f:
        json.dump(local_ents, f)

    with open(f"{MY_PATH}/Enron_{INBOX}/global_entities.json", "w") as f:
        json.dump(global_ents, f)

    with open(f"{MY_PATH}/Enron_{INBOX}/global_existence_cache.json", "w") as f:
        json.dump(global_existence_cache, f)

    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_local.json", "w") as f:
        json.dump(nertags2psgs_local, f)
    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_global.json", "w") as f:
        json.dump(nertags2psgs_global, f)


# DUPLICATE PASSAGES
def identify_duplicates(EnronPassages):
    ENT_OVERLAP_THRESH = 5
    OVERLAP_PCT_THRESH = 0.75

    entity_sets = []
    first_sentences = []
    first_sentence_map = defaultdict(list)
    duplicates_map = {}
    
    # metadata
    num_duplicates = 0
    sentences_matched = 0
    entities_overlapped = 0

    for key, passage in tqdm(EnronPassages.items()):
        sents = passage['sents']
        ents = passage['GLOBAL_ENTITIES'].copy()
        ents.extend(passage['LOCAL_ENTITIES'].copy())
        entity_set = set(ents)

        # check if it's a duplicate
        is_duplicate = 0
        
        for sent in sents:
            if sent in first_sentences:
                is_duplicate = 1
                sentences_matched += 1
                first_sentence_map[sent].append(key)
                break

        if not is_duplicate:
            for ent_set in entity_sets:
                overlap = len(entity_set.intersection(ent_set))
                if overlap > ENT_OVERLAP_THRESH and overlap/len(entity_set) >= OVERLAP_PCT_THRESH:
                    is_duplicate = 1
                    entities_overlapped += 1

        # save whether it's a duplicate or not
        if not is_duplicate:
            for sent in sents:
                if len(sent.split()) > 1:
                    first_sentences.append(sent)
                    break
            entity_sets.append(entity_set)
            first_sentence_map[sents[0]].append(key)
            duplicates_map[key] = False
        else:
            duplicates_map[key] = True
            num_duplicates += 1

    print(f"Marked {num_duplicates} passages as duplicates.")
    print(f"For {sentences_matched} passages, the first sentences matched exactly.")
    print(f"For {entities_overlapped} passages, the entity set had a high overlap with another passage's entity set.\n")

    with open("first_sentence_map.json", "w") as f:
        json.dump(first_sentence_map, f)

    return duplicates_map


def global_entity_valid_types(wikititles):
    print("Loading type information ...")
    with open ('/checkpoint/simarora/open_domain_data/BOOTLEG_entitydb/data/entity_db/entity_mappings/qid2title.json') as f:
        qid2title = json.load(f)

    with open("/checkpoint/simarora/open_domain_data/BOOTLEG_entitydb/data/entity_db/type_mappings/wiki/type_vocab.json") as f:
        wiki_type_vocab = json.load(f)
    
    with open("/checkpoint/simarora/open_domain_data/BOOTLEG_entitydb/data/entity_db/type_mappings/wiki/qid2typeids.json") as f:
        qid2types_wiki = json.load(f)

    with open("/checkpoint/simarora/mdr/wikipassages2sents.json") as f:
        wikipassages2sents = json.load(f) 

    wiki_typeid2name = {}
    for key, value in wiki_type_vocab.items():
        wiki_typeid2name[value] = key

    title2qid = {}
    for k, v in qid2title.items():
        title2qid[v] = k

    type2freq = Counter()
    type2qids = defaultdict(list)
    for title in tqdm(wikititles):
        if title in title2qid:
            qid = title2qid[title]
            types = qid2types_wiki[qid]
            for ty in types:
                type2freq[wiki_typeid2name[ty]] += 1
                if len(wikipassages2sents[title]) > 1:
                    type2qids[wiki_typeid2name[ty]].append(qid)

    # this is the list of types we want to allow for candidate entities 
    type2freq_filtered = {}
    type2qids_filtered = {}
    for ty, ct in type2freq.items():
        if ct >= 1000:
            type2freq_filtered[ty] = ct
            type2qids_filtered[ty] = type2qids[ty]

    with open("filteredEnronGlobalTypes.json", "w") as f:
        json.dump(type2freq_filtered, f)

    qid2types_wiki_filtered = {}
    for qid, types_lst in tqdm(qid2types_wiki.items()):
        filt_types = [wiki_typeid2name[ty] for ty in types_lst if wiki_typeid2name[ty] in type2freq_filtered]
        qid2types_wiki_filtered[qid] = filt_types

    return qid2types_wiki_filtered, title2qid, type2freq_filtered, type2qids_filtered, qid2title, type2qids


def filter_global_ent(title, wikipassages2sents, title2qid, qid2types_wiki_filtered):
    filter = 0
    MIN_PARAGRAPH_WORDS = 20

    # Filter 1: the passage is too short, meaning it's probably super vague or specific
    if len(wikipassages2sents[title]) <= 1:
        filter = 1

    # Filter 2: total words in the sentences for the passage, since there's probably too little content to write a q
    total_words = 0
    for sent in wikipassages2sents[title]:
        total_words += len(sent.split())
    if total_words < MIN_PARAGRAPH_WORDS:
        filter = 1

    # Filter 3: if the entity categories are not in the filtered types lists 
    if title not in title2qid:
        filter = 1
    else:
        qid_a = title2qid[title]
        if qid_a in qid2types_wiki_filtered:
            types_a = qid2types_wiki_filtered[qid_a]
        if not types_a:
            filter = 1
    return filter


def generate_passage_pairs():

    # GENERATE PASSAGE PAIRS
    def generate_global_global_pairs(wikititles, qid2types_wiki_filtered, title2qid):
        print("Creating global, global passage pairs.")
        random.seed(1)
        ks = KnowledgeSource()
        global_a = random.sample(wikititles, 50000)

        # load wiki corpus information
        wikititle_exists = defaultdict(int)
        for title in wikititles:
            wikititle_exists[title] = 1

        with open("/checkpoint/simarora/mdr/wikipassages2sents.json") as f:
            wikipassages2sents = json.load(f)    
            
        # produce pairs, first load existing saved anchors if it exists
	if os.path.exists("page2anchors.json"):
            with open("page2anchors.json") as f:
                page2anchors = json.load(f)
        else:
	    page2anchors = {}

        GLOBAL_GLOBAL_PAIRS = []
        added_wikients = []
        for title in tqdm(global_a):
            if title not in added_wikients:
                if title in page2anchors:
                    anchors = page2anchors[title]
                else:
                    page = ks.get_page_by_title(title)
                    if page:
                        anchors = page['anchors']
                        anchors_full = [anchor for anchor in anchors if anchor['paragraph_id'] == 1]
                        anchors = [anchor for anchor in anchors_full if wikititle_exists[anchor['text']]]
                        page2anchors[title] = anchors
                if anchors:
                    for anchor in anchors:
                        a, b = title, anchor['text']

                        # Filter the kinds of anchors we want by granularity
                        filter_a = filter_global_ent(a, wikipassages2sents, title2qid, qid2types_wiki_filtered)
                        filter_b = filter_global_ent(b, wikipassages2sents, title2qid, qid2types_wiki_filtered)

                        if not filter_a and not filter_b:
                            GLOBAL_GLOBAL_PAIRS.append({'wiki1':a, 'wiki2':b})
                            added_wikients.append(title)

        with open("page2anchors.json", "w") as f:
            json.dump(page2anchors, f)

        print(f"Collected {len(GLOBAL_GLOBAL_PAIRS)} global, global pairs\n")
        return GLOBAL_GLOBAL_PAIRS

    def generate_global_local_pairs(EnronPassages, duplicates_map, qid2types_wiki_filtered, title2qid):
        print("Creating global, local passage pairs.")
        MIN_PSG_ENTITIES = 3

        with open("/checkpoint/simarora/mdr/wikipassages2sents.json") as f:
            wikipassages2sents = json.load(f) 

        GLOBAL_LOCAL_PAIRS = []
        for key, passage in tqdm(EnronPassages.items()):
            is_duplicate = duplicates_map[key]
            if is_duplicate:
                continue

            if len(passage['GLOBAL_ENTITIES']) + len(passage['LOCAL_ENTITIES']) < MIN_PSG_ENTITIES:
                continue

            for ent in passage["GLOBAL_ENTITIES"]:
                filter_a = filter_global_ent(ent, wikipassages2sents, title2qid, qid2types_wiki_filtered)
                if not filter_a:
                    GLOBAL_LOCAL_PAIRS.append({'enron':key, 'wiki':ent})

        print(f"Collected {len(GLOBAL_LOCAL_PAIRS)} local, global pairs\n")
        return GLOBAL_LOCAL_PAIRS

    # Something I didn't include pairs-wise is local passage-pairs about global entities (due to the chance of knowledge intersection)
    def generate_local_local_pairs(EnronPassages, duplicates_map, freq_local):
        print("Creating local, local passage pairs.")
        MIN_PSG_ENTITIES = 3
        MAX_PSG_ENTITIES = 10
        FILT_LOCAL_LOCAL_PAIRS = []
        USED_PSG_PAIRS = []

        # get a mapping of the passages that contain each local entity
        localent2psgkey = defaultdict(list)
        for key, passage in tqdm(EnronPassages.items()):
            is_duplicate = duplicates_map[key]
            if is_duplicate:
                continue

            TOTAL_ENTITIES = len(passage['GLOBAL_ENTITIES']) + len(passage['LOCAL_ENTITIES'])
            if TOTAL_ENTITIES < MIN_PSG_ENTITIES or TOTAL_ENTITIES > MAX_PSG_ENTITIES:
                continue

            for ent in passage["LOCAL_ENTITIES"]:
                if ent in freq_local:
                    localent2psgkey[ent].append(key)
                
        # pick two passages that mention a local entity as a pair; one pair per local entity
        pair_counter = Counter()
        for ent, psgs in tqdm(localent2psgkey.items()):
            for psg1 in psgs:
                for psg2 in psgs:
                    if psg1 != psg2 and set([psg1, psg2]) not in USED_PSG_PAIRS:
                        FILT_LOCAL_LOCAL_PAIRS.append({'enron1':psg1, 'enron2':psg2, 'ent':ent})
                        USED_PSG_PAIRS.append(set([psg1, psg2]))
                        pair_counter[f"{psg1}_{psg2}"] += 1
                        break
                        
        LOCAL_LOCAL_PAIRS = []
        for pair in tqdm(FILT_LOCAL_LOCAL_PAIRS):
            # this filter is being used here as a sign of duplicacy
            if pair_counter[f"{pair['enron1']}_{pair['enron2']}"] < 5: 
                LOCAL_LOCAL_PAIRS.append(pair)

        print(f"Collected {len(LOCAL_LOCAL_PAIRS)} local, local pairs\n")
        return LOCAL_LOCAL_PAIRS

    # LOAD ENTITY SETS AND PERSONAL / GLOBAL CORPORA
    with open("{MY_PATH}/wikititles.json") as f:
        wikititles = json.load(f)
   
    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json") as f:
        EnronPassages = json.load(f)

    with open(f"{MY_PATH}/Enron_{INBOX}/global_entities.json") as f:
        global_ents = json.load(f)
        
    with open(f"{MY_PATH}/Enron_{INBOX}/local_entities.json") as f:
        local_ents = json.load(f)
    
    # Here we're choosing to use entities that appear above a THRESHOLD number of times in personal data
    THRESH = 5
    num = 0
    freq_local = []
    for key, value in local_ents.items():
        if value >= THRESH:
            num += 1
            freq_local.append(key)
    print(f"Found {len(local_ents)} local entities. and {num} entities appear over {THRESH} x.")

    num = 0
    freq_global = []
    for key, value in global_ents.items():
        if value >= THRESH:
            num += 1
            freq_global.append(key)
    print(f"Found {len(global_ents)} global entities and {num} global entities appear over {THRESH} x.\n")

    # GENERATE THE PASSAGE PAIRS
    qid2types_wiki_filtered, title2qid, _, _, _, _ = global_entity_valid_types(wikititles)
    fname = f"{MY_PATH}/Enron_{INBOX}/duplicate_enron_psg_map.json"
    if os.path.isfile(fname):
        with open(fname) as f:
            duplicates_map = json.load(f)
    else:
        duplicates_map = identify_duplicates(EnronPassages)
        with open(f"{MY_PATH}/Enron_{INBOX}/duplicate_enron_psg_map.json", "w") as f:
            json.dump(duplicates_map, f)
    print("Loaded duplicate passages map!\n")

    # global global passages
    GLOBAL_GLOBAL_PAIRS = generate_global_global_pairs(wikititles, qid2types_wiki_filtered, title2qid)
    with open(f"{MY_PATH}/Enron_{INBOX}/global_global_pairs.json", "w") as f:
        json.dump(GLOBAL_GLOBAL_PAIRS, f)

    # global local passages
    GLOBAL_LOCAL_PAIRS = generate_global_local_pairs(EnronPassages, duplicates_map, qid2types_wiki_filtered, title2qid)
    with open(f"{MY_PATH}/Enron_{INBOX}/global_local_pairs.json", "w") as f:
        json.dump(GLOBAL_LOCAL_PAIRS, f)

    # local local passages
    LOCAL_LOCAL_PAIRS = generate_local_local_pairs(EnronPassages, duplicates_map, freq_local)
    with open(f"{MY_PATH}/Enron_{INBOX}/local_local_pairs.json", "w") as f:
        json.dump(LOCAL_LOCAL_PAIRS, f)


def generate_comparison_passage_pairs():

    def generate_local_local_comparison(EnronPassages, duplicates_map, freq_local, nertags2psgs_local):
        print("Creating local, local passage pairs.")
        MIN_PSG_ENTITIES = 2
        MAX_PSG_ENTITIES = 10
        FILT_LOCAL_LOCAL_PAIRS = []
        USED_PSG_PAIRS = []

        # get a mapping of the passages that contain each local entity
        localent2psgkey = defaultdict(list)
        has_enough_ents = []
        for key, passage in tqdm(EnronPassages.items()):
            is_duplicate = duplicates_map[key]
            if is_duplicate:
                continue

            TOTAL_ENTITIES = len(passage['GLOBAL_ENTITIES']) + len(passage['LOCAL_ENTITIES'])
            if TOTAL_ENTITIES < MIN_PSG_ENTITIES or TOTAL_ENTITIES > MAX_PSG_ENTITIES:
                continue
            
            has_enough_ents.append(key)
            for ent in passage["LOCAL_ENTITIES"]:
                if ent in freq_local:
                    localent2psgkey[ent].append(key)
                
        # pick two passages that mention a local entity as a pair; one pair per local entity
        ner2psgkeys = defaultdict(list)
        psg2nertags = defaultdict(list)
        for NER_TAG, psgs in nertags2psgs_local.items():
            for psg in psgs:
                ner2psgkeys[NER_TAG].append(psg[0])
                psg2nertags[psg[0]].append([NER_TAG, psg[1]])

        by_common_ent = 1
        if by_common_ent:
            pair_counter = Counter()
            for NER_TAG, psgs in nertags2psgs_local.items():
                passages = psgs.copy()
                passages_keys = ner2psgkeys[NER_TAG].copy()
                print(f"NER TAG: {NER_TAG}")
                for tup1 in tqdm(passages):
                    inserted = 0
                    key1, title1 = tup1[0], tup1[1]
                    is_duplicate = duplicates_map[key1]
                    if is_duplicate:
                        continue
                    passage1 = EnronPassages[key1]
                    local_ents1 = [ent for ent in passage1['LOCAL_ENTITIES'] if ent != title1 and ent in freq_local]
                    for ent in local_ents1:
                        # iterate through passages with a matching local ent
                        other_passages = localent2psgkey[ent].copy()
                        random.shuffle(other_passages)
                        for other_psg in other_passages:
                            is_duplicate = duplicates_map[other_psg]
                            if is_duplicate:
                                continue
                            if other_psg in passages_keys:
                                other_nertags = psg2nertags[other_psg]
                                for tag in other_nertags:
                                    title2 = tag[1]
                                    key2 = other_psg
                                    if tag[0] == NER_TAG and title2 != ent and title2 != title1 and key1 != key2 and set([key1, key2]) not in USED_PSG_PAIRS and key1 in has_enough_ents and key2 in has_enough_ents:
                                        FILT_LOCAL_LOCAL_PAIRS.append({'enron1':key1, 'title1': title1, 'types':NER_TAG,
                                                                       'enron2':key2, 'title2': title2, 'ent':ent})
                                        USED_PSG_PAIRS.append(set([key1, key2]))
                                        pair_counter[f"{key1}_{key2}"] += 1
                                        inserted = 1
                                        break
                                    if inserted:
                                        break
                            if inserted:
                                break
                        if inserted:
                            break

        LOCAL_LOCAL_PAIRS = []
        for pair in tqdm(FILT_LOCAL_LOCAL_PAIRS):
            if pair_counter[f"{pair['enron1']}_{pair['enron2']}"] < 5: 
                LOCAL_LOCAL_PAIRS.append(pair)

        print(f"Collected {len(LOCAL_LOCAL_PAIRS)} local, local pairs\n")
        return LOCAL_LOCAL_PAIRS
        
    def generate_global_local_comparison(EnronPassages, duplicates_map, wikititles, nertags2psgs_local, nertitles2types_local, wikipassages2sents):
        _, _, type2freq_filtered, type2qids_filtered, qid2title, type2qids = global_entity_valid_types(wikititles)

        print("Creating local, global passage pairs.")
        MIN_PSG_ENTITIES = 2
        MAX_PSG_ENTITIES = 10

        # get a mapping of the passages that contain each local entity
        localent2psgkey = defaultdict(list)
        has_enough_ents = []
        for key, passage in tqdm(EnronPassages.items()):
            is_duplicate = duplicates_map[key]
            if is_duplicate:
                continue

            TOTAL_ENTITIES = len(passage['GLOBAL_ENTITIES']) + len(passage['LOCAL_ENTITIES'])
            if TOTAL_ENTITIES < MIN_PSG_ENTITIES or TOTAL_ENTITIES > MAX_PSG_ENTITIES:
                continue
            
            has_enough_ents.append(key)
            for ent in passage["LOCAL_ENTITIES"]:
                if ent in freq_local:
                    localent2psgkey[ent].append(key)
            
        titlehasWikiTypes = {}
        for tag, dic in nertitles2types_local.items():
            titlehasWikiTypes[tag] = []
            for title, lst in dic.items():
                if lst:
                    titlehasWikiTypes[tag].append(title)

        USED_TITLES = []
        FILT_GLOBAL_LOCAL_PAIRS = []
        USED_PSG_PAIRS = []
        pair_counter = Counter()
        for NER_TAG, psgs in nertags2psgs_local.items():
            passages = psgs.copy()
            print(f"NER TAG: {NER_TAG}")
            for tup1 in tqdm(passages):
                key1, title1 = tup1[0], tup1[1]

                if title1 not in titlehasWikiTypes[NER_TAG] or key1 not in has_enough_ents or duplicates_map[key1]:
                    continue

                types = nertitles2types_local[NER_TAG][title1]
                qids_lst = type2qids[types[0]].copy()
                while 1:
                    qid = random.choice(qids_lst)
                    wikititle = qid2title[qid]
                    qids_lst.remove(qid)
                    if len(wikipassages2sents[wikititle]) > 2:
                        break
                if wikititle not in USED_TITLES:
                    USED_TITLES.append(wikititle)

                FILT_GLOBAL_LOCAL_PAIRS.append({'enron1':key1, 'title1': title1, 'wiki':wikititle,
                                                'types': types[0]})
                USED_PSG_PAIRS.append(set([key1, wikititle]))
                pair_counter[f"{key1}_{wikititle}"] += 1

        print(f"Collected {len(FILT_GLOBAL_LOCAL_PAIRS)} local, global pairs\n")
        return FILT_GLOBAL_LOCAL_PAIRS
    
    # LOAD ENTITY SETS AND PERSONAL / GLOBAL CORPORA
    with open(f"{MY_PATH}/wikititles.json") as f:
        wikititles = json.load(f)
   
    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json") as f:
        EnronPassages = json.load(f)

    with open(f"{MY_PATH}/Enron_{INBOX}/global_entities.json") as f:
        global_ents = json.load(f)
        
    with open(f"{MY_PATH}/Enron_{INBOX}/local_entities.json") as f:
        local_ents = json.load(f)

    with open("/checkpoint/simarora/mdr/wikipassages2sents.json") as f:
        wikipassages2sents = json.load(f)

    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_local.json",) as f:
        nertags2psgs_local = json.load(f)
    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_global.json") as f:
        nertags2psgs_global = json.load(f)

    with open(f"{MY_PATH}/Enron_{INBOX}/nertitle2types_local.json") as f:
        nertitles2types_local = json.load(f)
    
    # Here we're choosing to use entities that appear above a THRESHOLD number of times in personal data
    THRESH = 5
    num = 0
    freq_local = []
    for key, value in local_ents.items():
        if value >= THRESH:
            num += 1
            freq_local.append(key)
    print(f"Found {len(local_ents)} local entities. and {num} entities appear over {THRESH} x.")

    num = 0
    freq_global = []
    for key, value in global_ents.items():
        if value >= THRESH:
            num += 1
            freq_global.append(key)
    print(f"Found {len(global_ents)} global entities and {num} global entities appear over {THRESH} x.\n")

    # GENERATE THE PASSAGE PAIRS
    # qid2types_wiki_filtered, title2qid, _, _, _, _ = global_entity_valid_types(wikititles)
    fname = f"{MY_PATH}/Enron_{INBOX}/duplicate_enron_psg_map.json"
    if os.path.isfile(fname):
        with open(fname) as f:
            duplicates_map = json.load(f)
    else:
        assert 0, print("no duplicate passages map")
    print("Loaded duplicate passages map!\n")

    # global local passages
    GLOBAL_LOCAL_PAIRS = generate_global_local_comparison(EnronPassages, duplicates_map, wikititles, nertags2psgs_local, nertitles2types_local, wikipassages2sents)
    with open(f"{MY_PATH}/Enron_{INBOX}/compare_global_local_pairs.json", "w") as f:
        json.dump(GLOBAL_LOCAL_PAIRS, f)

    # local local passages
    LOCAL_LOCAL_PAIRS = generate_local_local_comparison(EnronPassages, duplicates_map, freq_local, nertags2psgs_local)
    with open(f"{MY_PATH}/Enron_{INBOX}/compare_local_local_pairs.json", "w") as f:
        json.dump(LOCAL_LOCAL_PAIRS, f)


# COMPARISON Q HELPER FUNCTIONS
def filter_PERSON_ner(person_lst):
    clean_person_lst = []
    for tup in person_lst:
        person = tup[1]
        filt = 0
        if any(wd.isupper() for wd in person.split()):
            filt = 1

        elif any(wd in person.lower()for wd in ["corp", "california", "<<", ">>", "email", "greetings", "enron", "business", "smart",
                               "socal", "@", "director", "inc", "ect", "auditorium", "+", "=", "cos.", "staff", "www.", "pro",
                               "department", "manager", "co.", "cos", "strategy", "other", "news", "copyright", "land", "english"]):
            filt = 1

        elif len(person) > 40:
            filt = 1
        
        elif len(person.split()) != 2 or (len(person.split()[0]) <= 2 or len(person.split()[1]) <= 2):
            filt = 1

        elif any(ch.isdigit() for ch in person):
            filt = 1

        elif person in ["Andrew Rich Pinot Noir", "Gordon Smith Announce Partnership", "Jeff Energy Boss", 'Morgan Chase', 'Sac Bee',
                        "Gary Cohen Importance", "Fleetguard Nelson", "Price Falls", "Grosvenor Estates", 'Ventana Editions',
                        "Saloman Smith Barney India", "Fleetwood Enter", "Adobe Adobe Photoshop", "GungHo Atmosphere", "Bayless Cos.",
                        "Long Haul", "eProcurement Inbox", "Pass Code", "Graham Berkeley", "Natexis Investissement", "Walker Digital"]:
            filt = 1
        
        if not filt:
            clean_person_lst.append(tup)
    return clean_person_lst


def filter_ORG_ner(org_lst):
    clean_org_lst = []
    for tup in org_lst:
        org = tup[1]
        filt = 0
        if any((wd[0].islower() and wd not in stop_words) for wd in org.split()):
            filt = 1

        elif any(wd in org.lower()for wd in [","]):
            filt = 1

        elif len(org) > 50:
            filt = 1
        
        elif len(org.split()) >= 2 and (len(org.split()[0]) <= 2 or len(org.split()[1]) <= 2):
            filt = 1

        elif any(ch.isdigit() for ch in org):
            filt = 1

        if any(wd.lower() in ["council", "executive", "market"] for wd in org.split()):
            filt = 1

        if org in ["Philip Angelides", "Market Participant", "Independence Accounts"]:
            filt = 1

        elif org in []:
            filt = 1
        
        if not filt:
            clean_org_lst.append(tup)
    return clean_org_lst


def filter_EVENT_ner(event_lst):
    clean_event_lst = []
    event_words = [ "conference", "event", "session", "event", "weekend", "luncheon",
                    "festival", "workshop", "debate", "speech", "parade", "forum", 
                    "summit", "briefing", "lecture", "night"
    ]  
    for tup in event_lst:
        event = tup[1]
        filt = 0

        if any(wd in event.lower()for wd in [","]):
            filt = 1

        elif len(event) > 50:
            filt = 1
        
        elif len(event.split()) >= 2 and (len(event.split()[0]) <= 2 or len(event.split()[1]) <= 2):
            filt = 1

        elif any(ch.isdigit() for ch in event):
            filt = 1

        elif event in ["The Citysearch Weekend Guide", "Knowledge Forum", "Peyton Panel", "Bay Area Air"]:
            filt = 1

        elif not any(wd in event_words for wd in event.lower().split()):
            filt = 1
        
        if not filt:
            clean_event_lst.append(tup)
    return clean_event_lst


def filter_LOC_ner(event_lst):
    clean_event_lst = []
    event_words = [
        'bay', 'west', 'valley', 'north', 'south', 'east', 'the', 'coast', 'southern', 'central', 'river', 'area', 'district', 'pacific', 'northwest', 'california', 
        'silicon', 'island', 'san', 'lake', 'northern', 'asia', 'air', 'park', 'america', 'gulf', 'quality', 'sea', 'city', 'asiapacific', 'atlantic',  
        'drive', 'region', 'capital', 'western', 'basin', 'round', 'new', 'europe', 'county', 'border', 'desert', 'blvd', 'water', 'el', 'arctic',  
        'summit', 'inn', 'plant', 'southwest', 'road', 'st.', 'offshore', 'wind', 'regional', 'middle', 'indian', 'pine', 'wildlife', 'arabian', 
        'chapter', 'point', 'rim', 'ventures', 'islands', 'eastern', 'dieg', 'hill', 'mt.', 'jose', 'mission', 'avenue', 'castle', 'cleone', 'gardens', 
        'mendocino', 'schools', 'redwood', 'persian', 'board', 'field', 'san', 'jose', 'land', 'bluff', 'creek', 'dorado', 'hills', 
        'refuge',  'walla',  'little', 'mount', 'tower', 'energy', 'morro', 'upper', 'lands', 'block', 'american', 'plaza', 
        'pac', 'location', 'rock', 'marina', 'salt', 'generators', 'rto', 'verde', 'hudson', 'belt', 'orange', 'valley', 'ave', 'palm', 'napa', 'region', 
        'town', 'coasts', 'international', 'white', 'plains', 'angels', 'las', 'vegas', 'japan', 'los', 'england', 'india', 'great', 'basin', 'ocean', 
        'new', 'york', 'long', 'isle', 'woodlands', 'holland', 'arkansas'
    ]  
    for tup in event_lst:
        event = tup[1]
        filt = 0

        if len(event) > 50:
            filt = 1
        
        elif len(event.split()) >= 2 and (len(event.split()[0]) <= 2 or len(event.split()[1]) <= 2):
            filt = 1

        elif any(ch.isdigit() for ch in event):
            filt = 1

        elif any(wd.lower() in ["residents", "fund", "capital", "big", "council"] for wd in event.split()):
            filt = 1

        elif not any(wd in event_words for wd in event.lower().split()):
            filt = 1
        
        if not filt:
            clean_event_lst.append(tup)
    return clean_event_lst


def filter_LAW_ner(event_lst):
    clean_event_lst = []
    event_words = [
        'act', 'agreement',  'code', 'reform', 'bill', 'amendment', 'rights', 
        'rules', 'constitution', 'law', 'clause', 'compliance',  'bill', 
        'protocol', 'certification', 'policy', 'contract',  'standards'
    ]  
    for tup in event_lst:
        event = tup[1]
        filt = 0

        if len(event) > 50:
            filt = 1
        
        elif len(event.split()) >= 2 and (len(event.split()[0]) <= 2 or len(event.split()[1]) <= 2):
            filt = 1

        elif any(ch.isdigit() for ch in event):
            filt = 1

        elif not any(wd in event_words for wd in event.lower().split()):
            filt = 1
        
        if not filt:
            clean_event_lst.append(tup)
    return clean_event_lst


def filter_ner_maps():
    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_local.json") as f:
        nertags2psgs_local = json.load(f)
    
    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_global.json") as f:
        nertags2psgs_global = json.load(f)

    # PEOPLE
    clean_person_global = filter_PERSON_ner(nertags2psgs_global["PERSON"].copy())
    clean_person_local = filter_PERSON_ner(nertags2psgs_local["PERSON"].copy())
    nertags2psgs_global["PERSON"] = clean_person_global
    nertags2psgs_local["PERSON"] = clean_person_local

    # ORGS
    clean_org_global = filter_ORG_ner(nertags2psgs_global["ORG"].copy())
    clean_org_local = filter_ORG_ner(nertags2psgs_local["ORG"].copy())
    nertags2psgs_global["ORG"] = clean_org_global
    nertags2psgs_local["ORG"] = clean_org_local

    # EVENTS
    clean_event_global = filter_EVENT_ner(nertags2psgs_global["EVENT"].copy())
    clean_event_local = filter_EVENT_ner(nertags2psgs_local["EVENT"].copy())
    nertags2psgs_global["EVENT"] = clean_event_global
    nertags2psgs_local["EVENT"] = clean_event_local

    # LOC
    clean_loc_global = filter_LOC_ner(nertags2psgs_global["LOC"].copy())
    clean_loc_local = filter_LOC_ner(nertags2psgs_local["LOC"].copy())
    nertags2psgs_global["LOC"] = clean_loc_global
    nertags2psgs_local["LOC"] = clean_loc_local

    # LAW
    clean_law_global = filter_LAW_ner(nertags2psgs_global["LAW"].copy())
    clean_law_local = filter_LAW_ner(nertags2psgs_local["LAW"].copy())
    nertags2psgs_global["LAW"] = clean_law_global
    nertags2psgs_local["LAW"] = clean_law_local

    # PRODUCT
    nertags2psgs_global.pop("PRODUCT", None)
    nertags2psgs_local.pop("PRODUCT", None)

    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_local.json", 'w') as f:
        json.dump(nertags2psgs_local, f)
    
    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_global.json", 'w') as f:
        json.dump(nertags2psgs_global, f)


def get_bold_spans(ent_title, sents1=[], sents2=[]):
    bold_spans1 = []
    bold_spans2 = []
    ent_words = ent_title.split()
    ent_words = [wd for wd in ent_words if wd.lower() not in stop_words and (not any(ch.isdigit() for ch in wd))]

    if sents1:
        for sent in sents1:
            sent_spans = []
            for ind, tok in enumerate(sent.split()):
                if any(wd in tok for wd in ent_words):
                    sent_spans.append(ind)
            bold_spans1.append(sent_spans)

    if sents2:
        for sent in sents2:
            sent_spans = []
            for ind, tok in enumerate(sent.split()):
                if any(wd in tok for wd in ent_words):
                    sent_spans.append(ind)
            bold_spans2.append(sent_spans)

    return bold_spans1, bold_spans2


# MAIN ALGORITHM
def save_final_passage_pairs():

    # load key pairs
    with open(f"{MY_PATH}/Enron_{INBOX}/global_global_pairs.json") as f:
        GLOBAL_GLOBAL_PAIRS = json.load(f)

    with open(f"{MY_PATH}/Enron_{INBOX}/global_local_pairs.json") as f:
        GLOBAL_LOCAL_PAIRS = json.load(f)

    with open(f"{MY_PATH}/Enron_{INBOX}/local_local_pairs.json") as f:
        LOCAL_LOCAL_PAIRS = json.load(f)

    # load corpora
    with open(f"{MY_PATH}/wikititles.json") as f:
        wikititles = json.load(f)

    with open("/checkpoint/simarora/mdr/wikipassages2sents.json") as f:
        wikipassages2sents = json.load(f) 

    with open(f"{MY_PATH}/Enron_{INBOX}/EmailsCorpus_FILTERED.json") as f:
        EnronPassages = json.load(f)

    # load category information
    _, _, type2freq_filtered, type2qids_filtered, qid2title, _ = global_entity_valid_types(wikititles)


    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_local.json",) as f:
        nertags2psgs_local = json.load(f)
    with open(f"{MY_PATH}/Enron_{INBOX}/nertags2psgs_global.json") as f:
        nertags2psgs_global = json.load(f)

    # save final passage pairs
    bridge_passage_pairs = defaultdict(dict)
    pair_unq_idx = 0
    DATASET_SIZE = 80000
    while pair_unq_idx < DATASET_SIZE:
         r1 = 1
         hop1 = random.random() < 0.5
         hop2 = random.random() < 0.5
         unq_idx = f"PAIRIDX:{pair_unq_idx}"

         if r1:
             if hop1 and hop2:
                 if GLOBAL_GLOBAL_PAIRS:
                     pair = random.choice(GLOBAL_GLOBAL_PAIRS)
                     GLOBAL_GLOBAL_PAIRS.remove(pair)
                     boldspans_psg1, boldspans_psg2 = get_bold_spans(pair['wiki2'], sents1=wikipassages2sents[pair['wiki1']], sents2=wikipassages2sents[pair['wiki2']])
                     bridge_passage_pairs[pair_unq_idx] = {
                         'sents1': wikipassages2sents[pair['wiki1']],
                         'title1': pair['wiki1'],
                         'sents2': wikipassages2sents[pair['wiki2']],
                         'title2': pair['wiki2'],
                         'bold_spans1': boldspans_psg1,
                         'bold_spans2': boldspans_psg2,
                         'hint': f"Consider forming questions which use the entity '{pair['wiki2']}', since it's mentioned in both passages!",
                         'domains': [hop1, hop2],
                         'type':'bridge',
                         'unq_idx': unq_idx
                     }
                     pair_unq_idx += 1
            
             elif hop1 and not hop2:
                 if GLOBAL_LOCAL_PAIRS:
                     pair = random.choice(GLOBAL_LOCAL_PAIRS)
                     GLOBAL_LOCAL_PAIRS.remove(pair)
                     boldspans_psg1, boldspans_psg2 = get_bold_spans(pair['wiki'], sents1=wikipassages2sents[pair['wiki']], sents2=EnronPassages[pair['enron']]['sents'])
                     bridge_passage_pairs[pair_unq_idx] = {
                         'sents2': EnronPassages[pair['enron']]['sents'],
                         'title2': f"Enron Email Number: {EnronPassages[pair['enron']]['id']}",
                         'sents1': wikipassages2sents[pair['wiki']],
                         'title1': pair['wiki'],
                         'bold_spans1': boldspans_psg1,
                         'bold_spans2': boldspans_psg2,
                         'bold_spans': [],
                         'hint': f"Consider forming questions which use the entity '{pair['wiki']}', since it's mentioned in both passages!",
                         'domains': [hop1, hop2],
                         'type':'bridge',
                         'unq_idx': unq_idx
                     }
                     pair_unq_idx += 1

             elif not hop1 and hop2:
                 if GLOBAL_LOCAL_PAIRS:
                     pair = random.choice(GLOBAL_LOCAL_PAIRS)
                     GLOBAL_LOCAL_PAIRS.remove(pair)
                     boldspans_psg1, boldspans_psg2 = get_bold_spans(pair['wiki'], sents1=EnronPassages[pair['enron']]['sents'], sents2=wikipassages2sents[pair['wiki']])
                     bridge_passage_pairs[pair_unq_idx] = {
                         'sents1': EnronPassages[pair['enron']]['sents'],
                         'title1': f"Enron Email Number: {EnronPassages[pair['enron']]['id']}",
                         'sents2': wikipassages2sents[pair['wiki']],
                         'title2': pair['wiki'],
                         'bold_spans1': boldspans_psg1,
                         'bold_spans2': boldspans_psg2,
                         'bold_spans': [],
                         'hint': f"Consider forming questions which use the entity '{pair['wiki']}', since it's mentioned in both passages!",
                         'domains': [hop1, hop2],
                         'type':'bridge',
                         'unq_idx': unq_idx
                     }
                     pair_unq_idx += 1

             elif not hop1 and not hop2:
                 if LOCAL_LOCAL_PAIRS:
                     pair = random.choice(LOCAL_LOCAL_PAIRS)
                     LOCAL_LOCAL_PAIRS.remove(pair)
                     boldspans_psg1, boldspans_psg2 = get_bold_spans(pair['ent'], sents1=EnronPassages[pair['enron1']]['sents'], sents2=EnronPassages[pair['enron2']]['sents'])
                     bridge_passage_pairs[pair_unq_idx] = {
                         'sents1': EnronPassages[pair['enron1']]['sents'],
                         'title1': f"Enron Email Number: {EnronPassages[pair['enron1']]['id']}",
                         'sents2': EnronPassages[pair['enron2']]['sents'],
                         'title2': f"Enron Email Number: {EnronPassages[pair['enron2']]['id']}",
                         'bold_spans1': boldspans_psg1,
                         'bold_spans2': boldspans_psg2,
                         'bold_spans': [],
                         'hint': f"Consider forming questions which use the entity '{pair['ent']}', since it's mentioned in both passages!",
                         'domains': [hop1, hop2],
                         'type':'bridge',
                         'unq_idx': unq_idx
                     }
                     pair_unq_idx += 1

             else:
                 assert 0, print("Error in path selection!")
       
         if pair_unq_idx % 1000 == 0:
             print(f"Wrote {pair_unq_idx} questions.")
     print("Done collecting bridge pairs.")


    with open(f"{MY_PATH}/Enron_{INBOX}/compare_local_local_pairs.json") as f:
        LOCAL_LOCAL_PAIRS = json.load(f)

    with open(f"{MY_PATH}/Enron_{INBOX}/compare_global_local_pairs.json") as f:
        GLOBAL_LOCAL_PAIRS = json.load(f)

    comparison_passage_pairs = defaultdict(dict)
    COMPARISON_SIZE = 30000
    USED_TITLES = []
    pair_unq_idx = DATASET_SIZE 
    while pair_unq_idx < DATASET_SIZE+COMPARISON_SIZE:
        r2 = random.random() < 0.5
        hop1 = random.random() < 0.5
        hop2 = random.random() < 0.5
        sents1 = []
        sents2 = []
        unq_idx = f"PAIRIDX:{pair_unq_idx}"
        if 1:
            if not hop1 or not hop2:
                if not hop1 and not hop2:
                    if LOCAL_LOCAL_PAIRS:
                        pair = random.choice(LOCAL_LOCAL_PAIRS)
                        LOCAL_LOCAL_PAIRS.remove(pair)
                        
                        title1 = pair['title1']
                        key1 = pair['enron1']
                        sents1 = EnronPassages[key1]['sents']
                        boldspans_psg1, _ = get_bold_spans(title1, sents1=sents1)

                        title2 = pair['title2']
                        key2 = pair['enron2']
                        sents2 = EnronPassages[key2]['sents']
                        _, boldspans_psg2 = get_bold_spans(title2, sents2=sents2)

                        words1 = title1.split()
                        words2 = title2.split()
                        if len(set(words1).intersection(set(words2))) > 0:
                            continue

                        boldspans_psg1_ent, boldspans_psg2_ent = get_bold_spans(pair['ent'], sents1=sents1, sents2=sents2)

                        types = NER_TYPES_DICT[pair['types']]
                        common_ent = f"The entity {pair['ent']} also appears in both these passages."

                elif not hop1 and hop2:
                    if GLOBAL_LOCAL_PAIRS:
                        pair = random.choice(GLOBAL_LOCAL_PAIRS)
                        GLOBAL_LOCAL_PAIRS.remove(pair)
                        
                        title1 = pair['title1']
                        key1 = pair['enron1']
                        sents1 = EnronPassages[key1]['sents']
                        boldspans_psg1, _ = get_bold_spans(title1, sents1=sents1)

                        title2 = pair['wiki']
                        sents2 = wikipassages2sents[title2]
                        _, boldspans_psg2 = get_bold_spans(title2, sents2=sents2)

                        words1 = title1.split()
                        words2 = title2.split()
                        if len(set(words1).intersection(set(words2))) > 0:
                            continue

                        types = pair['types']
                        common_ent =''

                elif hop1 and not hop2:
                    if GLOBAL_LOCAL_PAIRS:
                        pair = random.choice(GLOBAL_LOCAL_PAIRS)
                        GLOBAL_LOCAL_PAIRS.remove(pair)
                        
                        title1 = pair['wiki']
                        sents1 = wikipassages2sents[title1]
                        boldspans_psg1, _ = get_bold_spans(title1, sents1=sents1)

                        title2 = pair['title1']
                        key2 = pair['enron1']
                        sents2 = EnronPassages[key2]['sents']
                        _, boldspans_psg2 = get_bold_spans(title2, sents2=sents2)

                        words1 = title1.split()
                        words2 = title2.split()
                        if len(set(words1).intersection(set(words2))) > 0:
                            continue

                        types = pair['types']
                        common_ent= ''

                if r2:
                    hint = f"Write a YES or NO question. Some information that can help you (feel free to ignore!) is that entity {title1} in passage 1 and entity 2 {title2} may have the '{types}' property in common. {common_ent}"
                    choices = [{"option":"Yes"}, {"option":"No"}]
                else:
                    hint = f"Some information that can help you (feel free to ignore!) is that entity {title1} in passage 1 and entity 2 {title2} may have the '{types}' property in common. {common_ent}"
                    choices = [{"option":title1}, {"option":title2}]
            else:
                WikiCategory = random.choice(list(type2freq_filtered.keys()))
                qids_lst = type2qids_filtered[WikiCategory].copy()
                
                qid1 = random.choice(qids_lst)
                qids_lst.remove(qid1)
                title1 = qid2title[qid1]
                if title1 not in USED_TITLES:
                    USED_TITLES.append(title1)
                    sents1 = wikipassages2sents[title1]
                    key1 = title1
                    boldspans_psg1, _ = get_bold_spans(title1, sents1=sents1)
                
                qid2 = random.choice(qids_lst)
                qids_lst.remove(qid2)
                title2 = qid2title[qid2]
                if title2 not in USED_TITLES:
                    USED_TITLES.append(title2)
                    sents2 = wikipassages2sents[title2]
                    key2 = title2
                    _, boldspans_psg2 = get_bold_spans(title2, sents2=sents2)

                if r2:
                    hint = f"Both passages are about '{WikiCategory.upper()}' entities:  '{title1} 'in paragraph 1, and '{title2}' in paragraph 2. Write a question that compares the two entities and can be answered with YES or NO."
                    choices = [{"option":"Yes"}, {"option":"No"}]
                else:
                    hint = f"Both passages are about '{WikiCategory.upper()}' entities: '{title1}' in paragraph 1, and '{title2}' in paragraph 2. Write a question that compares the two entities."
                    choices = [{"option":title1}, {"option":title2}]
            if sents1 and sents2:
                comparison_passage_pairs[pair_unq_idx] = {
                    'sents1': sents1,
                    'title1': title1,
                    'entity1': key1,
                    'sents2': sents2,
                    'title2': title2,
                    'entity2': key2,
                    'bold_spans1': boldspans_psg1,
                    'bold_spans2': boldspans_psg2,
                    'multiple_choice': choices,
                    'hint': hint,
                    'domains': [hop1, hop2],
                    'type':'comparison',
                    'unq_idx': unq_idx
                }
                pair_unq_idx += 1
        
        if pair_unq_idx % 1000 == 0:
            print(f"Wrote {pair_unq_idx} questions.")
    print("Done collecting comparison pairs.")

    # format for the frontend interface
    preprocessed = []
    for key, pair in bridge_passage_pairs.items():
         preprocessed.append(pair)
    output_dict = {
         'entries': preprocessed
    }
    with open(f"{MY_PATH}/Enron_{INBOX}/BRIDGE_PASSAGE_PAIRS_110121.json", "w") as f:
         json.dump(output_dict, f)

    preprocessed = []
    for key, pair in comparison_passage_pairs.items():
        preprocessed.append(pair)
    output_dict = {
        'entries': preprocessed
    }
    with open(f"{MY_PATH}/Enron_{INBOX}/COMPARISON_PASSAGE_PAIRS_01152022.json", "w") as f:
        json.dump(output_dict, f)


def convert_pairs_to_batches():
    BATCH_SIZE = 10
    with open(f"{MY_PATH}/Enron_{INBOX}/BRIDGE_PASSAGE_PAIRS_110121.json") as f:
        ALL_PAIRS = json.load(f)
    with open(f"{MY_PATH}/Enron_{INBOX}/COMPARISON_PASSAGE_PAIRS_01152022.json") as f:
        ALL_PAIRS = json.load(f)

    BATCH = {'entries': []}
    BATCH_NUM = 0
    for entry in tqdm(ALL_PAIRS['entries']):
        BATCH['entries'].append(entry)
        if len(BATCH['entries']) > BATCH_SIZE:
            with open(f"{MY_PATH}/Enron_{INBOX}/ComparisonBatches01152022/BATCH_{BATCH_NUM}.json", "w") as f:
                json.dump(BATCH, f)
            BATCH = {'entries': []}
            BATCH_NUM += 1
            if BATCH_NUM > 10:
                BATCH_SIZE = 100

    # trailing pairs in the final batch
    if len(BATCH['entries']) > 0:
            with open(f"{MY_PATH}/Enron_{INBOX}/ComparisonBatches01152022/BATCH_{BATCH_NUM}.json", "w") as f:
                json.dump(BATCH, f)
            BATCH = {'entries': []}
            BATCH_NUM += 1

    print(BATCH_NUM)


if __name__ == "__main__":

    # DETERMINE WHICH STEPS TO RUN
    create_corpus = 1
    prepare_entity_maps = 1
    prepare_comparison_qs = 1
    generate_pairs = 1

    ####### PRODUCE CORPUS ########
    if create_corpus:
        # 90 minutes
        create_local_passages_wrapper() 

        # 4.5 minutes
        identify_duplicates_by_text() 

    ######## REFINE LISTS OF LOCAL AND GLOBAL ENTITIES ########
    if prepare_entity_maps:
        # 0.5 minutes
        st = time.time()
        filter_named_entities() 
        print(f"Filtered named entities in time: {time.time() - st}")

        # 35 minutes
        st = time.time()
        create_named_entity_maps() 
        print(f"Created named entities map in time: {time.time() - st}")

    if prepare_comparison_qs:
        filter_ner_maps()

    ######## GENERATE PASSAGE PAIRS ########
    if generate_pairs:
        extra_cleaning()
        # 1 hour for global/global anchors are presaved, multiple days if we need to query the kilt database 
        # 1 minute for just global/local and local/local
        generate_passage_pairs() 

        generate_comparison_passage_pairs()

        # 1 minute
        save_final_passage_pairs()

    convert_pairs_to_batches()

