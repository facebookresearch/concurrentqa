import os
import sys
import argparse
import json as json
import pandas as pd
from collections import Counter, defaultdict
from importlib import reload
from email.parser import Parser


# recursively get the document body
def get_body(body):
    if type(body) == str:
        return [body]
    else:
        body_results = []
        for b in body:
            b_value = b.get_payload()
            if type(b_value) != str:
                body_results.append(get_body(b_value))
            else:
                body_results.append(b_value)
        return body_results

def parse_document(f):
    try:
        doc = f.read()
    except Exception as e:
        print(f"Exception, bad email: {e}!")
        doc = ""
    email = Parser().parsestr(doc)
    parse = defaultdict(list)
    for key in email.keys():
        parse[key] = email[key]
    body = email.get_payload()
    parse["Body"] = get_body(body)
    return parse

# recursive inspection because some sub directories have sub directories
def inspect_sub_dir(email_filename):
    if os.path.isfile(email_filename):
        with open(email_filename, "r") as f:
            entry = parse_document(f)
            entry["EMAIL_ID"] = email_filename
            assert type(entry["Body"]) == list
            return [entry]
    else:
        emails = os.listdir(email_filename)
        emails.sort()
        database = []
        for email in emails:
            file_name = email_filename + "/" + email
            database.extend(inspect_sub_dir(file_name))
        return database

def make_df(args, inbox):
    database = []
    sub_dirs = os.listdir(args.data_dir + inbox)
    print(sub_dirs)
    for sub_dir in sub_dirs:
        emails_dir = args.data_dir + inbox + "/" + sub_dir
        emails = os.listdir(emails_dir)
        emails.sort()
        for email in emails:
            email_filename = emails_dir + "/" + email
            database.extend(inspect_sub_dir(email_filename))
    return database

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Load datasets for enron.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/checkpoint/simarora/PersonalDatasets/Enron/maildir/",
        help="Raw enron data.",
    )

    parser.add_argument(
        "--db_dir",
        type=str,
        default="/checkpoint/simarora/PersonalDatasets/Enron/parsed_maildir",
        help="Parsed emails directory.",
    )

    args = parser.parse_args()
    inboxes = os.listdir(args.data_dir)
    inboxes.sort()
    for inbox in inboxes:
        if os.path.exists(f"{args.db_dir}/{inbox}_09082021.csv"):
            continue
        print(f"STARTING FOR INBOX: {inbox}")
        try:
            database = make_df(args, inbox)
            print(f"MADE INITIAL DB: {len(database)}")
            email_keys = database[0].keys() 
            df = pd.DataFrame(database) 
            outfile = f"{args.db_dir}/{inbox}_09082021.csv"
            df.to_csv(outfile)  
        except:
            print(f"FAILED ON INBOX: {inbox}")
