
Here we include the scripts we used to generate the passage pairs that were presented to the crowdworkers during the data collection process. Note that effective questions are generated when the passages in the pair have some commonalities. A strategy applied in the HotpotQA work to generate passage pairs was to select passages that contain overlapping sets of entity-mentions. A difficulty for our setting is that off-the-shelf entity-linkers work well on public entities (e.g., Wikipedia entities), but struggle on the private entities occurring in emails for instance. We thus use a combination of off-the-shelf NER and entity-linking models, and heuristic and manual checks to construct our pairs. 

We use [spacy-entity-linker](https://github.com/egerber/spaCy-entity-linker), [Bootleg](https://github.com/HazyResearch/bootleg), and [KILT](https://github.com/facebookresearch/KILT) in the passage pair construction pipeline. See ```cleanEnron.py``` for the main pipeline and ```EnronParser.py``` for downloading and parsing the raw Enron data. We present the passage pairs to crowdworkers and manage the data collection process using the Mephisto tool. Please see the paper for additional details.


We additionally include a small evaluation set we hand-wrote using the J. Skilling inbox of the Enron data (note our main benchmark uses the J. Dasovich inbox) in ```Enron_skilling-j```.
