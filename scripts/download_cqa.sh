# download the retriever and reader training and evaluation files for ConcurentQA
wget https://dl.fbaipublicfiles.com/concurrentqa/data/CQA_dev_all.json
wget https://dl.fbaipublicfiles.com/concurrentqa/data/CQA_train_all.json
wget https://dl.fbaipublicfiles.com/concurrentqa/data/CQA_test_all.json

wget https://dl.fbaipublicfiles.com/concurrentqa/data/Retriever_CQA_dev_all_original.json
wget https://dl.fbaipublicfiles.com/concurrentqa/data/Retriever_CQA_train_all_original.json
wget https://dl.fbaipublicfiles.com/concurrentqa/data/Retriever_CQA_test_all_original.json

# download the background corpora from which to retrieve
wget https://dl.fbaipublicfiles.com/concurrentqa/corpora/enron_only_corpus.json
wget https://dl.fbaipublicfiles.com/concurrentqa/corpora/combined_corpus.json
wget https://dl.fbaipublicfiles.com/concurrentqa/corpora/title2sent_map.json
