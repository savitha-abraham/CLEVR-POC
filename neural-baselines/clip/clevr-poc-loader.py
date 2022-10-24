import json
import os
import pandas
import datasets
from random import randrange


DATA_FOLDER = os.path.expanduser('/home/marjan/myworks/code/python/CLEVR-POC/clevr-poc-dataset-gen')
DATASET_NAME = 'output-2000'

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class ClevrPOC(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    
    def read_labels():
        with open(os.path.join(DATA_FOLDER, 'data', 'properties.json'), encoding="utf-8") as f:
            properties = json.load(f)
        sorted_key_properties = sorted(properties.keys())

        key_properties_values = []
        for key_property in sorted_key_properties:
            key_properties_values.extend(sorted(properties[key_property].keys()))
        labels = {k: v for v, k in enumerate(key_properties_values)}
        return labels
            
    labels = read_labels()

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="clevr-poc", version=VERSION, description="CLEVR-poc general")
    ]

    DEFAULT_CONFIG_NAME = "clevr-poc"

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "constraint_type": datasets.Value("int64"),
                "image": datasets.Image(),
                "label": datasets.Value("int64")
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation            
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        
        """
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        image_dir = dl_manager.download_and_extract(_URLS['images'])
        image_dir_test = dl_manager.download_and_extract(_URLS['test_images'])
#        image_dir = _URLS['localimg']#dl_manager.download_and_extract(_URLS['images'])
        """

        data_path = os.path.join(DATA_FOLDER, DATASET_NAME, 'incomplete')
        questions_path = 'questions/'
        images_path = os.path.join(data_path, 'images')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_path, questions_path, 'training.json'),
                    "imgpath": images_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_path, questions_path, 'testing.json'),
                    "imgpath": images_path,
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_path, questions_path, 'validation.json'),
                    "imgpath": images_path,
                    "split": "val",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, imgpath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            json_file = json.load(f)
            df = pandas.json_normalize(json_file['questions'])
            for key, sample in df.iterrows():
                img_path = os.path.join(imgpath,
                                        sample['split'],
                                        sample['image_filename'])
                
                #TODO yield value_inputs so that it is possible to filter on them
                #print({"question": sample["question"], "image": img_path, "label": sample["answer"]})
            
                yield key, {
                    "question": sample["question"],
                    "constraint_type": sample["constraint_type"],
                    "image": img_path,
                    #"label": sample["answer"]
                    "label": ClevrPOC.labels[sample["answer"]]
                }