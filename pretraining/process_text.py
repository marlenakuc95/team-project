import os
import random
from tqdm import tqdm
import io
from ml_things import plot_dict, fix_text
def pubmed_abstracts_to_file(path_data: str, path_texts_file: str):
    r"""Reading in all data from path and saving it into a single `.txt` file.

    Arguments:

        path_data (:obj:`str`):
          Path to the Movie Review Dataset partition. We only have `\train` and
          `test` partitions.

        path_texts_file (:obj:`str`):
          File path of the generated `.txt` file that contains one example / line.

    """

    # Check if path exists.
    if not os.path.isdir(path_data):
        # Raise error if path is invalid.
        raise ValueError('Invalid `path` variable! Needs to be a directory')
    # Check max sequence length.

    # Since the labels are defined by folders with data we loop
    # through each label.

    # We are randomly splitting them to 80% train and 20% test
    all_folders = [folder for folder in os.listdir(path_data) if os.path.isdir(os.path.join(path_data,folder))]
    random.shuffle(all_folders)

    num_folders = len(all_folders)

    train = all_folders[:int(num_folders*0.8)]
    test = all_folders[int(num_folders*0.8):]

    for train_or_test in ["train", "test"]:
        texts = []
        print('Reading `%s` partition...' % (os.path.basename(path_data)))

        # We are defining which one we are processing now
        if train_or_test == "train":
            current_batch = train
        else:
            current_batch = test


        for folder in current_batch:
            folder_path = os.path.join(path_data, folder)

            # Get all files from path.
            files_names = os.listdir(folder_path)  # [:30] # SAMPLE FOR DEBUGGING.
            # Go through each file and read its content.
            for file_name in tqdm(files_names, desc=folder, unit='files'):
                file_path = os.path.join(folder_path, file_name)

                # Read content.
                content = io.open(file_path, mode='r', encoding='ISO-8859-1').read()
                # Fix any unicode issues.
                content = fix_text(content)
                # Save content.
                texts.append(content)
        # Move list to single string.
        all_texts = '\n'.join(texts)
        # Send all texts string to single file.
        io.open(file=path_texts_file+"_"+train_or_test+".txt", mode='w', encoding='utf-8').write(all_texts)
        # Print when done.
        print(f'.txt file saved in {path_texts_file+"_"+train_or_test+".txt"} \n')

    return

pubmed_abstracts_to_file(path_data='/work-ceph/glavas-tp2021/team_project/datasets/pubmed/parsed_tr', path_texts_file='/work-ceph/glavas-tp2021/team_project/pretraining/all_texts')

