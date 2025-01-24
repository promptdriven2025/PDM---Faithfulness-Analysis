import pickle

def merge_dicts(dict1, dict2):
    for primary_key, secondary_dict in dict2.items():
        if primary_key not in dict1:
            dict1[primary_key] = secondary_dict
        else:
            for secondary_key, value in secondary_dict.items():
                if secondary_key not in dict1[primary_key]:
                    dict1[primary_key][secondary_key] = value
    return dict1

def load_dict_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def save_dict_to_pkl(dict_data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dict_data, file)

file1 = 'cache_folder/dict11.pkl'
file2 = 'cache_folder/dict12.pkl'
output_file = 'cache_folder/cache_offline.pkl'

dict1 = load_dict_from_pkl(file1)
dict2 = load_dict_from_pkl(file2)

combined_dict = merge_dicts(dict1, dict2)

save_dict_to_pkl(combined_dict, output_file)

print("Dictionaries combined and saved to", output_file)
