import pandas as pd
'''
This is a custom automation testing library for stress testing the NLP model in recsysV1.py
'''
class ClubGenerator:
    def select_random_clubs(count):
        '''
        count -> number of clubs to select (int)
        Selects a random "count" number of clubs from the dataset.
        '''
        file_path = "C:\\Users\\linja\\Documents\\BU Documents\\BUDSA\\bu_organizations.csv" #please use your own local directory
        df = pd.read_csv(file_path)
        output_list = []
        if count > len(df):
            raise ValueError("Requested count exceeds the total number of clubs.")
            return []
        selected_orgs = df.sample(n=count)
        for org in selected_orgs['Organization Name']:
            output_list.append(org)
        return output_list

#  user_selected_clubs = select_random_clubs(5)