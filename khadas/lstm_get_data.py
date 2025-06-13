import os
from lstm_tools import convert_data_str_int, save_data

class GetDataPipeline:
    def __call__(self, data, letter:str='a', name='default', id=1):
        data = convert_data_str_int(data)

        os.makedirs(os.path.join('data', name, letter), exist_ok=True)

        save_data(data, filepath=os.path.join(
            'data', name, letter, f'{letter}-{id}.txt'))
        
        return data
        