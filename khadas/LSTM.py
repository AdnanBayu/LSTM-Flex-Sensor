import torch
import numpy as np
from models.LSTM.LSTMModel import SIBILSTMModel
from lstm_tools import apply_threshold, read_config, output_audio
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM_WEIGHTS = "models/LSTM/lstm_flex_imu_best_model6.pth"
LSTM_WEIGHTS = "models/LSTM/lstm_flex_best_model6.pth"

class LSTMPipeline:
    def __init__(self, 
                 model_path=LSTM_WEIGHTS, 
                 config_path='default', 
                 seq_len=20,
                 counter_limit = 30
                 ):
        # self.model = SIBILSTMModel(input_size=11, hidden_size=64, num_layers=3, output_size=26, seq_length=seq_len).to(device)
        self.model = SIBILSTMModel(input_size=5, hidden_size=64, num_layers=3, output_size=26, seq_length=seq_len).to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))

        self.thresholds = read_config(config_path=config_path)
        self.seq_len = seq_len
        self.counter_limit = counter_limit
        self.cur_data = []

        self.counter = 0
        self.prediction = ''
        self.prev_prediction = ''

    def predict(self, data_torch):
        out = self.model(data_torch)
        prediction = chr(torch.argmax(out).item() + 97)
        return prediction
    
    def save_data(self):
        pass
    
    def output_audio(self, prediction):
        output_audio(prediction)
    
    def str_to_torch(self):
        data = self.cur_data[-self.seq_len:]
        
        data_np = []
        for i, data in enumerate(data):
            d = data.split(',')
            flex_data = [round(apply_threshold(float(d[i]), 2500, 4100), 2) for i in range(5)]
            # d = flex_data + d[8:-1] + [d[-1].strip()]  
            d = flex_data
            d = [float(x) for x in d]
            # for i in range(5, 8):
            #     d[i] = round(d[i]/20, 2)
                
            data_np.append(d)
        data_np = np.array(data_np)
        # data_np = data_np.reshape((1, 20, 11))
        data_np = data_np.reshape((1, 20, 5))
        data_torch = torch.from_numpy(data_np).float()
        return data_torch
    
    def __call__(self, data):
        self.cur_data.append(data)
        if len(self.cur_data) >= self.seq_len :
            data_torch = self.str_to_torch()
            self.prediction = self.predict(data_torch)
        
        if self.counter >= self.counter_limit:
            self.output_audio(self.prediction)
            self.counter = 0
        else : 
            if self.prediction == self.prev_prediction : 
                self.counter += 1
            else : 
                self.prev_prediction = self.prediction
                self.counter = 0

        print(f"Current Prediction : {self.prediction}")
        print(f'Current Counter : {self.counter}/{self.counter_limit}')

        return self.prediction