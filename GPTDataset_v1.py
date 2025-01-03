class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #Encodes entire text
        token_ids = tokeniser.encode(txt)

        #Uses a sliding window algo to chunk the book into overlapping sequences of max.length
        for i in range (0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    #Returns total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    #Returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]