from torch.utils.data import Dataset

class ContrastiveDataset(Dataset):
    def __init__(self, instructions, answers1, answers2, answers3):
        self.instructions = instructions
        self.answers1 = answers1
        self.answers2 = answers2
        self.answers3 = answers3

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        answer1 = self.answers1[idx]
        answer2 = self.answers2[idx]
        answer3 = self.answers3[idx]
        return instruction, answer1, answer2, answer3


