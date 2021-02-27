import torchtext
import re
def tokenizer(text):
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9]')
    text = regex.sub(" ", text)
    return text.split() 

def load_dataset(root="Dataset/spam.csv", batch_size = 128):
    TEXT = torchtext.data.Field(
        unk_token="",
        pad_token="",
        tokenize=tokenizer, 
        sequential=True,
        use_vocab=True,
        batch_first=True, 
        fix_length=50)
    LABEL = torchtext.data.Field(
        sequential=False, 
        use_vocab=False, 
        batch_first=True)
    
    train, test = torchtext.data.TabularDataset(path=root, format="CSV", fields=[("label", LABEL), ("data", TEXT)]).split()
    TEXT.build_vocab(train, test, vectors="glove.6B.100d")
    
    train_set = torchtext.data.Iterator(
        dataset=train, 
        batch_size=batch_size,
        shuffle=True)
    test_set = torchtext.data.Iterator(
        dataset=test, 
        batch_size=batch_size,
        shuffle=True)
    
    return train_set, test_set, TEXT, len(train), len(test), len(TEXT.vocab)