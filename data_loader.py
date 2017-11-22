import numpy as np

EOS = [0]*50
maxword = 20400
maxsent = 10
maxques = 4

def load_data(glove_file, train_file, test_file):
    global maxword, maxsent, maxques

    word2vec = {}
    train_dict = {}
    train_input = []
    train_input_len = []
    train_question = []
    train_question_len = []
    train_gate = []
    train_target = []
    test_input = []
    test_input_len = []
    test_question = []
    test_question_len = []
    test_gate = []
    test_target = []

    glove_f = open(glove_file, "r")
    lines = glove_f.readlines()

    print "Building Glove Word Embedding Model..."
    for line in lines:
        tokens = line.split()
        word = tokens[0]
        embedding = [float(val) for val in tokens[1:]]
        word2vec[word] = embedding

    train_f = open(train_file, "r")
    lines = train_f.readlines()

    train_f = open(train_file, "r")
    lines = train_f.readlines()

    print "Building Train Set..."
    input_seq = []
    input_len_seq = []
    gate_idx = []
    question_cnt = 0
    idx = 0
    for line in lines:
        tokens = line.strip().split("\t")
        sentence = tokens[0].strip().lower().split(' ')
        # reset
        if idx > int(sentence[0]):
            input_seq = []
            input_len_seq = []
            gate_idx = []
            question_cnt = 0
        idx = int(sentence[0])
        sentence = sentence[1:-1] + [sentence[-1][:-1]] + [sentence[-1][-1]]
        for word in sentence:
            if train_dict.get(word) == None:
                train_dict[word] = len(train_dict)
        gate_idx.append(idx-1-question_cnt)

        # Question
        if len(tokens) > 1:
            target = tokens[1].lower()
            gate = gate_idx[int(tokens[2])-1]
            if train_dict.get(target) == None:
                train_dict[target] = len(train_dict)
            train_question.append(np.append([word2vec.get(word) for word in sentence], EOS*(maxques - len(sentence))))
            train_question_len = np.append(train_question_len, len(sentence))
            train_target = np.append(train_target, [train_dict.get(target)])
            train_gate.append(gate)

            if len(input_seq) < maxword:
                train_input.append(np.append(input_seq, ([0]*(maxword - len(input_seq)))))
                train_input_len.append(np.append(input_len_seq, ([0]*(maxsent - len(input_len_seq)))))
            else:
                if len(input_seq) > maxword:
                    print len(input_seq)
                train_input.append(input_seq)
                train_input_len.append(input_len_seq)
            question_cnt += 1
        # Informantion
        else:
            input_len_seq = np.append(input_len_seq, len(sentence))
            try:
                input_seq = np.append(input_seq, [word2vec.get(word) for word in sentence])
            except:
                print word, "is unkwon word"
                continue

    train_f.close()

    test_f = open(test_file, "r")
    lines = test_f.readlines()

    print "Building Test Set..."
    input_seq = []
    input_len_seq = []
    idx = 0
    gate_idx = []
    question_cnt = 0
    for line in lines:
        tokens = line.strip().split("\t")
        sentence = tokens[0].strip().lower().split(' ')
        # Reset
        if idx > int(sentence[0]):
            input_seq = []
            input_len_seq = []
            gate_idx = []
            question_cnt = 0
        idx = int(sentence[0])
        sentence = sentence[1:-1] + [sentence[-1][:-1]] + [sentence[-1][-1]]
        gate_idx.append(idx-1-question_cnt)
        
        # Question
        if len(tokens) > 1:
            target = tokens[1].lower()
            gate = gate_idx[int(tokens[2])-1]
            test_question.append(np.append([word2vec.get(word) for word in sentence], EOS*(maxques - len(sentence))))
            test_question_len = np.append(test_question_len, len(sentence))
            if train_dict.get(target) is None:
                "Unknown target in test...", target
            else:
                test_target.append([train_dict.get(target)])
            test_gate.append(gate)
            
            if len(input_seq) < maxword:
                test_input.append(np.append(input_seq, ([0]*(maxword - len(input_seq)))))
                test_input_len.append(np.append(input_len_seq, ([0]*(maxsent - len(input_len_seq)))))
            else:
                if len(input_seq) > maxword:
                    print len(input_seq)
                test_input.append(input_seq)
                test_input_len.append(input_len_seq)
            question_cnt += 1
        # Informantion
        else:
            input_len_seq.append(len(sentence))
            try:
                input_seq = np.append(input_seq, [word2vec.get(word) for word in sentence])
                #input_seq.append([word2vec.get(word) for word in sentence])
            except:
                print word, "is unkwon word"
                continue

    test_f.close()
    del word2vec

    return train_dict, train_input, train_input_len, train_question, train_question_len, train_target, train_gate, test_input, test_input_len, test_question, test_question_len, test_target, test_gate

def batch_iter(data, batch_size):
    data = np.array(data, dtype=object)
    data_size = len(data)
    num_batches = int(data_size/batch_size) 
    for batch_num in xrange(num_batches):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size, data_size)
        # How to seperate each batches
        yield data[start_index:end_index]
