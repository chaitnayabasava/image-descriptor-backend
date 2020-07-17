def clean_sentence(output, vocab):

    words_sequence = []

    for i in output:
        # if i == 1:
        #     continue
        words_sequence.append(vocab.idx2word[i])

    words_sequence = words_sequence[1:-1]
    if(words_sequence[-1] != "."):
        sentence = " ".join(words_sequence)
    else:
        sentence = " ".join(words_sequence[:-1])
        sentence += "."
    sentence = sentence.capitalize()

    return sentence
