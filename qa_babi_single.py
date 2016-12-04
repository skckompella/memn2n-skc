
from memn2n import memn2n
from data_utils import load_tasks, vectorize_stories


PATH_TO_DATA = '/Users/skc/Projects/my_projects/memn2n-skc/data/bAbi/en'
NUM_TASKS = 20
TASK_ID = 1


def main():

    train_stories, test_stories = load_tasks(PATH_TO_DATA, TASK_ID)
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories +  test_stories)))

    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    print('---------------------------------------------------------')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of training stories:', len(train_stories))
    print('Number of test stories:', len(test_stories))
    print('---------------------------------------------------------')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train_stories[0])
    print('---------------------------------------------------------')
    print('Vectorizing the word sequences...')

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    S_train, Q_train, A_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)

    S_test, Q_test, A_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

    print('---------------------------------------------------------')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('S_train shape:', S_train.shape)
    print('inputs_test shape:', S_test.shape)
    print('---------------------------------------------------------')
    print('queries: integer tensor of shape (samples, max_length)')
    print('Q_train shape:', Q_train.shape)
    print('queries_test shape:', Q_test.shape)
    print('---------------------------------------------------------')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('A_train shape:', A_train.shape)
    print('answers_test shape:', A_test.shape)
    print('---------------------------------------------------------')
    print('Compiling...')
    model = memn2n(vocab_size, story_maxlen, query_maxlen)
    model.fit(S_train, Q_train, A_train)
    print('---------------------------------------------------------')
    print('Testing...')
    answers_predictions = model.predict(S_test, Q_test, A_test)
    print answers_predictions



if __name__ == '__main__':
    main()