# llm

Ground-up implementation of LLMs with `NumPy` and `regex` as the only dependencies.

The purpose of this repo is educational - explore the main concepts of LLMs without relying on
3rd party packages for the heavy lifting. In particular, the computation of gradients (i.e., the
"backward pass") is done manually, instead of relying on autograd systems, like the ones in
PyTorch or Tensorflow. The Byte-Pair Encoding (BPE) algorithm is also manually implemented with
no dependencies other than a regex library, and supports training the merges.

A good amount of effort has been spent on making all aspects of the BPE tokenizer (train, encode, decode) fast through algorithmic optimizations and by implementing the core operations in Python extension modules with Cython.


## Quick Start

1. **Setup a virtual environment** and activate it. Code has been tested on Python 3.12. To avoid needing to set the PYTHONPATH each time, the `export` command can be added to the `activate` script.
```shell
python3.12 -m venv venv
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
make install_requirements
```

2. **Compile Cython extension modules**
```shell
make python_extensions
```

3. **Run tests**
```shell
make test
```

4. **Download data** This downloads a text file containing all of Shakespeare's plays. It is about 5MB. The raw text and splits can be found in `assets/text`.
```shell
make download_text
```

5. **Train the BPE tokenizer merges**  This uses the training split to learn the merge order. The trained BPE tokenizer merge order and vocab can be found in `assets/bpe_checkpoints/default_10k`.
```shell
python scripts/train_tokenizer.py -y -n default_10k
```

6. **Tokenize the data splits** Use the trained tokenizer to convert the text data splits to a sequence of tokens. These are then directly used for training the model. The tokenized splits can be found in `assets/tokens/default_10k`.
```shell
python scripts/tokenize_splits.py -n default_10k
```

7. **Train the Transformer model** Trains a small model of 1.6M parameters. Hyperparameters can be changed by modifying the model constructor in the script. The script has the ability to continue training from a checkpoint with a different learning rate, batch size, and number of batches. The trained model checkpoints can be found in `assets/model_checkpoints/{v1, v2}`.
```shell
python scripts/train_model.py -y -n v1 -t default_10k -bs 4 -nb 100
python scripts/train_model.py -y -n v2 -t default_10k -bs 4 -nb 100 -s v1 -c 100 -lr 0.0001
```

8. **Generate output from the model**:
```shell
python scripts/generate_text.py -t default_10k -n v2 -c 100
```


## Sample Generation

Here's an example of the generation output for a model trained on the works of Shakespeare. It uses a vocabulary of 10,256 tokens, also trained on the same source text, and context size of 128. It has 1.6M paramers, 1,000x smaller than GPT-2, and 100,000x smaller than GPT-3. It was trained on 100,000 randomly sampled batches of size 16 with decaying learning rate.

The context window was seeded with `"ACT I. Scene I.\n\n"` for generation:
```text
ACT I. Scene I.


Enter ANGELO with a letter

  ISABELLA. The man of friar should never be affords,
    But holy sir, in Vienna, I foe,
    Come to light again, if not one y'are mad,
    I shall be reveng'd i' his bed.
  ANGELO. I pray ye show myself.
    I have leave you, sir! I have heard of your justice
    Brid your affection.
  LUCIO. [To ISABELLA] You know your private guation is
    A gentleman of his knowledge; if your reason
    Will put yourself into an intent
    When he is in love to good manners, as oft
    It might not satisfy them. Heaven increase he
    That wears by us thus to give 'em. Yet tell me
    These griefs and not what he would serve my bear
    A lord of France! The deputy leads him
    In a wonder who waits the husband,
    Is left the three-alloon, that he hath fought,
    And takes this a flap of brazen pay and hate
    Runs by his vacant in his cursed wrath.
    So, March have we all against a line
    Be heat for mercy left enemy to me,
    And pray'rs if you were mistaken to show,
    Such French got Sackles thy cheeks and ghosts,
    Even, to stop their tags, in despiteful truth,
    When thou disasters of Salisbury fought,
    And to the nobility of the world,
    When thousands vanquisherly besiege,
    Poor fiery sceptresadoes the minor stands,
    The Bishop, his and quantityilties,
    Who writ behaviour! He hath no actor drops
    Cleopatra traitor; in onehere I learn'd
    To any o' so wise with the heavens; which we call
    The loveful fathers prick'd, perus'd by rushes-
    For husbands, which should not bear a wild man,
    Mine honour'd is but weak as many wrong.
  DUNCAN. Welcome, Queen.
  DUNCAN. Guildenstern sadly yourselves.
    Is that the tyrant my master made true?
  BRUTUS. Why, Sir?
  LENNOX. To draw aside; he is heavy, and therefore
    ARCHBISHOP Hunted.
  MACBETH. I am eag strangers
    He'll seek his laws, and meet us with none.
                          Enter Banquo.

  BANQUO. Hail, fair countrymen.
    These bear superfluous
```
Though the text is non-sensical, this very small model has captured the cadence and structure of a Shakespeare play, understanding the essence of dialogue and play directions. Some modeling of dependencies is also captured, with the character Banquo entering and then speaking near the end of the generation.


## Serving Requests
The `serving` directory contains a very basic example of performing tokenization and text completions as a service. This is also illustrative for better understanding tokenization.

Once you've trained a tokenizer and a model, from one terminal run:
```shell
python serving/server.py -t default_10k -n v2 -c 100
```

From another terminal run:
```shell
python serving/client.py -e tokenize
...
Enter text to tokenize:MACBETH
    77: [M]
   609: [AC]
 1,243: [BE]
   538: [TH]
Enter text to tokenize: MACBETH
 1,410: [ MACBETH]
...
```
The difference in tokenization here occurs because of the logic of the GPT-4 split pattern used to train the tokenizer, which attaches leading whitespace to split words.

## Repo Structure

- `llm`: Contains a library implementation of a Transformer model architecture and BPE tokenizer
- `scripts`: Implements various functionalities on top of the core libraries for training and generation
- `serving`: Sample server/client implementation for tokenization and text-completion as a service


## Future Work

- Add support for special tokens in `RegexTokenizer`
- Convert the plain Python dictionary to an LRU cache in `RegexTokenizer` so it can be used in a serving system.
- Speed up text generation in `Transformer` using a KV-cache

