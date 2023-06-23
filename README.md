# text-generation

This is example project for training
text generation model using Tensorflow.

Additionally, you can follow
[this](https://www.tensorflow.org/text/tutorials/text_generation)
tutorial from official Tensorflow documentation.

To be able to reuse trained model, weights
are saved in checkpoints (it still may not know if it's last epoch).

## How to use

1. Install project dependencies:

    - via `pipenv`

        ```bash
        pipenv shell
        pipenv install
        ```

    - globally

        ```bash
        pip install numpy tensorflow
        ```

2. Run the project

    ```bash
    python main.py
    ```

## Example output

```txt
Hello, da Big World!" use print() function in Python in Python, Tea the progress of eagress of entire world!
Who is da Biggy Worldy?
Yes, the programprint mmanctien.
Pmamdiig
Pmamdicting the future.
```
