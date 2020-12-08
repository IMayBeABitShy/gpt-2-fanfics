# Use GPT-2 to generate fanfiction.

## What is the aim of this project?

This project aims to provide a simple way to generate fanfictions using the GPT-2 algorithm.

## Requirements

- python3.7 (higher versions will likel not work due to gpt-2-simple requirements)
- all packages in `requirements.txt` installed
- `tensorflow` < 2.0, optionally the GPU version of the package
- a *ff2zim* project containing all fanfics you want to train to on
- (optional) EPUB files of the original story
- **Important:** ensure your hardware meet the GPT-2 requirements. These depend on the model you want to use and may be *very* high.

## Usage

This section walks you trough the process of generating fanfiction using GPT-2.

### Step 0: Preparation

Before we can start, there are still some things to do.

**1. Ensure that:**

- All required dependencies are installed.
- You have a ff2zim project containing your data source.

**2. Terms and variables used**

Many of the specifics on how to do this depend on your exact setup like directory structure. To make things simpler, here are some variables/placeholders we use and how you have to fill them in.

As a rule of thumb, whenever this document uses an ALL_CAP name possibly containing underscores, it refers to a value you have to subsitute with a specific value.

Here is the list:

| Name                     | Value                                                        |
| ------------------------ | ------------------------------------------------------------ |
| `PATH_TO_FF2ZIM_PROJECT` | The path to the ff2zim project containing the fanfics you want to train on. |
| `PATH_TO_TRAININGFILE`   | Path to the trainingfile. The trainingfile will be created during this process and written to the specified path. Please ensure that the parent directory exists. For most purposes, `trainingfile.txt` will be enough. |
| `MODEL`                  | GPT-2 model to use. Can be either `124M`, `355M`, `774M` or `1558M`. Larger values will yield better results, but require way better hardware to finetune and generate. Generally speaking, it is unlikely that your hardware meets the requirements to finetune non-124M models with GPU acceleration, though it is not impossible. |
| `RUN_NAME`               | Think of this value as the identifier for the finetuned model. Something like `MyFanfictionModel`. Using different values will  use different checkpoints ("saves", so to speak). |
Remember to escape whitespaces in arguments.




### Step 1: Generate the trainingfile

GPT-2 is a *large* text prediction model. Retraining, or rather *finetuning* it requires a different dataset from which it will learn. In this step, we generate the *trainingfile*, which is a file containing the data to finetune GPT-2 on.

Simply run `python3.7 generate-trainingfile PATH_TO_FF2ZIM_PROJECT PATH_TO_TRAININGFILE`. Remember to substitute the placeholders.

**Arguments:**

- `PATH_TO_FF2ZIM_PROJECT`: The path to your ff2zim project. Remember to escape whitespace, etc...
- `PATH_TO_TRAININGFILE`: the path to write the trainingfile to. Normally, `trainingfile.txt` will be enough for most purposes.
- (optional) `--add-epub [PATH [PATH ...]]`: If you want to include source material, you can specify this argument to include one or more EPUBs. Simply replace PATH (and the `[]...`) with the path to the EPUB files. If a path points to a directory, all contained EPUBs will be included.

Congratulations, you now have a trainingfile containing the text to train GPT-2 on. If you are experienced with GPT-2 already, you probably know ho to proceed from here on.

### [Optional] Step 2: Encode the trainingfile

This step is optional.

Basically, starting the finetuning process on the raw data takes a significant amount of time just for loading the dataset. But worry not, this step solves the problem.

Instead of pre-processing the dataset everytime we (re)start the finetuning process, we can simply do this one time manually. This will save some time later on.

Simply run `python3.7 finetuner.py PATH_TO_TRAININGFILE NEW_PATH MODEL`, where `NEW_PATH` is the path to save the encoded trainingfile to.

**Afterwards, whenever we refer to the `PATH_TO_TRAININGFILE`, we now mean the `NEW_PATH` you have specified.**

### Step 3: Finetune GPT-2

**Note: this step takes a loooong time and muuuch computation power. If you have some knowledge about programming, you can use the free [Google Colaboratory Notebook](https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce) (made by [the gpt-2-simple authors](https://github.com/minimaxir/gpt-2-simple)) to train in the cloud.**

It's finally time, now we can actually teach GPT-2 to generate fanfiction for your. However, this will take a long time. More specifically, it will depend on whether you are finetuning GPT-2 using a GPU and which model size you use. Finetuning GPT-2's 124M model with a GPU can be done within a day, but it will take much longer when only using a CPU.



Now, before continuing, **read this:** 

First, to be *really* clear about this , I am not kidding when I say that this will take a significant amount of time. More specifically, this will probably run until you cancel it. During this process, each *generation* the network learns to improve its generation. So letting it run longer will yield better results. Every 100 generations it will output the current generation and some statistics. Every 500 generations it will print a sample generated text and save the model. It should also save when you cancel, so canceling may take some time.



To perform this step, simply run `python3.7 finetuner.py finetune --model MODEL --run-name RUN_NAME PATH_TO_TRAININGFILE`.  I recommend to cancel the process using ctrl+c once you have reached a good amount of generations. Personally, I use 30000 generations as a baseline.



### Step 4: Generate the fanfiction

Now it is time to generate some fanfiction. **Be advised that a large amount of the results will be crappy. Just retry a couple of times, you won't have to retrain.**

There are multiple ways to do this, some of them listed below. You'll likely need something called a *prompt* . A *prompt* is basically just the start of the story. It can be a couple of sentences long, but using a simple prompt like `Anna kisses Bob and tells him ` (trailing space) will yield good enough results.

**Using this project:**

This project already includes a simple way to do this. However, there are multiple modes and you'll have to decide for one:

- `complete` is the simplest mode. It will take a simple prompt and try to complete it.
- `chapter` is a bit more complex. It will interpret the prompt as the description of a story and will attempt to complete the text until it reaches a mark indicating that a chapter has ended. Be warned that this may take a bit of time as the code will attempt to filter out bad results and may even delete already generated text sections.
- `story` is like `chapter`, but will complete until it reaches a mark indicating the end of the story.



**Using a Web UI**

I've created a simple WebUI for completing text. You can get it [here](https://github.com/IMayBeABitShy/gpt-2-simple-webui).

After downloading it and installing all requirements, simply run it using `python3.7 webeditor.py -i localhost -p 8080 --run-name RUN_NAME`. Then open `https://localhost:8080/` in your webbrowser. Type text in the left field, then press TAB and wait. You should now see the completed text.

## Acknowledgements

This project is just a wrapper around [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple), which uses OpenAI's [GPT-2 model](https://openai.com/blog/better-language-models/), as well as [ff2zim](https://github.com/IMayBeABitShy/ff2zim), which is a wrapper around [FanFicFare](FanFicFare).