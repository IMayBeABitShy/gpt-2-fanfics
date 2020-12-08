"""
Use GPT-2 to generate fanfictions.

This requires you to have a ff2zim project with the fanfics first.
"""

import argparse
import os
import time
import sys

from ff2zim.project import Project
from ff2zim.epubconverter import Html2EpubConverter

from html2text import html2text
from fanficfare.htmlcleanup import removeAllEntities
import ebooklib, ebooklib.epub


# before importing gpt-2, set verbosity level if debug mode enabled
if "-d" in sys.argv or "--debug" in sys.argv:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gpt_2_simple as gpt2


# ----- tokens ---- #

TOKEN_STORY_START = "<|storystart|>"
TOKEN_STORY_END = "<|storyend|>"
TOKEN_DESCRIPTION_START = "<|descriptionstart|>"
TOKEN_DESCRIPTION_END = "<|descriptionend|>"
TOKEN_CHAPTER_START = "<|chapterstart|>"
TOKEN_CHAPTER_END = "<|chapterend|>"


class Converter(Html2EpubConverter):
    """
    Converter for converting a fanficfare story to a gpt-2 usable format.
    """
    def write(self, path=None):
        if path is None:
            path = os.path.join(self.path, "trainingfile.txt")
        with open(path, "w") as fout:
            self.append_to_file(fout)
        
    def append_to_file(self, f):
        """
        Append the story to a file-like object.
        
        @param f: file-like object to write to
        @type f: L{str}
        """
        story = self.getStory()
        # write story start token
        f.write("\n" + TOKEN_STORY_START + "\n")
        # write description
        description = story.getMetadata("description", removeallentities=True)
        f.write("\n" + TOKEN_DESCRIPTION_START + "\n")
        f.write(description)
        f.write("\n" + TOKEN_DESCRIPTION_END + "\n")
        # write chapters
        for chapter in story.getChapters():
            content = chapter["html"]
            processed = self.process_chapter_content(content)
            f.write("\n" + TOKEN_CHAPTER_START + "\n")
            f.write(processed)
            f.write("\n" + TOKEN_CHAPTER_END + "\n")
        f.write("\n" + TOKEN_STORY_END + "\n")
    
    def process_chapter_content(self, content):
        """
        Process a chapter content.
        
        @param content: content to process
        @type content: L{str}
        @return: the processed content
        @rtype: L{str}
        """
        content = removeAllEntities(content)
        content = html2text(content)
        return content



def extract_epub_text(path):
    """
    Extract text from an EPUB.
    
    @param path: path to EPUB
    @type path: L{str}
    @return: the text of the EPUB with the correct tokens.
    @rtype: L{str}
    """
    # read book
    book = ebooklib.epub.read_epub(path)
    text = "\n" + TOKEN_STORY_START + "\n"
    # metadata
    descriptions = book.get_metadata("DC", "")
    if descriptions:
        text += "\n" + TOKEN_DESCRIPTION_START + "\n"
        text += descriptions[0][0]
        text += "\n" + TOKEN_DESCRIPTION_END + "\n"
    # story content
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        itemhtml = item.get_content().decode("utf-8")
        itemtext = html2text(itemhtml)
        is_image = itemtext.strip().startswith("![") and "](" in itemtext
        is_backnote = "#back_note_" in itemtext
        if not is_image and not is_backnote and itemtext.strip():
            # write chapter
            text += "\n" + TOKEN_CHAPTER_START + "\n"
            text += itemtext
            text += "\n" + TOKEN_CHAPTER_END + "\n"
    # end book
    text += "\n" + TOKEN_STORY_END + "\n"
    return text
        


class GPT2Finetuner(object):
    """
    Class for finetuning a gpt-2 project with a ff2zim project.
    
    @param path: path to the project
    @type path: L{str}
    """
    def __init__(self, path, epubpaths=[]):
        self.path = path
        self.epubpaths = epubpaths
        self.project = Project(self.path)
    
    def get_paths_to_include(self):
        """
        Return a list of story directories to include.
        
        @return: a list of the story directories to include
        @rtype: L{list} of L{str}
        """
        all_metadata = self.project.collect_metadata()
        to_include = []
        for metadata in all_metadata:
            if self.filter_story(metadata):
                fp = os.path.join(self.path, "fanfics", metadata["siteabbrev"], metadata["storyId"])
                to_include.append(fp)
        return to_include
    
    def filter_story(self, metadata):
        """
        Decide whether a story should be included, depending on the metadata.
        
        @param metadata: metadata of story
        @type metadata: L{dict}
        @return: whether the story should be included
        @rtype: L{bool}
        """
        # filter non-english stories
        if metadata["language"].lower() != "english":
            return False
        # filter crossover fanfics
        if "," in metadata["category"]:
            return False
        # filter short&incomplete stories
        if metadata["status"].lower() == "in-progress" and metadata["numWords"] < 10000:
            return False
        return True
    
    def create_training_file(self, path):
        """
        Create a training file for gpt-2 and write it to path.
        
        @param path: path to write to
        @type path: L{str}
        @return: a tupple of (num_fanfics_included, num_epubs_included)
        @rtype: L{tupple} of (L{int}, L{int})
        """
        storypaths = self.get_paths_to_include()
        n_stories_included = len(storypaths)
        n_epubs_included = 0
        with open(path, "w") as fout:
            # include epubs
            if self.epubpaths:
                for p in self.epubpaths:
                    if os.path.isdir(p):
                        fpaths = [os.path.join(p, e) for e in sorted(os.listdir(p)) if e.lower().endswith(".epub")]
                    else:
                        fpaths = [p]
                    for fp in fpaths:
                        fout.write(extract_epub_text(fp))
                        n_epubs_included += 1
            
            # include projects
            for storypath in storypaths:
                converter = Converter(storypath, include_images=False)
                converter.parse()
                converter.append_to_file(fout)
        
        return (n_stories_included, n_epubs_included)


def is_looping(text):
    """
    Check if a text is looping.
    
    @param text: text to check.
    @type text: L{str}
    @return: True if the text is looping
    @rtype: L{str}
    """
    lines = text.split("\n")
    # remove empty lines
    lines = [line for line in lines if line.strip()]
    for i, line in enumerate(lines):
        if lines.count(line) >= 5:
            # line is repeating often, very likely looping
            return True
        if line in lines[max(0, i-3):i]:
            # line is present in one of the previous 3 lines.
            return True
        words = line.split(" ")
        for word in words:
            if len(word) == 1:
                # skip single characters, possible style related.
                continue
            if words.count(word) >= 10:
                # word very often repeating, likely looping
                return True
    return False

def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser(description="Finetune a GPT-2 model using ff2zim")
    parser.add_argument("-d", "--debug", action="store_true", help="show debug information")
    subparsers = parser.add_subparsers(dest="action", help="action to perform", required=True)
    
    # parser for generating trainingfile
    tfparser = subparsers.add_parser("generate-trainingfile", help="generate the trainingfile from a ff2zim project")
    tfparser.add_argument("project", help="path to ff2zim project")
    tfparser.add_argument("trainingfile", help="path to write trainingfile to")
    tfparser.add_argument("--add-epub", action="store", nargs="*", help="add an epub or a directory of epubs to the trainingfile", metavar="PATH", dest="epubpaths")
    
    # parser for encoding the trainingfile
    eparser = subparsers.add_parser("encode-trainingfile", help="encode a trainingfile for better performance")
    eparser.add_argument("trainingfile", help="path to trainingfile to encode")
    eparser.add_argument("outfile", help="path to write to")
    eparser.add_argument("model", help="model to encode for")
    
    # parser for finetuning
    finetuneparser = subparsers.add_parser("finetune", help="finetune a gpt-2 model using a trainingfile")
    finetuneparser.add_argument("trainingfile", help="path to trainingfile")
    finetuneparser.add_argument("--model", action="store", default="124M", help="model to use")
    finetuneparser.add_argument("--run-name", action="store", dest="runname", default="run1", help="run name for finetuned model.")
    
    # parser for generating
    genparser = subparsers.add_parser("generate", help="generate a sample with an interactive prompt")
    genparser.add_argument("--run-name", action="store", dest="runname", default="run1", help="run name for finetuned model.")
    genparser.add_argument("-n", "--numsamples", action="store", type=int, help="number of samples to generate", default=1)
    genparser.add_argument("-m", "--mode", action="store", choices=("story", "chapter", "complete"), default="story")
    
    ns = parser.parse_args()
    
    if ns.action == "generate-trainingfile":
        print("Generating trainingfile...")
        trainingfile = ns.trainingfile
        finetuner = GPT2Finetuner(ns.project, ns.epubpaths)
        num_stories, num_epubs = finetuner.create_training_file(trainingfile)
        print("Trainingfile successfully created.")
        print("Included: {} fanfics and {} epubs.".format(num_stories, num_epubs))
        return
    
    elif ns.action == "encode-trainingfile":
        print("Encoding trainingfile...")
        gpt2.encode_dataset(ns.trainingfile, out_path = ns.outfile, model_name=ns.model)
        print("Done.")
        return
    
    elif ns.action == "finetune":
        model = ns.model
        if not os.path.isdir(os.path.join("models", model)):
            print("Downloading the '{}' model...".format(model))
            gpt2.download_gpt2(model_name=model)
            print("Download finished.")
        print("Starting TF session...")
        sess = gpt2.start_tf_sess()
        print("TF session started.")
        print("Finetuning...")
        gpt2.finetune(
            sess,
            ns.trainingfile,
            model_name=model,
            run_name=ns.runname,
            print_every=100,
            sample_every=500,
            save_every=500,
            use_memory_saving_gradients=True,
            accumulate_gradients=1,
        )
    elif ns.action == "generate":
        prepend_story_start = False
        print("========== Generate a story ==========")
        if ns.mode in ("story", "chapter"):
            story_start = "\n" + TOKEN_STORY_START + "\n"
            description_s = "\n" + TOKEN_DESCRIPTION_START + "\n"
            description = input("Description of story: ")
            description_s += description + "\n" + TOKEN_DESCRIPTION_END + "\n"
            story_start += description_s + "\n" + TOKEN_CHAPTER_START + "\n"
            prepend_story_start = True
        elif ns.mode == "complete":
            story_start = input("Prompt: ")
        print("========== Generating... =========")
        print("Starting TF session...")
        sess = gpt2.start_tf_sess()
        print("TF session started.")
        print("Loading gpt-2...")
        gpt2.load_gpt2(sess)
        print("Loaded.")
        print("Generating: ", end="", flush=True)
        results=[]
        for i in range(ns.numsamples):
            finished = False
            storyparts = []
            while not finished:
                if not storyparts:
                    # first generation
                    prefix = story_start
                elif prepend_story_start:
                    # also include story start
                    prefix = description_s
                    prefix += " ".join(storyparts[-1].split(" ")[-21:-1])
                else:
                    prefix = " ".join(storyparts[-1].split(" ")[-21:-1])
                multisamples = True
                gpt2results = gpt2.generate(
                    sess,
                    run_name=ns.runname,
                    prefix=prefix,
                    return_as_list=True,
                    # nsamples=ns.numsamples,
                    seed=int(time.time()),
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    nsamples=(5 if multisamples else 1),
                )
                result = None
                for gpt2result in gpt2results:
                    gpt2result = gpt2result[len(prefix):]
                    if not is_looping(gpt2result):
                        result = gpt2result
                        break
                    if result is None:
                        # set default just to be sure
                        result = gpt2result
                
                if ns.debug:
                    print("=====")
                    print("#storyparts: ", len(storyparts))
                    if len(storyparts) > 0:
                        print("-----\nLast storypart: \n-----\n", storyparts[-1])
                    print("-----\nResult: \n-----\n", result)
                    print("=====")
                
                if ns.mode == "story" or ns.mode == "chapter":
                    if is_looping(result):
                        print("L", end="", flush=True)
                        # remove last part to reduce chance of looping
                        storyparts = storyparts[:-1]
                        continue
                    
                    # append result
                    storyparts.append(result)
                    if TOKEN_CHAPTER_END in result:
                        print("C", end="", flush=True)
                        if ns.mode == "chapter":
                            finished = True
                    elif TOKEN_STORY_END in result:
                        print("S", end="", flush=True)
                        finished = True
                    else:
                        print(".", end="", flush=True)
                elif ns.mode == "complete":
                    # set result
                    storyparts = [prefix + result]
                    finished = True
            # results.append(story[len(prefix):])
            results.append("".join(storyparts))
        print("\n", flush=True)
        for text in results:
            print("========= Result =========")
            print(text)


if __name__ == "__main__":
    main()
