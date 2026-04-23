from download import run_download
from generate_paraphrases import run_generate_paraphrases
from align_spans import run_align_spans
from build_mia_dataset import run_build_mia_dataset

def main():
    run_download()
    run_generate_paraphrases()
    run_align_spans()
    run_build_mia_dataset()

if __name__ == "__main__":
    main()