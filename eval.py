# _*_coding: utf-8_*_

import os
import sys
import argparse
from pyrouge import Rouge155
import config as config


def rouge_log(args, results_dict, dir_to_write):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)
    if not os.path.exists(dir_to_write):
        os.mkdir(dir_to_write)

    results_file = os.path.join(dir_to_write, args.mode+".ROUGE_results.txt")
    print("Writing final ROUGE results to %s..." % (results_file))
    with open(results_file, "a+") as f:
        f.write(args.model_name)
        f.write("\n")
        f.write("----"*10)
        f.write(log_str)
        f.write("****" * 10)
        f.write("\n\n")


def getRouge_score():
    file_path = os.path.join("results", config.dataset)
    r = Rouge155()
    r.system_dir = os.path.join(file_path, "predict")
    r.model_dir = os.path.join(file_path, 'ref')
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'

    command = '-e /home/ghost/metrics/rouge/tools/ROUGE-1.5.5/data/ -a -c 95 -m -n 2 -s'
    output = r.convert_and_evaluate(rouge_args=command)
    # output = r.convert_and_evaluate()
    # print(r.output_to_dict(output))
    return r.output_to_dict(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input process model name")
    parser.add_argument("--model_name", default="debug")
    parser.add_argument("--mode", default="cnndm-pg")

    args = parser.parse_args()
    
    config.dataset = args.mode+"-"+args.model_name.split("/")[-1].split("-")[-1]
    print(config.dataset)
    save_rouge_path = "rouge_results" 
    output = getRouge_score()
    rouge_log(args, output, save_rouge_path)


