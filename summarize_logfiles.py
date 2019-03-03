import os

import argparse
import ast
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--logdir', type=str, default=None, help="Place all the run.sh perturb outputs in a logdir")
parser.add_argument('--target_filetype', type=str, default='csv')


args = parser.parse_args()

class Converter(object):
    def __init__(self, args):
        self.args = args
        self.logdir = args.logdir
        self.filenames = [join(self.logdir, f) for f in listdir(self.logdir) if isfile(join(self.logdir, f)) and ".txt" in f and "last_few_only" not in f]
        self.last_few_only_filenames = [join(self.logdir, f) for f in listdir(self.logdir) if isfile(join(self.logdir, f)) and ".txt" in f and "last_few_only" in f]

        self.valid_metrics = ['ppl'] #['f1', 'bleu']
        self.metric_output_types = ["exact", "delta"]
        desired_perturb_types = ["only_last", "drop_first", "drop_last", "shuffle", "reverse_utr_order", "verbdrop_random", "noundrop_random", "worddrop_random", "wordshuf_random", "wordreverse_random"]
        self.parsed_data = {}
        assert self.filenames
        for filename in self.filenames:
            self.parsed_data[filename] = self.parse(filename, desired_perturb_types)
        for filename in self.last_few_only_filenames:
            self.parsed_data[filename] = self.parse(filename)
            

    def parse(self, filename, desired_perturb_types=None):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        data = {}
        for metric_output_type in ["exact", "delta", "percent"]:
            data["metric_{}_values".format(metric_output_type)] = []
        data["metric_types"] = []
        
        lid = 0
        while lid < len(lines):
            if "MODELTYPE" in lines[lid]:
                data['modeltype'] = lines[lid].split(":")[-1].strip()
            if "CONFIG :" in lines[lid]:
                perturb_type = lines[lid].split("test_")[-1]
                if desired_perturb_types:
                    if perturb_type not in desired_perturb_types and "NoPerturb" not in perturb_type:
                        lid += 1
                        continue
                assert "FINAL_REPORT: " in lines[lid+1]
                metrics_str = lines[lid+1].split("FINAL_REPORT: ")[-1]
                metrics = ast.literal_eval(metrics_str)
                for metric_type in self.valid_metrics:
                    data["metric_exact_values"].append("{0:.2f}".format(metrics[metric_type]))
                    data["metric_types"].append("{}_{}".format(perturb_type, metric_type))
                    if "NoPerturb" in perturb_type:
                        data["metric_noperturb_{}".format(metric_type)] = metrics[metric_type]
                        data["metric_percent_values"].append("{0:.2f}".format(metrics[metric_type]))
                        data["metric_delta_values"].append("{0:.2f}".format(metrics[metric_type]))
                    else:
                        delta_change = metrics[metric_type] - data["metric_noperturb_{}".format(metric_type)]
                        try:
                            percent_change = (delta_change * 100.0) / data["metric_noperturb_{}".format(metric_type)]
                        except:
                            percent_change = 0.0
                        
                        data["metric_percent_values"].append("{0:.2f}".format(percent_change))
                        data["metric_delta_values"].append("{0:.2f}".format(delta_change))
                        
                        
            lid += 1
        return data

    def convert(self):
        if self.args.target_filetype == "csv":
            print("Conversting to a csv file...")
            self.convert_to_csv(self.filenames, "logs_for_table.csv")
            self.convert_to_csv(self.last_few_only_filenames, "logs_for_plot.csv")
        elif self.args.target_filetype == "latex":
            self.convert_to_latex_style(self.filenames, "logs_for_table.latex")
        else:
            assert "unsupported type : {}. Valid : csv".format(self.args["target_filetype"])

    def convert_to_csv(self, filenames, target_filename):
        target_filename = join(self.logdir, target_filename)
        with open(target_filename, 'w') as f:   
            for metric_output_type in self.metric_output_types:
                for i, filename in enumerate(filenames):
                    parsed_data = self.parsed_data[filename]
                    if i == 0:
                        line1 = "Model({}), ".format(metric_output_type) + " , ".join(parsed_data["metric_types"])
                        f.write("{}\n".format(line1))
                    line = "{}, ".format(parsed_data['modeltype']) + " , ".join(parsed_data["metric_{}_values".format(metric_output_type)])
                    f.write("{}\n".format(line))
                    print("Writing line {}".format(i+1))
            print("Done writing to {}".format(target_filename))

    def convert_to_latex_style(self, filenames, target_filename):
        target_filename = join(self.logdir, target_filename)
        with open(target_filename, 'w') as f:
            for metric_output_type in self.metric_output_types:
                for i, filename in enumerate(filenames):
                    parsed_data = self.parsed_data[filename]
                    if i == 0:
                        line1 = "Model({}), ".format(metric_output_type) + " & ".join(parsed_data["metric_types"])
                        f.write("{}\n".format(line1))
                    line = "{}, ".format(parsed_data['modeltype']) + " & ".join(parsed_data["metric_{}_values".format(metric_output_type)])
                    f.write("{}\n".format(line))
                    print("Writing line {}".format(i+1))
            print("Done writing to {}".format(target_filename))
                    
if __name__ == "__main__":
    converter = Converter(args)
    converter.convert()
