import argparse
import shortuuid
import os
from buddy.data_creator import DataCreator
from buddy.trainer import Trainer
from buddy.interactive_runner import InteractiveRunner

def run_create(content_format, imgdir, output_dir, output_fname):
    dc = DataCreator()
    if content_format == "static":
        dc.capture_via_static(output, imgdir)
    else:
        dc.capture_via_cam(output_dir, output_fname)

def run_train(training_file, model_type, output_dir, output_fname):
    trainer = Trainer(training_file, output_dir, output_fname)

    if model_type == "rf":
        trainer.create_random_forest_model()
    elif model_type == "nn":
        trainer.create_neural_net_model()

def run_interactive(model):
    interactive_runner = InteractiveRunner(model)
    interactive_runner.run()
    pass

def main(): 
    parser = argparse.ArgumentParser(description='Mode for running')

    parser.add_argument('mode', type=str, \
    help='selected mode for using gesture recognition library tool')

    parser.add_argument('--content', type=str, \
    help='format of content being captured upon data creation (live/static)', \
    default="live")

    parser.add_argument('--model-type', type=str, \
    help='model classifier to build in training', \
    default="rf")

    parser.add_argument('--training-file', type=str, \
    help='File to get training data from', \
    default="~")

    parser.add_argument('--model', type=str, \
    help='path to a joblib compatible model to use for interactive mode')

    parser.add_argument('--imgdir', type=str, \
    help='directory with images for capturing training data points', default=f"{os.getcwd()}")

    parser.add_argument('--output-fname', type=str, \
    help='output file name for training data points csv or ML model', default=shortuuid.uuid())

    parser.add_argument('--output-dir', type=str, \
    help='output directory for training data points csv or ML model', default=f"{os.getcwd()}")
    
    args = parser.parse_args()

    if args.mode == "create":
        run_create(args.content, args.imgdir, args.output_dir, args.output_fname)
    
    if args.mode == "train":
        run_train(args.training_file, args.model_type, args.output_dir, args.output_fname)

    if args.mode == "run-interactive":
        run_interactive(args.model)

if __name__ == "__main__":
    main()
