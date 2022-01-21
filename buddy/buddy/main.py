import argparse
import os
from data_creator import DataCreator

def run_create(content_format, imgdir, output):
    dc = DataCreator()
    if content_format == "static":
        dc.capture_via_static(output, imgdir)
    else:
        dc.capture_via_cam(output)

def main(): 
    parser = argparse.ArgumentParser(description='Mode for running')

    parser.add_argument('mode', type=str, \
    help='selected mode for using gesture recognition library tool')

    parser.add_argument('--content', type=str, \
    help='format of content being captured upon data creation (live/static)', \
    default="live")

    parser.add_argument('--imgdir', type=str, \
    help='directory with images for capturing training data points', default=f"{os.getcwd()}/TestData/static")

    parser.add_argument('--output', type=str, \
    help='output file name for training data points csv', default="gesture_train")

    args = parser.parse_args()

    if args.mode == "create":
        run_create(args.content, args.imgdir, args.output)
            
main()
