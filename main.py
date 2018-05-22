import argparse
from myPackage import tools as tl
from myPackage import preprocess
from myPackage import minutiaeExtraction as minExtract
from enhancementFP import image_enhance as img_e
from os.path import basename, splitext, exists
import time
from numpy import mean, std


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="-p Source path where the images are stored.")
    ap.add_argument("-r", "--results", required= False,
                    help="-r Destiny path where the results will be stored.")
    args = vars(ap.parse_args())

    # Configuration
    image_ext = '.tif'
    plot = False
    path = None
    # ratio = 0.2
    # Create folders for results
    # -r ../Data/Results/fingerprints
    if args.get("results") is not None:
        if not exists(args["results"]):
            tl.makeDir(args["results"])
        path = args["results"]
    # Extract names
    all_images = tl.natSort(tl.getSamples(args["path"], image_ext))
    # Split train and test data
    # train_data, test_data = tl.split_train_test(all_images, ratio)
    print("\nAll_images size: {}\n".format(len(all_images)))
    all_times= []
    for image in all_images:
        start = time.time()
        name = splitext(basename(image))[0]
        print("\nProcessing image '{}'".format(name))
        cleaned_img = preprocess.blurrImage(image, name, plot)
        enhanced_img = img_e.image_enhance(cleaned_img, name, plot)
        cleaned_img = preprocess.cleanImage(enhanced_img, name, plot)
        # skeleton = preprocess.zhangSuen(cleaned_img, name, plot)
        skeleton = preprocess.thinImage(cleaned_img, name, plot)
        minExtract.process(skeleton, name, plot, path)
        all_times.append((time.time()-start))
    mean = mean(all_times)
    std = std(all_times)
    print("\n\nAlgorithm takes {:2.3f} (+/-{:2.3f}) seconds per image".format(mean, std))