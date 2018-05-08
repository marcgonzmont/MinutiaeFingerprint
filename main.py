import argparse
from myPackage import tools as tl
from myPackage import preprocess
from myPackage import minutiaeExtraction as minExtract
from enhancementFP import image_enhance as img_e
from os.path import basename, splitext, exists


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="-p Source path where the images are stored.")
    ap.add_argument("-r", "--results", required= False,
                    help="-r Destiny path where the results will be stored.")
    args = vars(ap.parse_args())

    # Configuration
    image_ext = '.tif'
    plot = True
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
    for image in all_images[:3]:
        name = splitext(basename(image))[0]
        print("\nProcessing image '{}'".format(name))
        cleaned_img = preprocess.blurrImage(image, name, plot, path)
        enhanced_img = img_e.image_enhance(cleaned_img, name, plot, path)
        cleaned_img = preprocess.cleanImage(enhanced_img, name, plot, path)
        # skeleton = preprocess.zhangSuen(cleaned_img, name, plot, path)
        skeleton = preprocess.thinImage(cleaned_img, name, plot, path)
        minExtract.process(skeleton, name, plot, path)