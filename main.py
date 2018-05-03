import argparse
from myPackage import tools as tl
from myPackage import preprocess
from myPackage import minutiaeExtraction as minExtract


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="-p Source path where the images are stored.")
    args = vars(ap.parse_args())

    # Configuration
    image_ext = '.tif'
    plot = True
    ratio = 0.2
    # Extract names
    all_images = tl.natSort(tl.getSamples(args["path"], image_ext))
    # Split train and test data
    train_data, test_data = tl.split_train_test(all_images, ratio)
    print("All_images size: {}\n"
          "Train_data size: {}\n"
          "Test_data size: {}\n".format(len(all_images), len(train_data), len(test_data)))
    for image in train_data[:3]:
        bin_img_inv = preprocess.cleanImage(image, plot)
        skeleton = preprocess.process_skeleton(bin_img_inv, plot)
        # skeleton = preprocess.thinning(bin_img_inv, plot)
        # skeleton *= 255
        minExtract.process(skeleton, plot)