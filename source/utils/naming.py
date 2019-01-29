import os


def __robust_respath_search():
    """
    Resolve the path for resources from anywhere in the code.
    :return: The real path of the resources
    """
    curpath = os.path.realpath(__file__)
    basepath = curpath
    while os.path.split(basepath)[1] != 'source':
        newpath = os.path.split(basepath)[0]
        if newpath == basepath:
            print("ERROR: unable to find source from path " + curpath)
            break
        basepath = os.path.split(basepath)[0]
    return os.path.join(os.path.split(basepath)[0], "resources")


# ######### RESOURCES DIRECTORIES DEFINITION ###########

RESPATH = __robust_respath_search()
TBFOLDER = "tbdata"
MODELSFOLDER = "models"
DATASETFOLDER = "dataset"
FIVEKFOLDR = "../../../fivek"

def resources_path(*paths):
    """
    Very base function for resources path management.
    Return the complete path from resources given a sequence of directories
    eventually terminated by a file, and makes all necessary subdirectories
    :param paths: a sequence of paths to be joined starting from the base of resources
    :return: the complete path from resources (all necessary directories are created)
    """
    p = os.path.join(RESPATH, *paths)
    if os.path.splitext(p)[1] != '':
        basep = os.path.split(p)[0]
    else:
        basep = p
    os.makedirs(basep, exist_ok=True)
    return p

# ############################## BASE DIRECTORY-RELATIVE PATHS ###############


def tensorboard_path(*paths):
    """
    Builds the path starting where all tensorboard data should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(TBFOLDER, *paths)


def models_path(*paths):
    """
    Builds the path starting where all model data should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(MODELSFOLDER, *paths)


def dataset_path(*paths):
    """
    Builds the path starting where all datasets should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(DATASETFOLDER, *paths)


def fivek_path(*paths):
    """
    Builds the path starting where all fivek data should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(FIVEKFOLDR, *paths)


def fivek_element(idx, expert=None):
    """
    Builds the path starting where all fivek original data should be.
    :param idx: index of the frame to be returned.
    :param expert: index of expert to be searched. Original frame if None.
    :return: The path relative to this standard folder
    """

    if not isinstance(expert, int) or expert < 0 or expert > 4:
        exp = "original"
    else:
        exp = "expert%d" % expert
    return fivek_path(exp, "%d.png" % idx)


def fivek_dimension():
    """
    Checks the total number of available images of fivek data
    :return: the number of available images in the original folder of fivek. Int.
    """
    return len(os.listdir(fivek_path("original")))


if __name__ == '__main__':
    t1 = fivek_element(5, 8)
    t2 = fivek_element(890)
    t3 = fivek_element(31510, 0)
    show = lambda t: print((t, os.path.exists(t)))
    show(t1)
    show(t2)
    show(t3)
    print("Dataset dimension: %d" % fivek_dimension())




