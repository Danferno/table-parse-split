import os, shutil

def makeDirs(path, replaceDirs):
    try:
        os.makedirs(path)
    except FileExistsError:
        if replaceDirs == 'warn':
            raise Exception('Not yet implemented')
        elif replaceDirs:
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise FileExistsError