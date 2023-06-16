import os, shutil
import click

def replaceDir(path):
    shutil.rmtree(path)
    os.makedirs(path)

def makeDirs(path, replaceDirs='warn'):
    try:
        os.makedirs(path)
    except FileExistsError:
        if replaceDirs == 'warn':
            if click.confirm(f'This will remove folder {path}. Are you sure you are okay with this?'):
                replaceDir(path)
            else:
                raise InterruptedError(f'User was not okay with removing {path}')
        elif replaceDirs:
            replaceDir(path)
        else:
            raise FileExistsError