def import_fix():
    id        = "auvsi_suas"
    path      = os.getcwd()
    path_list = path.split(os.path.sep)
    dir_path  = os.path.sep.join(path_list[:path_list.index(id)])
    sys.path.append(dir_path)

import_fix()
print("shouldn't something be happening?...")
