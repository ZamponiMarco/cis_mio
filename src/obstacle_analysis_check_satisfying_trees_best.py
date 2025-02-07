from analysis.tree_selection import process_directories

BASE_DIR_SOURCE = "resources/obstacle/"
BASE_DIR_TARGET = "resources/verified_obstacle/"
DIRECTORIES = ["S1", "S2", "S3", "S4"]

if __name__ == "__main__":
    process_directories(BASE_DIR_SOURCE, BASE_DIR_TARGET, DIRECTORIES)
