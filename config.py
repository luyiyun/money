ROOT_DIR = 'G:/dataset/热身赛-面额识别'
TRAIN_LABEL_CSV = 'train_face_value_label.csv'
TRAIN_DIR = 'train_data'
TEST_DIR = 'public_test_data'
LABELS = [0.1, 0.2, 0.5, 1., 2., 5., 10., 50., 100.]
LABEL_MAPPER = {l: i for i, l in enumerate(LABELS)}
