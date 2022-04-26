MAP2VIS = {
    'OK': 'OK',
    'AirRoomShake': 'SK',
    'Dead': 'DEAD',
    'Empty': 'EM',
    'NoAirRoom': 'NORM',
    'Split': 'ST',
    'Weak': 'WK',
    'Flower': 'FR'
}  # shorthand for better visualization

FILTER_CLASSES = []

CLASS_NAMES = [
    'OK', 'AirRoomShake', 'Dead', 'Empty', 'NoAirRoom', 'Split', 'Weak',
    'Flower'
]

CLASS_NAMES = [name for name in CLASS_NAMES if name not in FILTER_CLASSES]

VIS_ALL_LABELS = [MAP2VIS[NAME] for NAME in CLASS_NAMES]

VIS_BINARY_LABELS = ['OK', 'NO_OK']

BN_CLASS_NAMES = ['OK', 'NoOK']

HEADER_NAMES = ['filename'] + CLASS_NAMES

THRESH = [0.9, 0.8, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

BASE_HEADERS = ['Fold', 'Rank', 'Epoch', 'Step']

MODEL_SELECTION_RECORD_HEADERS = BASE_HEADERS + [
    'Loss', 'ROC_AUC', 'Other_Loss', 'OTHER_ROC_AUC'
]

PERFORMANCE_RECORD_HEADERS = BASE_HEADERS + [
    'class', 'precision', 'in_precision', 'recall', 'f1-score', 'support'
]


def update_by_filter_names(hparams):
    filter_classes = hparams.filter_classes
    classes = hparams.classes
    global FILTER_CLASSES, CLASS_NAMES, VIS_ALL_LABELS, HEADER_NAMES
    FILTER_CLASSES = filter_classes
    CLASS_NAMES = [name for name in classes if name not in filter_classes]
    VIS_ALL_LABELS = [MAP2VIS[NAME] for NAME in CLASS_NAMES]
    HEADER_NAMES = ['filename'] + CLASS_NAMES
    hparams.classes = CLASS_NAMES
