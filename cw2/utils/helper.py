import json
from collections import defaultdict, OrderedDict
from itertools import repeat
import numpy as np
import uuid

from sklearn import metrics


def read_jsonl(data_path):
    with open(data_path) as fh:
        for line in fh:
            yield json.loads(line)


class MistakeClass:
    def __init__(self, expected, predicted):
        self.expected = expected
        self.predicted = predicted

    def __repr__(self) -> str:
        return f"{self.expected}->{self.predicted}"

    def __hash__(self):
        return hash((self.expected, self.predicted))

    def __eq__(self, other) -> bool:
        if not isinstance(other, MistakeClass):
            return False
        return self.expected == other.expected and \
            self.predicted == other.predicted


class Mistake:
    def __init__(self, expected, predicted, word, context):
        self.expected = expected
        self.predicted = predicted
        self.word = word
        self.context = context

    def __repr__(self):
        return f"{self.expected}->{self.predicted}\t{self.word}\t{' '.join(self.context)}"


def get_classification_report(labels, predictions, top_k=None, criterion='support', reverse=True,
                              chunks=None, ctx_chunks=None, name=None, **kwargs):
    """Get sorted classification report, optionally filtering the top-k classes.

    Args:
        labels (List): target labels.
        predictions (List): predicted labels.
        top_k (int, optional): number of classes to include in the report. Defaults to None (all classes).
        criterion (bool, optional): report key to be used as criterion to filter classes.
        (precision | recall | f1-score | support). Defaults to 'support'.
        reverse: (bool, optional): if True return items sorted in reverse order.

    Returns:
        [type]: [description]
    """
    report = metrics.classification_report(labels, predictions, **kwargs)
    report_dict = metrics.classification_report(labels, predictions, output_dict=True, **kwargs)
    # report_dict['accuracy'] = {'accuracy': report_dict['accuracy'], 'support': report_dict['macro avg']['support']}
    report_dict.pop('accuracy')

    mistakes = defaultdict(list)
    corrects = defaultdict(int)
    print(f"Complete mistakes:")
    for label, prediction, chunk, ctx in zip(labels, predictions,
                                             chunks or repeat("UNK"),
                                             ctx_chunks or repeat(["UNK"])):
        if label != prediction:
            mistake_class = MistakeClass(label, prediction)
            mistake = Mistake(label, prediction, chunk, ctx)
            mistakes[mistake_class].append(mistake)
        else:
            corrects[label] += 1

    mistakes = OrderedDict(sorted(mistakes.items(),
                                  key=lambda kv: len(kv[1]),
                                  reverse=True))
    with open(f"mistakes_{name or 'unnamed'}_{uuid.uuid4()}.txt", "w+") as f:
        for mistake_group in mistakes.values():
            for mistake in mistake_group:
                f.write(str(mistake))
                f.write("\n")
    print(f"Most common mistakes:")
    for kv in list(mistakes.items())[:10]:
        print(f"{kv[0]}\t{len(kv[1])}")
    print(f"Correct by label:")
    for label in sorted(np.unique(labels)):
        print(f"{label}: {corrects[label]}/{labels.count(label)}")
    print(f"Total correct: {sum(corrects.values())}")
    print(f"Len of dataset: {len(labels)}")

    if criterion:
        report_dict = {k: v for k, v in sorted(report_dict.items(), key=lambda item: item[1][criterion], reverse=reverse)}

    report_keys = [k for ii, (k, v) in enumerate(report_dict.items()) if top_k is None or ii < top_k]
    report_keys += ['accuracy', 'support']
    report_lines = report.split('\n')
    report = '\n'.join([line for line in report_lines if any([key in line for key in report_keys])])
    return report
