class InputExample(object):
    # a single training / test example for simple sequence classification
    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Construct s input Example.
        Args:
            guid: unqiue id for the example.
            text_a: string, the untokenzied text of the first seq. for single sequence
                tasks, only this sequction msut be specified.
            text_b: (Optional) string, the untokenized text of the second sequence.
            label: (Optional) string, the label of the example, This should be specifi
                for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class GroupFeature(object):
    # a single set of features of data
    def __init__(self, doc_tokens, q_type, entity_type, relations, input_features):
        self.doc_tokens = doc_tokens
        self.type = q_type
        self.entity_type = entity_type
        self.relations = relations
        self.input_features = input_features


class InputFeature(object):
    # a single set of features of data
    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_mask, valid_id):
        self.input_ids = input_ids # ques_i + doc_token
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_mask = label_mask
        self.valid_id = valid_id


class MRCExample(object):
    """A single training/test example for the Squad dataset."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 entity_type,
                 q_type,
                 relations,
                 label=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.label = label
        self.q_type = q_type
        self.entity_type = entity_type
        self.relations = relations

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        s += ", label: [%s]" % (" ".join(self.label))
        return s