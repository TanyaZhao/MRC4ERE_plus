import sys

sys.path.append("..")

import json
import collections
from .mrc_example import MRCExample, InputFeature, GroupFeature
from log.get_logger import logger
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer

def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            all_relations = paragraph["relations"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                questions = qa["questions"]
                label = qa["label"]
                q_type = qa["type"] # 问的是entity还是relation
                question_type = qa["entity_type"] # 问的是哪个类型的entity

                relations = []
                if q_type == "entity":
                    for relation in all_relations:
                        # if relation["label"] == question_type and relation["e1_text"] in questions[0]: #  比如问题是问work for的， relation中有work_for的关系，则加进来
                        tmp_relation = relation.copy()
                        relations.append(tmp_relation)

                example = MRCExample(
                    qas_id=qas_id,
                    question_text=questions,
                    doc_tokens=doc_tokens,
                    label=label,
                    q_type=q_type,
                    entity_type = question_type,
                    relations = relations # list of dict
                )
                examples.append(example)

                # if len(examples) == 20:
                #     return examples

    return examples


def convert_relation_examples_to_features(examples, tokenizer, label_list, max_seq_length, max_query_length, doc_stride):
    # load a data file into a list of "InputBatch"
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        group_features = []
        for query in example.question_text: # question text is not tokenized
            query_tokens = tokenizer.tokenize(query)
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[: max_query_length]

            all_doc_tokens = example.doc_tokens
            all_doc_labels = example.label # original label
            tok_to_orig_index = list(range(len(all_doc_tokens)))

            assert len(example.doc_tokens) == len(example.label)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            _DocSpan = collections.namedtuple(
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:  # add tokens
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                orig_all_doc_labels = all_doc_labels
                input_labels= ["[CLS]"] + ["O"] * len(query_tokens) + ["[SEP]"] \
                            + all_doc_labels + ["[SEP]"]
                label_ids = [label_map[tmp] for tmp in input_labels]

                assert len(label_ids) == len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    label_ids.append(label_map["O"])


                # while len(all_doc_labels) < max_tokens_for_doc:
                #     all_doc_labels.append("O")
                # label_ids = [label_map[tmp] for tmp in all_doc_labels]
                # label_ids = [label_map["[CLS]"]] + [label_map["O"]] * len(query_tokens) + [label_map["[SEP]"]] \
                #            + label_ids + [label_map["[SEP]"]]

                if len(label_ids) != max_seq_length:
                    print(len(orig_all_doc_labels))

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(label_ids) == max_seq_length

                group_features.append(
                    InputFeature(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_ids,
                        label_mask=label_mask,
                        valid_id=valid
                    )
                )

        features.append(group_features)

    return features


def convert_examples_to_features(examples, tokenizer, label_list, max_seq_length, max_query_length, doc_stride):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.doc_tokens
        labellist = example.label
        doc_tokens = []
        labels = []
        doc_valid = []
        label_mask = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            orig_to_tok_index.append(len(doc_tokens))
            doc_tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                tok_to_orig_index.append(i)
                if m == 0:
                    labels.append(label_1)
                    doc_valid.append(1)
                    # label_mask.append(1)
                else:
                    doc_valid.append(0)

        # annotate at 11.20
        # for relation in example.relations:  # {"e1_ids": "e2_ids": ...}
        #         e1_ids = [orig_to_tok_index[_id] for _id in relation["e1_ids"]]
        #         e2_ids = [orig_to_tok_index[_id] for _id in relation["e2_ids"]]
                # e1_ids, e2_ids = [], []
                # for id in relation["e1_ids"]:
                #     e1_ids.extend(orig_to_tok_index[id])
                # for id in relation["e2_ids"]:
                #     e2_ids.extend(orig_to_tok_index[id])
        #     relation["e1_ids"] = e1_ids
        #     relation["e2_ids"] = e2_ids

        input_features = []
        label_ids = []
        label_ids.append(label_map["[CLS]"])
        label_ids.extend([label_map[l] for l in labels])
        label_ids.append(label_map["[SEP]"])
        label_mask = [1] * len(label_ids)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        for q_idx, query in enumerate(example.question_text):
            query_tokens = tokenizer.tokenize(query)
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[: max_query_length]
            max_doc_length = max_seq_length - len(query_tokens) - 3
            if len(doc_tokens) >= max_doc_length:
                doc_tokens = doc_tokens[0:max_doc_length]
                labels = labels[0:max_doc_length]
                doc_valid = doc_valid[0:max_doc_length]
                label_mask = label_mask[0:max_doc_length]

            ntokens = []
            segment_ids = []
            valid = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.append(1) # [CLS] is valid

            # add question_tokens
            for i, token in enumerate(query_tokens):
                ntokens.append(token)
                segment_ids.append(0)  # sentence A
                valid.append(0)
                # if len(labels) > i:
                #     label_ids.append(label_map[labels[i]])
            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(0)
            # add doc tokens
            for i, token in enumerate(doc_tokens):
                ntokens.append(token)
                segment_ids.append(1) # sentence B
                valid.append(doc_valid[i])
                # if len(labels) > i:
                #     label_ids.append(label_map[labels[i]])
            ntokens.append("[SEP]")
            segment_ids.append(1)
            valid.append(1)
            # label_mask.append(1) # attention_mask_label
            # label_ids.append(label_map["[SEP]"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            # label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
                # label_ids.append(0)
                # label_mask.append(0)
            # while len(label_ids) < max_seq_length:
            #     label_ids.append(0)
            #     label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            input_features.append(
                InputFeature(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_ids,
                        label_mask=label_mask,
                        valid_id=valid
                    )
            )

        group_feature = GroupFeature( # 一个doc的某一种问题，对应多个不同表达方式但意思相同的问题
            doc_tokens=example.doc_tokens,
            q_type=example.q_type,
            entity_type=example.entity_type,
            relations=example.relations,
            input_features=input_features
        )
        features.append(group_feature)
    return features


def _divide_group(example_list, n_groups):
    grouped_list = []
    for i in range(0, len(example_list), n_groups):
        grouped_list.append(example_list[i:i + n_groups])
    return grouped_list


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def iob2_to_iobes(tags):
    """
    checked
    IOB2 -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B', 'S'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I', 'E'))
        else:
            raise Exception('Invalid IOB format!')

    assert len(new_tags) == len(tags)

    return new_tags