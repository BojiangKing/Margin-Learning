# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-11 02:47
from hanlp.common.dataset import SortingSamplerBuilder
from hanlp.components.tokenizers.transformer import TransformerTaggingTokenizer
from hanlp.datasets.cws.sighan2005.pku import SIGHAN2005_PKU_TRAIN_ALL, SIGHAN2005_PKU_TEST
from hanlp.datasets.ner.msra import MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_TRAIN, MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_DEV, \
    MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_TEST
from tests import cdroot

cdroot()
tokenizer = TransformerTaggingTokenizer()
save_dir = 'data/model/cws/sighan2005_pku_bert_base_96.70'
tokenizer.fit(
    SIGHAN2005_PKU_TRAIN_ALL,
    SIGHAN2005_PKU_TEST,  # Conventionally, no devset is used. See Tian et al. (2020).
    save_dir,
    'bert-base-chinese',
    max_seq_len=300,
    char_level=True,
    hard_constraint=True,
    sampler_builder=SortingSamplerBuilder(batch_size=32),
    epochs=3,
    adam_epsilon=1e-6,
    warmup_steps=0.1,
    weight_decay=0.01,
    word_dropout=0.1,
    seed=1609836303,
)
tokenizer.evaluate(SIGHAN2005_PKU_TEST, save_dir)
print(f'Model saved in {save_dir}')
