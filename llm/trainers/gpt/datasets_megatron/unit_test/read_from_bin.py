from llm.trainers.gpt.datasets_megatron.core_class.indexed_dataset import MMapIndexedDataset
from llm.trainers.gpt.datasets_megatron.core_class.gpt_dataset import GPTDataset, GPTDatasetConfig
from transformers import AutoTokenizer
from llm.trainers.gpt.datasets_megatron.core_class.utils import Split
from llm.trainers.gpt.datasets_megatron.core_class.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from llm.trainers.gpt.datasets_megatron.core_class.utils import compile_helpers
import os
import numpy
from torch.utils.data import DataLoader
from transformers import default_data_collator

def is_dataset_built_on_rank():
    return True

def main():
    processed_data = '/N/scratch/jindjia/thepile/pile_text_document'
    indexed_dataset = MMapIndexedDataset(os.path.join(processed_data))
    idx = 0
    toks = indexed_dataset

    num_elements = indexed_dataset.sequence_lengths.shape[0]
    print(f'datasets has {num_elements} rows:')

    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    sequence_length = 1024
    dataset_config = GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=1234,
        sequence_length=sequence_length,
        blend=processed_data,
        blend_per_split=None,
        split='969, 30, 1',
        path_to_cache='/N/slate/jindjia/bash_scripts/baixi/cache',
        return_document_ids=False,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        eod_id=tokenizer("<|endoftext|>")["input_ids"][0]
    )
    
    num_elements = indexed_dataset.sequence_lengths.shape[0]
    print('num_elements:',num_elements)
    split = [(0, 0.969), (0.969, 0.999), (0.999, 1.0)]
    split_indices = []
    for i in range(len(split)):
        if split[i] is not None:
            beg = int(round(split[i][0] * float(num_elements)))
            end = int(round(split[i][1] * float(num_elements)))
            split_indices.append(
                numpy.arange(start=beg, stop=end, step=1, dtype=numpy.int32)
            )
        else:
            split_indices.append(None)

    # print(f'split[0] elements len: {len(split_indices[0])} num tokens per epoch: {int(numpy.sum(indexed_dataset.sequence_lengths[split_indices[0]]))}')
    train_ds = GPTDataset(indexed_dataset, split_indices[0], len(split_indices[0])//sequence_length , Split.train, dataset_config)
    print(f'train_ds len: {len(train_ds)}')

def pretrain_dataset():
    compile_helpers()
    # tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')
    sequence_length = 1024
    data_path = ['/N/scratch/jindjia/thepile/pile_text_document']
    dataset_config = GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=1234,
        sequence_length=sequence_length,
        blend=data_path,
        blend_per_split=None,
        split='969, 30, 1',
        path_to_cache='/N/slate/jindjia/bash_scripts/baixi/cache',
        return_document_ids=False,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        eod_id=tokenizer("<|endoftext|>")["input_ids"][0]
    )



    # train_samples = train_iters * global_batch_size
    # eval_iters = (train_iters // eval_interval + 1) * \
    #              eval_iters
    # test_iters = eval_iters
    # train_val_test_num_samples = [train_samples,
    #                               eval_iters * global_batch_size,
    #                               test_iters * global_batch_size]
    train_val_test_num_samples = [1000,
                                  10,
                                  1]
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        dataset_config
    ).build()

    train_dataloader = DataLoader(
        train_ds,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=1,
    )
    eval_dataloader = DataLoader(
        valid_ds,
        collate_fn=default_data_collator,
        batch_size=2,
    )
    print(f'train dataload len: {len(train_dataloader)}')
    for step, batch in enumerate(train_dataloader):
        if step >= 1:
            break
        print(batch)
        print(tokenizer.decode(batch['input_ids'][0]))

if __name__ == "__main__":
    # main()
    pretrain_dataset()