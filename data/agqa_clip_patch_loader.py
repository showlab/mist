# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json

import clip

sys.path.insert(0, '../')
from util import tokenize, transform_bb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
# from tools.object_align import align
import os.path as osp
import h5py
import random as rd
import numpy as np


# from IPython.core.debugger import Pdb
# dbg = Pdb()
# dbg.set_trace()


class VideoQADataset(Dataset):
    def __init__(
            self,
            data_dir='../data/datasets/agqa/',
            split='train',
            feature_dir='../data/feats/agqa/',
            qmax_words=20,
            amax_words=5,
            bert_tokenizer=None,
            a2id=None
    ):
        """

        """
        self.data = pd.read_json(osp.join(data_dir, f'agqa_{split}.jsonl'))
        self.frame_size = json.load(open(osp.join(data_dir, 'agqa_frame_size.json')))
        self.dset = 'agqa'
        self.video_feature_path = feature_dir
        self.bbox_num = 16
        self.use_frame = True
        self.use_mot = False
        self.qmax_words = qmax_words
        self.a2id = json.load(open(osp.join(data_dir, 'vocab.json')))
        self.bert_tokenizer = bert_tokenizer
        self.candidate_answer = json.load(open(osp.join(data_dir, f'agqa_{split}_candidates.json')))

        # bbox_feat_file = osp.join(self.video_feature_path, f'region_feat_n/faster_rcnn_32f20b.h5')
        # print('Load {}...'.format(bbox_feat_file))

        app_feat_file = osp.join(self.video_feature_path, f'frame_feat/clip_patch_feat_all.h5')
        print('Load {}...'.format(app_feat_file))
        encoding = 'utf-8'
        self.frame_feats = {}
        with h5py.File(app_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['features']
            print(feats.shape)  # v_num, clip_num, feat_dim
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                vid = vid.decode(encoding)
                self.frame_feats[vid] = feat

    def __len__(self):
        return len(self.data)

    def get_video_feature(self, raw_vid_id, width=1, height=1):
        """
        :param vid_id:
        :param width:
        :param height:
        :return:
        """
        vid_id = raw_vid_id if raw_vid_id in self.frame_feats else raw_vid_id.strip('.mp4')

        # generate bbox coordinates of patches
        patch_bbox = []
        patch_size = 224
        grid_num = 4
        width, height = patch_size * grid_num, patch_size * grid_num

        for j in range(grid_num):
            for i in range(grid_num):
                patch_bbox.append([i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size])
        roi_bbox = np.tile(np.array(patch_bbox), (32, 1)).reshape(32, 16, -1)  # [frame_num, bbox_num, -1]
        bbox_feat = transform_bb(roi_bbox, width, height)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)

        try:
            roi_feat = self.frame_feats[vid_id][:, 1:, :]  # [frame_num, 16, dim]
            roi_feat = torch.from_numpy(roi_feat).type(torch.float32)
            region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)
        except:
            from IPython.core.debugger import Pdb
            dbg = Pdb()
            dbg.set_trace()

        # vid_id = raw_vid_id if raw_vid_id in self.frame_feats else raw_vid_id.strip('.mp4')

        frame_feat = self.frame_feats[vid_id][:, 0, :]
        frame_feat = torch.from_numpy(frame_feat).type(torch.float32)

        # print('Sampled feat: {}'.format(region_feat.shape))
        return region_feat, frame_feat

    def __getitem__(self, index):
        cur_sample = self.data.iloc[index]
        raw_vid_id = str(cur_sample["video_id"])
        qid = str(cur_sample['question_id'])

        vid_id = raw_vid_id.strip(".mp4")
        frame_size = self.frame_size[vid_id]
        width, height = frame_size['width'], frame_size['height']

        video_o, video_f = self.get_video_feature(raw_vid_id, width, height)

        vid_duration = 8  # video_f.shape[0]

        question_txt = cur_sample['question']
        question_embd = torch.tensor(
            self.bert_tokenizer.encode(
                question_txt,
                add_special_tokens=True,
                padding="longest",
                max_length=self.qmax_words,
                truncation=True,
            ),
            dtype=torch.long,
        )

        question_clip = clip.tokenize(question_txt)

        answer_txts = cur_sample["answer"]
        answer_id = self.a2id.get(answer_txts, -1)

        type = cur_sample['answer_type']
        seq_len = 0

        return {
            "video_id": raw_vid_id,
            "video": (video_o, video_f),
            # "video_f": video_f,
            "video_len": vid_duration,
            "question": question_embd,
            "question_clip": question_clip,
            "question_txt": question_txt,
            "type": type,
            "answer_id": answer_id,
            "answer_txt": answer_txts,
            "answer": answer_id,
            "seq_len": seq_len,
            "question_id": qid
        }


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))

    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )

    return default_collate(batch)


def get_videoqa_loaders(args, features, a2id, bert_tokenizer, test_mode):
    data_dir = os.path.join(args.dataset_dir, args.dataset)
    if test_mode:
        test_dataset = VideoQADataset(
            data_dir=data_dir,
            split='test',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
        )
        train_loader, val_loader = None, None
    else:

        train_dataset = VideoQADataset(
            data_dir=data_dir,
            split='train',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=True,
            drop_last=True,
            collate_fn=videoqa_collate_fn,
        )
        if args.dataset.split('/')[0] in ['tgifqa', 'tgifqa2', 'msrvttmc']:
            args.val_csv_path = args.test_csv_path
        val_dataset = VideoQADataset(
            data_dir=data_dir,
            split='val',
            feature_dir=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            collate_fn=videoqa_collate_fn,
        )
        test_loader = None

    return train_loader, val_loader, test_loader
